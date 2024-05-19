import torch
import torch.nn as nn
import yaml
import os
import math
import numpy as np
from .VQGAN import VQGAN
from .Transformer import BidirectionalTransformer


# TODO2 step1: design the MaskGIT model
class MaskGit(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.vqgan = self.load_vqgan(configs["VQ_Configs"])

        self.num_image_tokens = configs["num_image_tokens"]
        self.mask_token_id = configs["num_codebook_vectors"]
        self.choice_temperature = configs["choice_temperature"]
        self.gamma_type = configs["gamma_type"]
        self.gamma = self.gamma_func(mode=configs["gamma_type"])
        self.transformer = BidirectionalTransformer(configs["Transformer_param"])

    def load_transformer_checkpoint(self, load_ckpt_path):
        self.transformer.load_state_dict(torch.load(load_ckpt_path))

    @staticmethod
    def load_vqgan(configs):
        cfg = yaml.safe_load(open(configs["VQ_config_path"], "r"))
        model = VQGAN(cfg["model_param"])
        model.load_state_dict(torch.load(configs["VQ_CKPT_path"]), strict=True)
        model = model.eval()
        return model

    ##TODO2 step1-1: input x fed to vqgan encoder to get the latent and zq
    @torch.no_grad()
    def encode_to_z(self, x):
        z_q, z_indices, _ = self.vqgan.encode(x)
        # reshape z_indices to [batch_size, num_image_tokens]
        z_indices = z_indices.view(z_q.shape[0], self.num_image_tokens)
        return z_indices

    ##TODO2 step1-2:
    def gamma_func(self, ratio=None, mode="cosine"):
        """Generates a mask rate by scheduling mask functions R.

        Given a ratio in [0, 1), we generate a masking ratio from (0, 1].
        During training, the input ratio is uniformly sampled;
        during inference, the input ratio is based on the step number divided by the total iteration number: t/T.
        Based on experiements, we find that masking more in training helps.

        ratio:   The uniformly sampled ratio [0, 1) as input.
        Returns: The mask rate (float).

        """

        if ratio is None:
            ratio = np.random.uniform(0, 1, size=1)

        if mode == "linear":
            return 1 - ratio
        elif mode == "cosine":
            # only within the range of 0 ~ 0.5pi since 0.5pi ~ pi is negative
            return math.cos(ratio * math.pi * 0.5)
        elif mode == "square":
            return 1 - ratio**2
        else:
            raise NotImplementedError

        return ratio

    ##TODO2 step1-3:
    def forward(self, x):
        device = x.get_device()

        # ground truth
        z_indices = self.encode_to_z(x).to(device)

        # the number of masked tokens = gamma * num_image_tokens
        num_masked_tokens = math.ceil(
            self.gamma_func(mode=self.gamma_type) * self.num_image_tokens
        )

        # create mask based on the indices
        mask = torch.zeros(z_indices.shape, dtype=torch.bool, device=device)
        for i in range(mask.shape[0]):
            masked_indices = torch.from_numpy(
                # randomly select num_masked_tokens indices to be masked
                np.random.choice(
                    np.arange(self.num_image_tokens),
                    size=num_masked_tokens,
                    replace=False,
                )
            ).to(device)
            # True means masked
            mask[i][masked_indices] = True

        # apply masking
        masked_z_indices = mask * self.mask_token_id + ~mask * z_indices

        # use transformer to predict the probability of tokens
        logits = self.transformer(masked_z_indices)

        return logits, z_indices

    ##TODO3 step1-1: define one iteration decoding
    @torch.no_grad()
    def inpainting(
        self,
        indices,
        mask,
        num_masked_tokens,  # this is fixed for each iteration
        current_iteration,
        total_iteration,
        mask_type="cosine",
    ):
        # to perform interative decoding at iteration t, we need to do the following steps:
        # 1. predict the probability of each token given the masked indices
        # 2. sample tokens from codebook based on the predicted probability
        # 3. decide how many tokens to mask in this iteration: n=[gamma(t/T) * N]
        #    - n: the number of mased tokens in this iteration t+1
        #    - t: current iteration,
        #    - T: total iteration
        #    - N: total number of masked tokens
        # 4. mask all tokens with confidence lower than the n-th lowest confidence

        device = indices.get_device()

        # mask indices according to the given mask
        masked_indices = indices
        masked_indices[mask] = self.mask_token_id

        # predict the probability of each token
        logits = self.transformer(masked_indices)
        # Apply softmax to convert logits into a probability distribution across the last dimension.
        logits = nn.Softmax(dim=-1)(logits)

        # set the probability of non-masked tokens to infinity, since we do not want to mask them
        # (if it is not inf, its confidence may be too low, and thus make it be masked)
        logits[~mask] = torch.inf

        # FIND MAX probability for each token value
        z_indices_predict_prob, z_indices_predict = logits.max(dim=-1)

        # the ratio is t/T
        ratio = current_iteration / total_iteration
        # predicted probabilities add temperature annealing gumbel noise as confidence
        g = (
            torch.distributions.gumbel.Gumbel(0, 1)
            .sample(z_indices_predict_prob.shape)
            .to(device)
        )  # gumbel noise
        temperature = self.choice_temperature * (1 - ratio)
        # the higher the confidence is, the more likely the token will not be masked
        confidence = z_indices_predict_prob + temperature * g

        # hint: If mask is False, the probability should be set to infinity, so that the tokens are not affected by the transformer's prediction
        # sort the confidence for the rank
        # define how much the iteration remain predicted tokens by mask scheduling
        # At the end of the decoding process, add back the original token values that were not masked to the predicted tokens

        # put original indices to where mask is False
        z_indices_predict[~mask] = indices[~mask]

        # sort confidence (from high to low)
        sorted_confidence, _ = confidence.sort(dim=-1)

        # how many tokens to mask in next iteration (formula in section 3.2.4 of paper)
        num_masked_tokens_next_iteration = math.ceil(
            self.gamma_func(ratio, mode=mask_type) * num_masked_tokens
        )

        # all tokens with confidence lower than the n-th lowest confidence will be masked
        mask_bc = confidence < sorted_confidence[:, num_masked_tokens_next_iteration]
        # if it is the last iteration, just remove the whole mask
        if current_iteration == total_iteration:
            mask_bc = torch.fill(mask_bc, False)

        return z_indices_predict, mask_bc


__MODEL_TYPE__ = {"MaskGit": MaskGit}
