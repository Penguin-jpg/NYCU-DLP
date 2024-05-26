import torch
import torch.nn.functional as F
from tqdm import tqdm


def extract(value, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    # gather is used to retrieve tensor at index of t
    out = value.gather(dim=-1, index=t)
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


# schedule for betas
def linear_schedule(timesteps):
    start_beta = 1e-4
    end_beta = 0.02
    return torch.linspace(start_beta, end_beta, timesteps)


class Diffusion:
    def __init__(
        self,
        diffusion_steps,
        sampling_steps,
        image_shape,
        use_ddim=False,
        eta=0.0,
        device=None,
    ):
        self.diffusion_steps = diffusion_steps
        self.sampling_steps = sampling_steps
        self.image_shape = image_shape
        self.use_ddim = use_ddim
        self.betas = linear_schedule(diffusion_steps).to(device)
        self.alphas = 1.0 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha = torch.sqrt(self.alphas)
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - self.alpha_bar)
        self.sqrt_recip_alpha_bar = torch.sqrt(1.0 / self.alpha_bar)
        self.sqrt_recip_alpha_bar_minus_one = torch.sqrt(1.0 / self.alpha_bar - 1.0)
        # this is for sqrt{alpha_bar_{t-1}}, pad to left so that we can get t-1 with t
        self.alpha_bar_prev = F.pad(self.alpha_bar[:-1], (1, 0), value=1.0)

        # variables for calculating the posterior q(x_{t-1}|x_t, x_0)
        # varaiance is beta_t * (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)
        self.posterior_variance = (
            self.betas * (1.0 - self.alpha_bar_prev) / (1.0 - self.alpha_bar)
        )
        # coefficient for x_0 is sqrt{alpha_bar_{t-1}} * beta_t / (1.0 - alpha_bar_t)
        self.posterior_mean_x_0_coef = (
            torch.sqrt(self.alpha_bar_prev) * self.betas / (1.0 - self.alpha_bar)
        )
        # coefficient for x_t is sqrt{alpha_t} * (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)
        self.posterior_mean_x_t_coef = (
            self.sqrt_alpha * (1.0 - self.alpha_bar_prev) / (1.0 - self.alpha_bar)
        )
        # log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        # from https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/diffusion_utils.py#L82
        self.posterior_log_variance = torch.log(
            self.posterior_variance.clamp(min=1e-20)
        )
        # coefficient for sigma
        self.eta = eta

        self.device = device

    def forward_process(self, x_0, t):
        # find mean and std of the distribution q(x_t|x_0) and use the formula
        # mean is sqrt{alpha_bar_t} * x_0
        # std is sqrt{1 - alpha_bar_t} * noise
        noise = torch.randn_like(x_0, device=self.device)
        sqrt_alpha_bar_t = extract(self.sqrt_alpha_bar, t, x_0.shape)
        sqrt_one_minus_alpha_bar_t = extract(
            self.sqrt_one_minus_alpha_bar, t, x_0.shape
        )
        return sqrt_alpha_bar_t * x_0 + sqrt_one_minus_alpha_bar_t * noise, noise

    def predict_x_0_from_noise(self, x_t, t, noise):
        # x_0 = sqrt{1 / alpha_bar_t} * (x_t - sqrt{1 - alpha_bar_t} * noise)
        sqrt_recip_alpha_bar_t = extract(self.sqrt_recip_alpha_bar, t, x_t.shape)
        sqrt_recip_alpha_bar_minus_one_t = extract(
            self.sqrt_recip_alpha_bar_minus_one, t, x_t.shape
        )
        return sqrt_recip_alpha_bar_t * x_t - sqrt_recip_alpha_bar_minus_one_t * noise

    def predict_noise_from_x_0(self, x_t, t, x_0):
        # rearrange the formula from predict_x_0_from_noise
        sqrt_recip_alpha_bar_t = extract(self.sqrt_recip_alpha_bar, t, x_t.shape)
        sqrt_recip_alpha_bar_minus_one_t = extract(
            self.sqrt_recip_alpha_bar_minus_one, t, x_t.shape
        )
        return (sqrt_recip_alpha_bar_t * x_t - x_0) / sqrt_recip_alpha_bar_minus_one_t

    def model_mean_variance(self, x_0, x_t, t):
        # tilde{u}_t(x_t, x_0)
        posterior_mean = (
            extract(self.posterior_mean_x_0_coef, t, x_t.shape) * x_0
            + extract(self.posterior_mean_x_t_coef, t, x_t.shape) * x_t
        )
        # tilde{beta}_t
        # log variance for better result
        posterior_log_variance = extract(self.posterior_log_variance, t, x_t.shape)
        return posterior_mean, posterior_log_variance

    # use inference_mode instead of no_grad because inference_mode is better for error detection
    @torch.inference_mode()
    def ddpm_sample(self, model, x_t, t, labels):
        B = x_t.shape[0]
        timestep_tensor = torch.full((B,), t, dtype=torch.long, device=self.device)
        noise = torch.randn_like(x_t, device=self.device) if t > 0 else 0.0

        predicted_noise = model(x_t, timestep_tensor, labels)
        predicted_x_0 = self.predict_x_0_from_noise(
            x_t, timestep_tensor, predicted_noise
        ).clamp(-1.0, 1.0)

        # use predicted x_0 to get model_mean and variance
        model_mean, model_log_variance = self.model_mean_variance(
            x_0=predicted_x_0, x_t=x_t, t=timestep_tensor
        )

        # x_{t-1} = model_mean (posterior mean) + torch.sqrt(model_variance) (posterior variance) * noise
        x_prev = model_mean + torch.exp(0.5 * model_log_variance) * noise

        return x_prev, predicted_x_0

    @torch.inference_mode()
    def ddpm_reverse(self, model, labels, num_samples=16):
        x = torch.randn([num_samples, *self.image_shape], device=self.device)
        images = []
        for t in tqdm(
            reversed(range(0, self.diffusion_steps)),
            desc="DDPM Sampling",
            colour="green",
            total=self.diffusion_steps,
        ):
            x, image = self.ddpm_sample(model, x, t, labels)
            images.append(image)

        return x, images

    @torch.inference_mode()
    def ddim_sample(self, model, x_t, t, t_prev, labels):
        # to sample a step in DDIM, we need:
        # 1. predicted x_0
        # 2. direction points to x_t
        # 3. random noise (don't need this if eta is 0)
        # apply the formula: x_{t-1} = sqrt{alpha_bar_{t-1}} * predicted_x_0 + direction_to_x_t + random noise

        B = x_t.shape[0]
        timestep_tensor = torch.full((B,), t, dtype=torch.long, device=self.device)
        prev_timestep_tensor = torch.full(
            (B,), t_prev, dtype=torch.long, device=self.device
        )
        noise = torch.randn_like(x_t, device=self.device)

        predicted_noise = model(x_t, timestep_tensor, labels)
        predicted_x_0 = self.predict_x_0_from_noise(
            x_t, timestep_tensor, predicted_noise
        ).clamp(-1.0, 1.0)

        # in ddim, we need to re-derive noise from predicted x_0
        predicted_noise = self.predict_noise_from_x_0(
            x_t, timestep_tensor, x_0=predicted_x_0
        )

        alpha_bar_t = extract(self.alpha_bar, timestep_tensor, x_t.shape)
        alpha_bar_prev = extract(self.alpha_bar, prev_timestep_tensor, x_t.shape)
        sigma_t = (
            self.eta
            * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar_t))
            * torch.sqrt(1 - alpha_bar_t / alpha_bar_prev)
        )
        # coefficient of direction pointing to x_t
        direction_coef = torch.sqrt(1 - alpha_bar_prev - sigma_t**2)

        x_prev = (
            torch.sqrt(alpha_bar_prev) * predicted_x_0
            + direction_coef * predicted_noise
            + sigma_t * noise
        )

        return x_prev, predicted_x_0

    @torch.inference_mode()
    def ddim_reverse(self, model, labels, num_samples=16):
        x = torch.randn([num_samples, *self.image_shape], device=self.device)
        images = []

        # equally split timesteps for sampling, for example,
        # 1000 diffusion steps with 250 sampling steps means that timesteps are [999, 995, 991, ...]
        # use -1 as the stop signal to avoid index out of range
        timesteps = torch.linspace(
            -1, self.diffusion_steps - 1, steps=self.sampling_steps + 1
        )
        timesteps = list(reversed(timesteps.int().tolist()))
        # pair of (current timestep, previous timestep)
        timestep_pairs = list(zip(timesteps[:-1], timesteps[1:]))

        for t, t_prev in tqdm(timestep_pairs, desc="DDIM Sampling", colour="green"):
            if t_prev < 0:
                continue
            x, image = self.ddim_sample(model, x, t, t_prev, labels)
            images.append(image)

        return x, images

    @torch.inference_mode()
    def sample(self, model, labels, num_samples=16):
        if self.use_ddim:
            return self.ddim_reverse(model, labels, num_samples)
        else:
            return self.ddpm_reverse(model, labels, num_samples)
