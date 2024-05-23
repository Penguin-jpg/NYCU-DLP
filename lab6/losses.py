import torch
import torch.nn.functional as F


# GAN losses
def hinge_loss(predicted, label):
    condition = label * 2 - 1
    return F.relu(1 - predicted * condition).sum(dim=-1).mean()


def generator_loss(fake, fake_logits, label, aux_weight=1.0):
    return -fake.mean() + aux_weight * hinge_loss(fake_logits, label)


def discriminator_loss(real, fake, real_logits, label):
    loss = (F.relu(1 - real) + F.relu(1 + fake)).mean()
    drift = (real**2).mean()
    return loss + hinge_loss(real_logits, label) + 0.001 * drift
