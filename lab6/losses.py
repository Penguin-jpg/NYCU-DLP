import torch.nn.functional as F

# modified from https://github.com/leo27945875/MHingeGAN-for-multi-label-conditional-generation


# this loss is multi-class hinge loss for auxiliary classifier from
# https://openaccess.thecvf.com/content/WACV2021/papers/Kavalerov_A_Multi-Class_Hinge_Loss_for_Conditional_GANs_WACV_2021_paper.pdf
# but since our task is multi-label conditional generation, we need to make some adjustments
def aux_loss(logits, label):
    # we first convert labels to -1 or 1
    condition = label * 2 - 1

    # the original multi-class hinge loss calculates E[max(0, 1 - C_y(x) + C_{~y}(x))] (x = G(z, y) for generator)
    # but multiply labels in -1 or 1, we can get the same effect but can apply on multi-label
    return F.relu(1 - logits * condition).sum(dim=-1).mean()


def generator_loss(fake, fake_logits, labels, aux_weight=1.0):
    # fake is the prediction of discriminator on fake images
    # fake_logits is the prediction of auxiliary classifier on fake images

    # as in the original paper, the loss of generator is -E[D(G(z, y), y)] + lambda * aux_loss(C(G(z, y)), y)
    g_loss = -fake.mean()
    return g_loss + aux_weight * aux_loss(fake_logits, labels)


def discriminator_loss(real, fake, real_logits, labels):
    # real is the prediction of discriminator on real images
    # real_logits is the prediction of auxiliary classifier on real images

    # as in the original paper, the loss of discriminator is E[max(0, 1 - D(x, y))] + E[max(0, 1 + D(G(z, y), y))] + aux_loss(C(x), y)
    d_loss = F.relu(1 - real).mean() + F.relu(1 + fake).mean()
    return d_loss + aux_loss(real_logits, labels)


def mse_loss(predicted, ground_truth):
    return F.mse_loss(predicted, ground_truth, reduction="mean")
