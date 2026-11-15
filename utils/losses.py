def bce(x, x_hat):
    loss = -(x * (x_hat + 1e-8).log() + (1 - x) * (1 - x_hat + 1e-8).log()).sum()
    return loss


def kl_divergence(mean, logvar):
    loss = -0.5 * (1 + logvar - mean**2 - logvar.exp()).sum()
    return loss


def vae_loss(x, x_hat, mean, logvar, beta=1.0):
    return bce(x, x_hat) + beta * kl_divergence(mean, logvar)
