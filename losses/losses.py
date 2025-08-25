import torch


class VAELoss(torch.nn.Module):
    """
    Calculates reconstruction loss and KL divergence loss for VAE.
    """

    def __init__(self):
        super(VAELoss, self).__init__()
        self.criterion = torch.nn.MSELoss(reduction="mean")

    def kld_loss(self, mu, logvar):
        return -0.5 * torch.mean(1 + logvar - mu**2 - logvar.exp())

    def forward(self, x, x0, mu, logvar):
        """
        Args:
            x (torch.Tensor): reconstructed input tensor
            x0 (torch.Tensor): original input tensor
            mu (torch.Tensor): latent space mu
            logvar (torch.Tensor): latent space log variance
        Returns:
            rec (torch.Tensor): Root Mean Squared Error (VAE recon loss)
            kld (torch.Tensor): KL divergence loss
        """
        rec = self.criterion(x0, x)
        kld = self.kld_loss(
            mu, logvar
        )  # -0.5 * torch.mean(1 + logvar - mu ** 2 - logvar.exp())

        return rec, kld
