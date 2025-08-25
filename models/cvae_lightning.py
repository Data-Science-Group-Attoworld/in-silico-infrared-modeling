import torch
import lightning.pytorch as pl
from models.cvae_model import CondVAE
from losses.losses import VAELoss
from utils.annealer import Annealer


class LitVAE(pl.LightningModule):
    def __init__(
        self,
        vae_params,
        annealer_params,
        datamodule,
        learning_rate,
        optimizer,
        rec_weight,
    ):
        super(LitVAE, self).__init__()

        """
        vae: pytorch model VAE
        loss (str): either "bvae"
        """
        # save hyperparams to log directory
        self.save_hyperparameters()

        # general setup
        self.vae = CondVAE(**vae_params)
        self.dm = datamodule
        self.lr = learning_rate
        self.optimizer = optimizer

        # Weights for loss function
        self.rec_weight = rec_weight
        self.objective = VAELoss()

        self.cyclical_annealing = annealer_params["cyclical_annealing"]
        if self.cyclical_annealing:
            self.annealer = Annealer(**annealer_params)
        else:
            self.annealer = None

    def training_step(self, batch, batch_idx):
        """
        perform any combination of: conditional or normal vae, bvae loss or tcvae loss
        """

        data = batch["spectrum"]  # contains samples
        labels = batch["labels"]  # contains age, sex, bmi labels

        _, decoded, z_samples, mu, logvar = self.vae(x=data, labels=labels)
        loss = self.train_step_calculate_and_log_bvae_loss(data, decoded, mu, logvar)

        return loss

    def on_train_epoch_end(self):

        # cyclical annealing step after each epoch
        if self.annealer is not None:
            self.annealer.step()

    def configure_optimizers(self):
        optimizers = {
            "Adam": torch.optim.Adam(self.parameters(), lr=self.lr),
            "AdamW": torch.optim.AdamW(self.parameters(), lr=self.lr),
            "Nadam": torch.optim.NAdam(self.parameters(), lr=self.lr),
            "SGD": torch.optim.SGD(self.parameters(), lr=self.lr),
        }
        return optimizers[self.optimizer]

    def train_step_calculate_and_log_bvae_loss(self, data, decoded, mu, logvar):
        loss_rec, loss_kld = self.objective(decoded, data, mu=mu, logvar=logvar)

        self.log("rec", loss_rec, on_step=False, on_epoch=True, prog_bar=True)
        self.log("kld", loss_kld, on_step=False, on_epoch=True, prog_bar=True)

        if self.cyclical_annealing:
            beta = self.annealer.get_beta()
            loss = self.rec_weight * loss_rec + beta * loss_kld
            self.log("beta", beta, on_step=False, on_epoch=True, prog_bar=True)

        else:
            loss = self.rec_weight * loss_rec + loss_kld

        self.log("loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss
