import lightning.pytorch as pl
import torch
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils.utils import encode_labels
from models.cbegan_architecture import GeneratorModel, DiscriminatorModel


class SpectralCBEGAN(pl.LightningModule):
    def __init__(
        self,
        generator_params,
        discriminator_params,
        max_epochs,
        noise_dim,
        initial_lr,
        eta_min,
        lambda_k,
        gamma,
        pos_encoding_embedding_dim=20,
        only_disc_epochs=10,
    ):
        super(SpectralCBEGAN, self).__init__()
        self.save_hyperparameters(ignore=["generator", "discriminator"])

        self.generator = GeneratorModel(**generator_params)
        self.discriminator = DiscriminatorModel(**discriminator_params)

        # Store hyperparameters explicitly if not relying solely on self.hparams
        self.max_epochs = max_epochs
        self.noise_dim = noise_dim
        self.initial_lr = initial_lr
        self.eta_min = eta_min
        self.lambda_k = lambda_k
        self.gamma = gamma
        self.only_disc_epochs = only_disc_epochs
        self.pos_encoding_embedding_size = pos_encoding_embedding_dim

        # Important: Use manual optimization for GANs
        self.automatic_optimization = False

        # Initialize BEGAN specific parameters
        self.k_t = 0.0

    def forward(self, noise, conditions):
        return self.generator(noise, conditions)

    @staticmethod
    def compute_rec_loss(x, d_output):
        # L1 Loss (Mean Absolute Error)
        return torch.mean(torch.abs(x - d_output))

    def training_step(self, batch, batch_idx):
        # Get optimizers
        opt_d, opt_g = self.optimizers()

        # --- Prepare Data ---
        real_spectra = batch["spectrum"].to(self.device)
        label_dict = batch["labels"]  # Assuming this is a dict of tensors

        # Process conditions (positional encoding)
        conditions = encode_labels(
            label_dict,
            embedding_size=self.pos_encoding_embedding_size,
            device=self.device,
        )

        # Generate noise and fake data
        batch_size = real_spectra.size(0)
        noise = torch.randn(batch_size, self.noise_dim, device=self.device)
        # We need G_output for both D and G training
        G_output = self.generator(noise, conditions)

        # --- Train Discriminator ---
        self.toggle_optimizer(opt_d)

        D_output_fake = self.discriminator(G_output.detach(), conditions)
        D_output_real = self.discriminator(real_spectra, conditions)

        rec_loss_real = self.compute_rec_loss(real_spectra, D_output_real)
        # Use G_output.detach() for calculating fake loss for D step
        rec_loss_fake_D = self.compute_rec_loss(G_output.detach(), D_output_fake)

        # BEGAN Discriminator Loss
        D_loss = rec_loss_real - self.k_t * rec_loss_fake_D

        opt_d.zero_grad()
        self.manual_backward(D_loss)
        opt_d.step()
        self.untoggle_optimizer(opt_d)

        # Log Discriminator losses (use sync_dist=True if using DDP)
        self.log(
            "train/D_loss",
            D_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "train/rec_loss_real",
            rec_loss_real,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "train/rec_loss_fake_D",
            rec_loss_fake_D,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )

        # --- Train Generator ---
        # Determine if we should train the generator in this epoch
        train_G = self.current_epoch > self.only_disc_epochs

        if train_G:
            self.toggle_optimizer(opt_g)

            # We need grads through G now, so use G_output directly (not detached)
            D_output_fake_for_G = self.discriminator(G_output, conditions)
            G_loss = self.compute_rec_loss(G_output, D_output_fake_for_G)

            opt_g.zero_grad()
            self.manual_backward(G_loss)
            opt_g.step()
            self.untoggle_optimizer(opt_g)

            # Log Generator loss (use sync_dist=True if using DDP)
            self.log(
                "train/G_loss",
                G_loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )
            # G_loss is the one used for balance when G is trained
            balance_loss_G = G_loss
        else:
            # Log G_loss as 0 or NaN when not training
            G_loss = torch.tensor(0.0, device=self.device)  # Or float('nan')
            self.log(
                "train/G_loss",
                G_loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )
            # Use the detached fake reconstruction loss for balance when G is not trained
            balance_loss_G = rec_loss_fake_D

        # --- Update k_t and Convergence Measure (Per Batch) ---
        # Use .item() to get Python floats and detach from graph
        rec_loss_real_item = rec_loss_real.item()
        balance_loss_G_item = (
            balance_loss_G.item()
        )  # Already detached if G_loss=0, item() detaches otherwise

        balance = self.gamma * rec_loss_real_item - balance_loss_G_item

        # Update k_t
        self.k_t += self.lambda_k * balance
        self.k_t = min(max(self.k_t, 0.0), 1.0)  # Clamp k_t to [0, 1]

        # Calculate convergence measure M
        convergence_measure = rec_loss_real_item + abs(balance)

        # Log k_t and convergence measure (use sync_dist=True if using DDP)
        # Log step value and epoch average for k_t
        self.log(
            "train/k_t",
            self.k_t,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )
        # Log step value and epoch average for convergence, show epoch average on prog bar
        self.log(
            "train/convergence_measure",
            convergence_measure,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

    def configure_optimizers(self):
        optimizerG = optim.Adam(self.generator.parameters(), lr=self.initial_lr)
        optimizerD = optim.Adam(self.discriminator.parameters(), lr=self.initial_lr)

        schedulerG = CosineAnnealingLR(
            optimizerG, T_max=self.max_epochs, eta_min=self.eta_min
        )
        schedulerD = CosineAnnealingLR(
            optimizerD, T_max=self.max_epochs, eta_min=self.eta_min
        )

        return (
            [optimizerD, optimizerG],
            [
                {"scheduler": schedulerD, "interval": "epoch", "name": "lr/disc"},
                {"scheduler": schedulerG, "interval": "epoch", "name": "lr/gen"},
            ],
        )
