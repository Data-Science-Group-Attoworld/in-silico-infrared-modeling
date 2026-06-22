import torch
import torch.nn as nn
import pytorch_lightning as pl
from utils.utils import encode_labels
from models.cdiff_model import CfgDiffusion
from torch_ema import ExponentialMovingAverage
import numpy as np

class LitCfgDiffusion(pl.LightningModule):
    def __init__(
        self,
        cdiff_params: dict,
        learning_rate: float = 1e-4,
        optimizer: str = "AdamW",
        weight_decay: float = 1e-4,
        lr_scheduler: str = "cosine",   # "cosine" | "none"
        warmup_steps: int = 500,
        ema_decay: float = 0.99,
        n_timesteps: int = 128,
        s_timestep_schedule: float = 0.008,
    ):
        super().__init__()
        self.save_hyperparameters()

        # ── Model ────────────────────────────────────────────────────────────
        self.cdiff = CfgDiffusion(**cdiff_params)
        self.pos_embedding_dim = cdiff_params["pos_embedding_dim"]

        self.ema = ExponentialMovingAverage(self.cdiff.parameters(), decay=ema_decay)

        # cosine schedule
        alpha_bar_full = self.cosine_beta_schedule(T=n_timesteps, s=s_timestep_schedule)    # T+1

        alpha_bar = alpha_bar_full[1:]           # (T,)  — ᾱ_1 … ᾱ_T, used at each diffusion step
        alpha_bar_prev = alpha_bar_full[:-1]     # (T,)  — ᾱ_0 … ᾱ_{T-1}, the "previous" step
        
        betas = (1 - alpha_bar / alpha_bar_prev)
        alphas = 1.0 - betas

        
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("alpha_bar_prev", alpha_bar_prev)
        self.register_buffer("sqrt_alpha_bar",     alpha_bar.sqrt())
        self.register_buffer("sqrt_one_minus_ab",  (1 - alpha_bar).sqrt())


    # ── Diffusion helpers ────────────────────────────────────────────────────

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, eps: torch.Tensor):
        """Forward process: add noise to x0 at timestep t."""
        s1 = self.sqrt_alpha_bar[t].view(-1, 1)        # (B, 1) for broadcasting
        s2 = self.sqrt_one_minus_ab[t].view(-1, 1)
        return s1 * x0 + s2 * eps

    def _drop_mask(self, batch_size: int) -> torch.Tensor:
        """Bernoulli mask: True where labels should be dropped (→ null token)."""
        return torch.rand(batch_size, device=self.device) < self.cdiff.p_uncond

    # ── Core step (shared by train/val) ─────────────────────────────────────

    def _step(self, batch, stage: str):
        x0 = batch["spectrum"].to(self.device)
        label_dict = batch["labels"]

        y = encode_labels(
            label_dict,
            embedding_size=self.pos_embedding_dim,
            device=self.device,
            pe_base=0.1
        )
        
        B = x0.size(0)
        t = torch.randint(0, len(self.betas), (B,), device=self.device)
        eps = torch.randn_like(x0)

        x_t = self.q_sample(x0, t, eps)

        drop_mask = self._drop_mask(B) if stage == "train" else None
        eps_pred = self.cdiff(x_t, y, t, drop_labels=drop_mask)

        # Simple SNR-based loss weighting (minSNR from Hang et al. 2023)
        # Compute SNR from cumulative alpha_bar (correct definition)
        ab = self.alpha_bar[t]                          # (B,) — ᾱ_t
        snr = ab / (1.0 - ab)                           # (B,) — signal-to-noise ratio
        
        # Min-SNR-γ weight for ε-prediction:
        # w_t = min{SNR, γ} / SNR = min{1, γ/SNR}
        gamma = 5.0
        weight = torch.clamp(snr, max=gamma) / snr      # (B,) — in [γ/max_snr, 1]
        weight = weight.view(B, 1)                      # broadcast over feature dim
        
        loss = (weight * (eps_pred - eps).pow(2)).mean()

        # simple mse loss
        #loss = nn.functional.mse_loss(eps_pred, eps)
        
        self.log(f"{stage}/loss", loss, on_step=(stage == "train"),
                 on_epoch=True, prog_bar=True)
        return loss

    # ── Lightning hooks ──────────────────────────────────────────────────────

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        with self.ema.average_parameters():
            self._step(batch, "val")

    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.ema.update()  # call after every optimizer step

    def on_fit_start(self):
        # Ensure EMA buffers are on the correct device after Lightning
        # has moved the model
        self.ema.to(self.device)

    def on_save_checkpoint(self, checkpoint):
        checkpoint["ema_state"] = self.ema.state_dict()
    
    def on_load_checkpoint(self, checkpoint):
        if "ema_state" in checkpoint:
            self.ema.load_state_dict(checkpoint["ema_state"])
            self.ema.to(self.device)

    def configure_optimizers(self):
        optimizers = {
            "Adam":  torch.optim.Adam,
            "AdamW": torch.optim.AdamW,
            "NAdam": torch.optim.NAdam,
            "SGD":   torch.optim.SGD,
        }
        opt_cls = optimizers.get(self.hparams.optimizer)
        if opt_cls is None:
            raise ValueError(f"Unknown optimizer: {self.hparams.optimizer}")

        # Only AdamW benefits from weight_decay; keep others at 0
        wd = self.hparams.weight_decay if self.hparams.optimizer == "AdamW" else 0.0
        optimizer = opt_cls(self.parameters(), lr=self.hparams.learning_rate,
                            weight_decay=wd)

        if self.hparams.lr_scheduler == "none":
            return optimizer

        # Cosine annealing with linear warmup
        def lr_lambda(step):
            if step < self.hparams.warmup_steps:
                return step / max(1, self.hparams.warmup_steps)
            # cosine decay to 1e-2 of peak lr
            progress = (step - self.hparams.warmup_steps) / max(
                1, self.trainer.estimated_stepping_batches - self.hparams.warmup_steps
            )
            return max(0.01, 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item()))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    # ── Sampling ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def sample_ddpm(
        self,
        y: torch.Tensor,                    # (B, n_conditions) — desired demographics
        guidance_scale: float = 3.5,
        prediction_clamp: float | None = None,
        inference_timestep_truncation: int | None = None,
    ) -> torch.Tensor:
        """DDPM ancestral sampling with CFG."""
        inference_timestep_truncation = inference_timestep_truncation if inference_timestep_truncation is not None else 0
        
        gamma = guidance_scale
        B = y.size(0)
        x = torch.randn(B, self.cdiff.input_features, device=self.device)

        with self.ema.average_parameters():
            indices = np.array(list(reversed(range(len(self.betas)))))-inference_timestep_truncation
            indices = indices[indices>=0]
            
            for t_idx in indices:
                t = torch.full((B,), t_idx, device=self.device, dtype=torch.long)
    
                # CFG: two forward passes
                eps_cond = self.cdiff.forward(x, y, t)
                eps_uncond = self.cdiff.forward_uncond(x, t)
                eps = eps_uncond + gamma * (eps_cond - eps_uncond)
    
                # DDPM reverse step
                alpha      = self.alphas[t_idx]
                alpha_b    = self.alpha_bar[t_idx]
                alpha_b_prev = self.alpha_bar_prev[t_idx]
                beta       = self.betas[t_idx]
    
                x0_pred = (x - (1 - alpha_b).sqrt() * eps) / alpha_b.sqrt()
                if prediction_clamp is not None:
                    x0_pred = torch.clamp(x0_pred, -prediction_clamp, prediction_clamp)
    
                mean = (alpha_b_prev.sqrt() * beta / (1 - alpha_b)) * x0_pred + \
                        (alpha.sqrt() * (1 - alpha_b_prev) / (1 - alpha_b)) * x
    
                if t_idx > 0:
                    beta_tilde = beta * (1 - alpha_b_prev) / (1 - alpha_b)
                    noise = torch.randn_like(x)
                    x = mean + beta_tilde.sqrt() * noise
                else:
                    x = mean
                
        return x


    @torch.no_grad()
    def sample_ddim(
        self,
        y: torch.Tensor,                        # (B, n_conditions)
        n_steps: int = 20,                      # inference steps << n_timesteps
        guidance_scale: float = 3.5,
        eta: float = 1.0,                       # eta=0 → deterministic, eta=1 → DDPM-like
        prediction_clamp: float | None = None,
        inference_timestep_truncation: int | None = None,
    ) -> torch.Tensor:
        """
        DDIM sampling (Song et al. 2020).
        eta=0: fully deterministic ODE (fastest, most reproducible)
        eta=1: recovers stochastic DDPM-like behaviour
        """
        inference_timestep_truncation = inference_timestep_truncation if inference_timestep_truncation is not None else 0
        gamma = guidance_scale
        B = y.size(0)
    
        # ── Build strided timestep subsequence ──────────────────────────────────
        total_T = len(self.betas)
        step = total_T // n_steps
        timesteps = torch.arange(total_T-1-inference_timestep_truncation, -1, -step)[:n_steps]
    
        x = torch.randn(B, self.cdiff.input_features, device=self.device)

        with self.ema.average_parameters():
            for i, t_idx in enumerate(timesteps):
        
                t = torch.full((B,), t_idx, device=self.device, dtype=torch.long)
        
                # ── CFG: two forward passes ──────────────────────────────────────
                eps_cond = self.cdiff.forward(x, y, t)
                eps_uncond = self.cdiff.forward_uncond(x, t)
                eps = eps_uncond + gamma * (eps_cond - eps_uncond)
        
                # ── Current ᾱ_t ─────────────────────────────────────────────────
                ab_t  = self.alpha_bar[t_idx]                           # ᾱ_t  (scalar)
    
                
                # ── Previous ᾱ_{t-1} (or 1.0 at the last step) ─────────────────
                # would be more elegant if we used self.alpha_bar_prev, but the current implementation works as well
                t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else -1
                ab_prev = self.alpha_bar[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=self.device)
                
                # ── Predict x0 from current x_t and eps ─────────────────────────
                # x_t = √ᾱ_t · x0 + √(1-ᾱ_t) · ε  →  solve for x0:
                
                x0_pred = (x - (1 - ab_t).sqrt() * eps) / ab_t.sqrt()
                if prediction_clamp is not None:
                    x0_pred = torch.clamp(x0_pred, -prediction_clamp, prediction_clamp)
        
                # ── DDIM variance term (sigma) ───────────────────────────────────
                # eta=0 → sigma=0 → fully deterministic
                sigma = eta * ((1 - ab_prev) / (1 - ab_t)).sqrt() * (1 - ab_t / ab_prev).sqrt()
        
                # ── Direction pointing to x_t (the "corrected" noise direction) ──
                # Ensures x_{t-1} is consistent with predicted eps after x0 update
                dir_xt = (1 - ab_prev - sigma ** 2).clamp(min=0).sqrt() * eps
        
                # ── DDIM update ──────────────────────────────────────────────────
                noise = torch.randn_like(x) if eta > 0 else torch.zeros_like(x)
                x = ab_prev.sqrt() * x0_pred + dir_xt + sigma * noise
    
        return x

    @torch.no_grad()
    def generate_with_condition_dict(self, condition_dict, n_steps=20, guidance_scale=3.5, eta=1.0, sampling_method='ddim', prediction_clamp=None, inference_timestep_truncation=None):
        y = encode_labels(
            condition_dict,
            embedding_size=self.pos_embedding_dim,
            device=self.device,
            pe_base=0.1,
        )

        if sampling_method == 'ddpm':
            x = self.sample_ddpm(y, guidance_scale, prediction_clamp, inference_timestep_truncation)
        elif sampling_method == 'ddim':
            x = self.sample_ddim(y, n_steps, guidance_scale, eta, prediction_clamp, inference_timestep_truncation)
        else:
            raise ValueError("sampling_method must be in {'ddpm', 'ddim'}")
        return x

        
    def cosine_beta_schedule(self, T: int, s: float = 0.008, T_truncation: int = 3):
        """
        Cosine schedule from Nichol & Dhariwal (2021).
        Returns alpha_bar directly to avoid cumprod roundtrip error.
        Truncates the last timesteps to avoid numerical issues.
        """
        T_truncation = T_truncation if T_truncation is not None else 0
        
        steps = torch.arange(T + T_truncation + 1, dtype=torch.float64)
        f = torch.cos(((steps / (T+T_truncation)) + s) / (1 + s) * torch.pi * 0.5) ** 2
        # normalise so ᾱ_0 = 1
        alpha_bar = (f / f[0]).float()

        if T_truncation > 0:
            return alpha_bar[:-T_truncation] 
        else:
            return alpha_bar
