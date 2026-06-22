import argparse
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from utils.utils import load_config
from dataset.dataset_builder import SpectralDatasetBuilder
from dataset.spectral_dataset import SpectralDataset
from dataset.data_classes import PytorchSpectraDataset

from models.cbegan_lightning import SpectralCBEGAN
from models.cvae_lightning import LitVAE
from models.cdiff_lightning import LitCfgDiffusion


def prepare_dataloaders(data_path, batch_size, data_split={"train": 0.8, "test": 0.2}):
    """
    Prepare dataloaders for training.
    """
    #dataset_builder = SpectralDatasetBuilder(data_path=data_path, data_split=data_split)
    dataset_builder = PytorchSpectraDataset(data_path=data_path)
    df_train = dataset_builder.df_train_scaled
    df_train = df_train[list(df_train.columns.values[:519])+['sex', 'bmi', 'age']]
    df_val = dataset_builder.df_val_scaled
    df_val = df_val[list(df_val.columns.values[:519])+['sex', 'bmi', 'age']]
    train_dataset = SpectralDataset(df_train)
    val_dataset = SpectralDataset(df_val)

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False, num_workers=8)
    print(f"Number of training samples: {len(train_dataset)}")

    return train_loader, val_loader, dataset_builder.standard_scaler


def train_cbegan(data_params, model_params, condition_params):
    """ """
    train_loader, _ = prepare_dataloaders(
        data_params["data_path"], data_params["batch_size"]
    )

    n_conditions = len(condition_params["continuous"]) + len(
        condition_params["categorical"]
    )
    generator_params = {
        "noise_dim": model_params["noise_dim"],
        "output_dim": model_params["feature_dim"],
        "n_conditions": n_conditions,
        "pos_embedding_dim": model_params["pos_embedding_dim"],
    }

    discriminator_params = {
        "input_dim": model_params["feature_dim"],
        "noise_dim": model_params["noise_dim"],
        "n_conditions": n_conditions,
        "pos_embedding_dim": model_params["pos_embedding_dim"],
    }

    spectral_cbegan = SpectralCBEGAN(
        generator_params=generator_params,
        discriminator_params=discriminator_params,
        max_epochs=model_params["max_epochs"],
        noise_dim=model_params["noise_dim"],
        initial_lr=model_params["initial_lr"],
        eta_min=model_params["eta_min"],
        lambda_k=model_params["lambda_k"],
        gamma=model_params["gamma"],
        pos_encoding_embedding_dim=model_params["pos_embedding_dim"],
    )

    logger = CSVLogger(save_dir="cbegan_logs")
    trainer = pl.Trainer(max_epochs=model_params["max_epochs"], logger=logger)
    trainer.fit(model=spectral_cbegan, train_dataloaders=train_loader)

    return 0


def train_cvae(data_params, model_params, condition_params, annealer_params):
    """ """
    train_loader, scaler = prepare_dataloaders(
        data_params["data_path"], data_params["batch_size"]
    )
    n_conditions = len(condition_params["continuous"]) + len(
        condition_params["categorical"]
    )

    vae_params = {
        "input_features": model_params["feature_dim"],
        "n_conditions": n_conditions,
        "mean": scaler.mean_,
        "sigma": scaler.var_,
        "n_layer": model_params["n_layer"],
        "latent_dim": model_params["latent_dim"],
        "activation_function": model_params["activation"],
        "pos_encoding_embedding_dim": model_params["pos_embedding_dim"],
    }

    annealer_params = {
        "cyclical_annealing": annealer_params["cyclical_annealing"],
        "total_steps": annealer_params["total_steps"],
        "shape": annealer_params["shape"],
        "baseline": annealer_params["baseline"],
        "cyclical": annealer_params["cyclical"],
    }

    lit_model = LitVAE(
        vae_params=vae_params,
        annealer_params=annealer_params,
        datamodule=train_loader,
        learning_rate=model_params["initial_lr"],
        optimizer=model_params["optimizer"],
        rec_weight=model_params["rec_weight"],
    )

    logger = CSVLogger(save_dir="cvae_logs")
    trainer = pl.Trainer(max_epochs=model_params["max_epochs"], logger=logger)
    trainer.fit(model=lit_model, train_dataloaders=train_loader)

    return 0


def train_cdiff(data_params: dict, model_params: dict, condition_params: dict) -> str:
    """
    Train a classifier-free guidance diffusion model on FTIR spectral data.

    Args:
        data_params:      data_path, batch_size, num_workers
        model_params:     feature_dim, hidden_dim, embedding_dim, n_layer,
                          n_timesteps, guidance_scale, p_uncond, activation,
                          learning_rate, optimizer, weight_decay,
                          lr_scheduler, warmup_steps, max_epochs,
                          patience, log_dir, run_name
        condition_params: continuous, categorical  (for n_conditions count only)

    Returns:
        Path to best checkpoint.
    """

    # ── Data ────────────────────────────────────────────────────────────────
    train_loader, val_loader, scaler = prepare_dataloaders(
        data_path=data_params["data_path"],
        batch_size=data_params["batch_size"],
    )

    n_conditions = len(condition_params["continuous"]) + len(
        condition_params["categorical"]
    )

    # ── Model params ────────────────────────────────────────────────────────
    cdiff_params = {
        "input_features": model_params["feature_dim"],
        "model_type": model_params["model_type"],
        "n_conditions":   n_conditions,
        "hidden_dim":     model_params["hidden_dim"],
        "pos_embedding_dim": model_params["pos_embedding_dim"],
        "timestep_embedding_dim": model_params["timestep_embedding_dim"],
        "n_layer":        model_params["n_layer"],
        "n_timesteps":    model_params["n_timesteps"],
        "p_uncond":       model_params["p_uncond"],
        "activation":     model_params["activation"],
    }

    lit_model = LitCfgDiffusion(
        cdiff_params=cdiff_params,
        learning_rate=model_params["learning_rate"],
        optimizer=model_params["optimizer"],
        weight_decay=model_params["weight_decay"],
        lr_scheduler=model_params["lr_scheduler"],
        warmup_steps=model_params["warmup_steps"],
        ema_decay=model_params["ema_decay"],
        n_timesteps=model_params["n_timesteps"],
        s_timestep_schedule=model_params["s_timestep_schedule"],
    )

    # ── Callbacks ────────────────────────────────────────────────────────────
    checkpoint_cb = ModelCheckpoint(
        monitor="val/loss",           # no val loader — monitor train loss
        mode="min",
        save_top_k=2,
        filename="cdiff-{epoch:02d}",
    )
    early_stop_cb = EarlyStopping(
        monitor="val/loss",
        patience=model_params.get("patience"),
        mode="min",
    )

    # ── Logger ───────────────────────────────────────────────────────────────
    logger = CSVLogger(
        save_dir=model_params.get("log_dir",  "cdiff_logs"),
        name=model_params.get("run_name", "run"),
    )

    # ── Trainer ──────────────────────────────────────────────────────────────
    trainer = pl.Trainer(
        max_epochs=model_params.get("max_epochs", 500),
        logger=logger,
        callbacks=[checkpoint_cb, early_stop_cb],
        log_every_n_steps=10,
        gradient_clip_val=1.0,
        deterministic=False,
    )

    trainer.fit(model=lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    return checkpoint_cb.best_model_path


if __name__ == "__main__":

    # parser argument: model type
    parser = argparse.ArgumentParser(
        description="Training arguments generative spectral model."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["cbegan", "cvae", "cdiff"],
        help="Enter model type (cvae, cbegan, or cdiff)",
    )
    args = parser.parse_args()
    model = args.model

    # config settings: data and model parameters
    cfg = load_config()
    data_params = cfg["data"]
    condition_params = cfg["condition_labels"]

    # train the the selected model
    if model == "cbegan":
        model_params = cfg["cbegan"]
        train_cbegan(data_params, model_params, condition_params)

    elif model == "cvae":
        model_params = cfg["cvae"]
        annealer_params = cfg["annealer"]
        train_cvae(data_params, model_params, condition_params, annealer_params)

    elif model == "cdiff":
        model_params = cfg["cdiff"]
        train_cdiff(data_params, model_params, condition_params)

    print("Finished training.")
