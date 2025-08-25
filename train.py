import argparse
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from lightning.pytorch.loggers import CSVLogger

from utils.utils import load_config
from dataset.dataset_builder import SpectralDatasetBuilder
from dataset.spectral_dataset import SpectralDataset

from models.cbegan_lightning import SpectralCBEGAN
from models.cvae_lightning import LitVAE


def prepare_dataloaders(data_path, batch_size, data_split={"train": 0.8, "test": 0.2}):
    """
    Prepare dataloaders for training.
    """
    dataset_builder = SpectralDatasetBuilder(data_path=data_path, data_split=data_split)
    train_dataset = SpectralDataset(dataset_builder.df_train)

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=8)
    print(f"Number of training samples: {len(train_dataset)}")

    return train_loader, dataset_builder.std_scaler


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


if __name__ == "__main__":

    # parser argument: model type
    parser = argparse.ArgumentParser(
        description="Training arguments generative spectral model."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["cbegan", "cvae"],
        help="Enter model type (cvae or cbegan)",
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

    print("Finished training.")
