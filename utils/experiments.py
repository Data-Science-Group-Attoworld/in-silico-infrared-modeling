import sys
sys.path.append('../')

import torch
import numpy as np
import pandas as pd


import joblib
import contextlib
import io

from models.cdiff_lightning import LitCfgDiffusion
from models.cvae_lightning import LitVAE
from models.cbegan_lightning import SpectralCBEGAN
from utils.utils import encode_labels, load_config

def load_colors():
    """define colors for models"""
    colors = {
        "CBEGAN": "#c31a1a",  # Red
        "CVAE": "#0287e8eb",  # Blue
        "CDIFF": "#ff8811",  # Orange
        "real": "#000000"   # black
    }
    return colors


def generate_batch_spectra(model, labels, column_names=None, device="cpu"):
    """
    """
    # Normalize Series to a single-row DataFrame
    if isinstance(labels, pd.Series):
        labels = labels.to_frame().T.reset_index(drop=True)

    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device).eval()
    num_samples = len(labels)
    
    # used a different order for the diffusion model
    # instead of retraining the order for gan and vae is hard coded here
    # not elegant, but works
    conditions_gan_vae = ['age', 'sex', 'bmi']     
    cfg = load_config('../config.yaml')
    continous_conditions = cfg["condition_labels"]["continuous"]
    categorical_conditions = cfg["condition_labels"]["categorical"]
    conditions = categorical_conditions + continous_conditions

    with torch.no_grad():
        if hasattr(model, "generator"):  # Handle CBEGAN
            labels_dict = {
                col: torch.tensor(labels[col].values, device=device, dtype=torch.float32)
                for col in conditions_gan_vae
            }
            noise_dim = model.noise_dim
            embedding_dim = model.pos_encoding_embedding_size
            noise = torch.randn(num_samples, noise_dim).to(device)
            encoded_conditions = encode_labels(labels_dict, embedding_dim, device)
            generated_data = model.generator(noise, encoded_conditions)
            
        elif hasattr(model, "vae"):  # Handle CVAE
            labels_dict = {
                col: torch.tensor(labels[col].values, device=device, dtype=torch.float32)
                for col in conditions_gan_vae
            }
            latent_dim = model.vae.latent_dim
            noise = torch.randn(num_samples, latent_dim).to(device)
            generated_data = model.vae.decode(noise, labels_dict)
            
        elif hasattr(model, "cdiff"):  # Handle CDIFF
            labels_dict = {
                col: torch.tensor(labels[col].values, device=device, dtype=torch.float32).unsqueeze(1)
                for col in conditions
            }
            generated_data = model.generate_with_condition_dict(
                labels_dict,
                n_steps=89,
                guidance_scale=1.10,
                eta=0.2,
                sampling_method='ddpm',
                prediction_clamp=None,
                inference_timestep_truncation=None,
            )
            
        else:
            raise ValueError("Model type not recognized. Expected a 'generator' or 'vae' attribute.")

    if column_names is not None:
        generated_data_df = pd.DataFrame(
            generated_data.cpu().detach().numpy(),
            columns=column_names
        )
    else:
        generated_data_df = pd.DataFrame(
            generated_data.cpu().detach().numpy(),
        )
    return generated_data_df



def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # CBEGAN
    cbegan = SpectralCBEGAN.load_from_checkpoint("../pretrained_models/weights_cbegan.ckpt", map_location=device)
    cbegan.to(device)
    cbegan.eval()
    
    # CVAE
    checkpoint = torch.load(
        "../pretrained_models/weights_cvae.ckpt",
        weights_only=False,
        map_location=device
    )
    
    with contextlib.redirect_stdout(io.StringIO()):
        hparams = checkpoint['hyper_parameters']
        cvae = LitVAE(**hparams)
        cvae.load_state_dict(checkpoint['state_dict'])
        cvae.to(device)
        cvae.eval()
    
    # CDIFF
    cdiff = LitCfgDiffusion.load_from_checkpoint("../pretrained_models/weights_cdiff.ckpt", map_location=device)
    cdiff.to(device)
    cdiff.eval();
    
    models = {
        'CVAE': cvae,
        'CBEGAN': cbegan,
        'CDIFF': cdiff,
    }

    return models