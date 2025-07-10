import yaml
import torch


def load_config(path="config.yaml"):
    """Loads config file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def position_encoding(timesteps, embedding_size, device):
    """Positional encoding for conditioning vectors."""
    timesteps = timesteps.to(device)
    inv_freq = 1.0 / (
        0.1
        ** (torch.arange(0, embedding_size, 2, device=device).float() / embedding_size)
    )
    pos_enc_a = torch.sin(timesteps.repeat(1, embedding_size // 2) * inv_freq)
    pos_enc_b = torch.cos(timesteps.repeat(1, embedding_size // 2) * inv_freq)
    return torch.cat([pos_enc_a, pos_enc_b], dim=1)


def encode_labels(labels_dict, embedding_size, device):
    """Gets condition/label dict and converts input into valid network input."""
    encoded_labels = []

    for _, label_tensor in labels_dict.items():
        label_tensor = label_tensor.to(device)
        if label_tensor.ndimension() < 2:
            label_tensor = label_tensor.unsqueeze(1)

        encoded = position_encoding(
            label_tensor, embedding_size=embedding_size, device=device
        )
        encoded_labels.append(encoded)

    return torch.cat(encoded_labels, dim=1).to(torch.float32)
