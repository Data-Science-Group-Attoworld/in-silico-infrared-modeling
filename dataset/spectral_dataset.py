import sys
import torch
from torch.utils.data import Dataset

sys.path.append("../")
from utils.utils import load_config


class SpectralDataset(Dataset):
    def __init__(self, data):
        self.data = data

        # load config and get column specification
        cfg = load_config()
        self.continous_conditions = cfg["condition_labels"]["continuous"]
        self.categorical_conditions = cfg["condition_labels"]["categorical"]
        self.conditions = self.categorical_conditions + self.continous_conditions

        # Feature columns
        self.features = data.columns.difference(self.conditions).tolist()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        spectrum = torch.tensor(row[self.features].values, dtype=torch.float32)
        condition_dict = {
            label: torch.tensor(row[label], dtype=torch.float32)
            for label in self.conditions
        }
        return {"spectrum": spectrum, "labels": condition_dict}
