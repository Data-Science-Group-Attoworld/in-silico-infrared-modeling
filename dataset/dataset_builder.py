import os
import sys
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize, StandardScaler, MinMaxScaler, LabelEncoder
from pandas.api.types import is_numeric_dtype

sys.path.append("../")
from utils.utils import load_config


class SpectralDatasetBuilder:
    def __init__(self, data_path, data_split):
        pd.set_option("future.no_silent_downcasting", True)
        np.random.seed(42)

        cfg = load_config()

        # Initialize paths and config
        self.data_path = data_path
        self.data_split = data_split
        df = self.load_data()

        # Extract label names
        self.continous_conditions = cfg["condition_labels"]["continuous"]
        self.categorical_conditions = cfg["condition_labels"]["categorical"]
        self.conditions = self.categorical_conditions + self.continous_conditions

        # Feature columns
        self.features = df.columns.difference(self.conditions).tolist()

        # Preprocessing
        self.label_encoder = LabelEncoder()
        self.std_scaler = StandardScaler()
        self.minmax_scaler = MinMaxScaler()

        df = self.encode_categorical_variables(df)
        df = self.normalize_data(df)

        # Train/Test Split only
        df_train, df_test = self.make_split(df)

        # Scale labels
        df_train[self.continous_conditions] = self.minmax_scaler.fit_transform(
            df_train[self.continous_conditions]
        )
        df_test[self.continous_conditions] = self.minmax_scaler.transform(
            df_test[self.continous_conditions]
        )

        # Standardize features
        self.df_train = self.standardize(df_train, fit=True)
        self.df_test = self.standardize(df_test, fit=False)

    def load_data(self):
        _, ext = os.path.splitext(self.data_path.lower())

        if ext == ".csv":
            return pd.read_csv(self.data_path)
        elif ext == ".parquet":
            return pd.read_parquet(self.data_path)
        elif ext == ".xlsx":
            return pd.read_excel(self.data_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    def encode_categorical_variables(self, df):
        for col in self.categorical_conditions:
            if not is_numeric_dtype(df[col]):
                df[col] = self.label_encoder.fit_transform(df[col])
        return df

    def normalize_data(self, df):
        df[self.features] = normalize(df[self.features], norm="l2", axis=1)
        return df

    def standardize(self, df, fit=False):
        if fit:
            df[self.features] = self.std_scaler.fit_transform(df[self.features])
        else:
            df[self.features] = self.std_scaler.transform(df[self.features])
        return df

    def make_split(self, df):
        """Split dataset into train/test DataFrames."""
        # Shuffle
        df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

        n = len(df)
        train_end = int(self.data_split["train"] * n)

        df_train = df.iloc[:train_end]
        df_test = df.iloc[train_end:]

        return df_train, df_test
