import torch
import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import normalize, StandardScaler, MinMaxScaler
from torch.utils.data import Dataset, DataLoader


# Preprocessing Values
SIGNAL_LENGTH = 1101
START_WAVENUMBER = 1000
END_WAVENUMBER = 3000
START_SILENT_REGION = 1800
END_SILENT_REGION = 2800


DATA_PATH = r"../data/h4h_raw_2024-09-16_with_subject_ids.parquet"

class PytorchSpectraDataset(Dataset):
    """
    A PyTorch Dataset for handling H4H data
    """

    def __init__(self, data_path = DATA_PATH ):
        
        pd.set_option('future.no_silent_downcasting', True)
        np.random.seed(42)  # Set the random seed for reproducibility
        
        self.data_path = data_path
        self.standard_scaler = StandardScaler()
        self.min_max_scaler = MinMaxScaler()
        
        # Load data and cut signal
        df_raw = self.load_data(self.data_path)
        df_raw, length_after_cutting = self.cut_signal(df_raw)
        
        # Normalization
        df_normalized = self.normalize_data(df_raw, length_after_cutting)
        
        # Cut silent region
        df_normalized, length_after_removing_silent_region = self.remove_silent_region(df_normalized, length_after_cutting)
        
        # Create train and test split
        df_test, df_train = self.make_split(df_normalized)
        df_val, df_train = self.make_split(df_train, split=0.1)


        # Create a column indicating the different single-visit cohorts
        df_test = self.create_single_visit_cohorts(df_test)
        df_test = self.rebalance_single_visit_cohorts(df_test)
        df_train["test_set"] = -99
        df_val["test_set"] = -99
        
        # Min Max Scaling of age and bmi
        df_train[["age", "bmi"]] = self.min_max_scaler.fit_transform(df_train[["age", "bmi"]])
        df_test[["age", "bmi"]]  = self.min_max_scaler.transform(df_test[["age", "bmi"]])
        df_val[["age", "bmi"]]  = self.min_max_scaler.transform(df_val[["age", "bmi"]])
        
        self.df_train_normalized = df_train
        self.df_test_normalized = df_test
        self.df_val_normalized = df_val

        # Standard Scaling
        self.df_train_scaled = self.scale_data(df_train, signal_length = length_after_removing_silent_region, mode = "fit_transform")
        self.df_test_scaled  = self.scale_data(df_test,  signal_length = length_after_removing_silent_region, mode = "transform")
        self.df_val_scaled  = self.scale_data(df_val,  signal_length = length_after_removing_silent_region, mode = "transform")

        # Prepare PyTorch tensors
        self.data, self.labels = self.turn_data_into_torch_tensors(self.df_train_scaled, length_after_removing_silent_region)
        
    def create_single_visit_cohorts(self, df):
        """
        Assigns unique random integers between 1 and max_group_size to each row per subject_id group.
        """
        # Compute the maximum group size across all groups
        max_group_size = df.groupby('subject_id').size().max()

        def shuffle_within_group(group):
            """Assigns a unique random integer within the range 1 to max_group_size to each row in the group."""
            group_size = len(group)
            # Sample without replacement to ensure unique values for each row
            random_values = np.random.choice(range(1, max_group_size + 1), size=group_size, replace=False)
            group['test_set'] = random_values
            return group

        # Apply the transformation to each group
        df_test = df.groupby('subject_id', group_keys=False).apply(shuffle_within_group)
        return df_test
    
    def rebalance_single_visit_cohorts(self, df, max_num_visits = 6):
        
        desired_test_set_size = math.floor(len(df) / max_num_visits)  # Target size for each test set
        
        for i in range(max_num_visits):
        
            # Get the sizes of each test set, sorted
            sorted_test_set_values = df.test_set.value_counts().sort_values()

            # Identify the test set with most and least samples
            test_set_max_index = sorted_test_set_values.index[-1]
            test_set_max_size = sorted_test_set_values.values[-1]
            test_set_min_index = sorted_test_set_values.index[0]
            test_set_min_size = sorted_test_set_values.values[0]

            # Calculate the number of samples to move
            num_samples_to_move = desired_test_set_size - test_set_min_size

            # Filter potential candidates to move from max set
            subjects_in_min = df[df.test_set == test_set_min_index].subject_id
            candidates_to_move = df[(df.test_set == test_set_max_index) & (~df.subject_id.isin(subjects_in_min))]

            # Randomly select samples to move
            sampled_indices = candidates_to_move.sample(n=num_samples_to_move, random_state=42).index

            # Update these samples to the new test set
            df.loc[sampled_indices, 'test_set'] = test_set_min_index
        
        return df


    def make_split(self,df, split=0.2):

        X0 = df.iloc[:int(len(df)*split)].copy()
        X1 = df.iloc[int(len(df)*split):].copy()

        last_id_in_X0 = X0.iloc[-1]['subject_id']
        remains_in_X1 = X1[X1['subject_id']==last_id_in_X0]

        X0 = pd.concat([X0, remains_in_X1])
        X1 = X1[len(remains_in_X1):]

        X0 = X0.sample(frac=1)
        X1 = X1.sample(frac=1)

        return X0, X1

    def load_data(self, data_path):
        
        df = pd.read_parquet(data_path)
        
        df = df.rename(columns = {"age_in_year": "age"})
        df.sex = df.sex.map({"Male - férfi": 0,
                             "Female - nő": 1}
                           )
        
        # drop some wrong data
        df = df.drop(df[df.subject_id ==  "H4H_05_0220"].index)
        df = df.drop(df[(df.subject_id ==  "H4H_05_0220") & (df.sample_id ==  "C_0037887")].index)

        return df

    def cut_signal(self, df):
        df_data = df.iloc[:, 2:-6]
        df_labels = df[["subject_id", "sample_id", "age", "sex", "bmi"]]

        df_data = df_data.loc[:,
                              (df_data.columns.astype(float) > START_WAVENUMBER) & 
                              (df_data.columns.astype(float) < END_WAVENUMBER)]

        df = pd.concat([df_data, df_labels], axis = 1).dropna()

        # return df with data and labels, and length of signal after cutting the signal
        return df, len(df_data.columns)
    
    def remove_silent_region(self, df, signal_length):
        df_data = df.iloc[:,:signal_length]
        df_labels = df[["subject_id", "sample_id", "age", "sex", "bmi"]]
        
        # Remove silent region
        df_data = df_data.loc[:,
                             (df_data.columns.astype(float) < START_SILENT_REGION) |
                             (df_data.columns.astype(float) > END_SILENT_REGION)]

        df = pd.concat([df_data, df_labels], axis = 1).dropna()

        # return df with data and labels, and length of signal after cutting the signal
        return df, len(df_data.columns)
    
    def normalize_data(self, df, signal_length):
        
        #print(len(df.columns))
        columns = df.columns.copy()
        features = df.iloc[:,:signal_length].values
        
        labels = df.iloc[:, signal_length:].reset_index(drop=True)
        #print(labels.shape)
        features = normalize(features, norm="l2", axis=1)
        #print(features.shape)
        
        df = pd.concat([pd.DataFrame(features), labels], axis = 1)
        #print(df.shape)
        df.columns = columns
        return df
    
    def scale_data(self, df, signal_length, mode):
        columns = df.columns
        features = df.iloc[:, :signal_length]
        labels = df.iloc[:, signal_length:].reset_index(drop=True)
        
        if mode == "fit_transform":
            features = self.standard_scaler.fit_transform(features)
        elif mode == "transform":
            features = self.standard_scaler.transform(features)
            
        df = pd.concat([pd.DataFrame(features), labels], axis = 1)
        df.columns = columns   
        return df
    
    def turn_data_into_torch_tensors(self, df, signal_length):

        data = df.iloc[:, :signal_length].values
        data_tensor = torch.tensor(data, dtype=torch.float32)

        labels = df[["age", "sex", "bmi"]].values
        labels_tensor = torch.tensor(labels, dtype=torch.float)

        return data_tensor, labels_tensor
            
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """Retrieve a single sample from the dataset."""
        return self.data[idx], self.labels[idx]
