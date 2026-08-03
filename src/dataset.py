import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class TimeSeriesDataset(Dataset):
    """
    Custom PyTorch Dataset for Sliding Window Time Series Data.
    
    Args:
        data (np.ndarray): Scaled 2D array of shape (num_samples, num_features).
                           The target feature MUST be at index `target_col_idx` (default 0).
        lookback (int): Number of past timesteps (hours) to use as input sequence (X).
        forecast_horizon (int): Number of future timesteps (hours) to predict (y).
        target_col_idx (int): Column index of the target variable (default 0 for AEP_MW).
    """
    def __init__(self, data: np.ndarray, lookback: int = 24, forecast_horizon: int = 1, target_col_idx: int = 0):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.lookback = lookback
        self.forecast_horizon = forecast_horizon
        self.target_col_idx = target_col_idx

    def __len__(self):
        # Total valid sequences that fit within the lookback and forecast window
        return len(self.data) - self.lookback - self.forecast_horizon + 1

    def __getitem__(self, idx):
        # Input sequence X: past 'lookback' hours across ALL features
        x = self.data[idx : idx + self.lookback]
        
        # Target y: future 'forecast_horizon' hours for ONLY the target column (AEP_MW)
        y = self.data[idx + self.lookback : idx + self.lookback + self.forecast_horizon, self.target_col_idx]
        
        return x, y


def create_dataloaders(train_data, val_data, test_data, lookback=24, forecast_horizon=1, batch_size=64):
    """
    Helper function to instantiate PyTorch DataLoaders for train, val, and test splits.
    """
    train_dataset = TimeSeriesDataset(train_data, lookback, forecast_horizon)
    val_dataset = TimeSeriesDataset(val_data, lookback, forecast_horizon)
    test_dataset = TimeSeriesDataset(test_data, lookback, forecast_horizon)

    # Note: shuffle=False for time series evaluation/validation to preserve order,
    # shuffle=True can be used for training batches if preferred.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader