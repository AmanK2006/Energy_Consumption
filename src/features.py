import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def prepare_train_val_test_splits(df_path='data/processed/processed_aep.csv', train_ratio=0.70, val_ratio=0.15):
    """
    Loads processed dataset, splits chronologically, and scales features without data leakage.
    """
    df = pd.read_csv(df_path, index_col=0, parse_dates=True)
    
    # Feature columns and values
    feature_cols = ['AEP_MW', 'hour', 'dayofweek', 'month', 'year']
    data = df[feature_cols].values
    
    # Chronological Split
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]
    
    # Initialize Scalers
    target_scaler = MinMaxScaler(feature_range=(0, 1))
    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    
    # Fit ONLY on training data
    target_scaler.fit(train_data[:, 0:1])
    feature_scaler.fit(train_data)
    
    # Transform all splits
    train_scaled = feature_scaler.transform(train_data)
    val_scaled = feature_scaler.transform(val_data)
    test_scaled = feature_scaler.transform(test_data)
    
    return train_scaled, val_scaled, test_scaled, target_scaler, feature_scaler