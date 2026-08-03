import os
import sys
import torch
import joblib
import pandas as pd
import numpy as np

# Ensure src modules are discoverable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.lstm_model import EnergyLSTM

class EnergyForecaster:
    """Production wrapper for loading PyTorch LSTM model and performing inference."""
    
    def __init__(self, model_path='models/best_lstm_model.pth',
                 feature_scaler_path='models/feature_scaler.pkl',
                 target_scaler_path='models/target_scaler.pkl',
                 hidden_size=64, num_layers=2, forecast_horizon=1):
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.forecast_horizon = forecast_horizon
        
        # 1. Load Scalers
        if not os.path.exists(feature_scaler_path) or not os.path.exists(target_scaler_path):
            raise FileNotFoundError("Scaler files not found. Ensure scalers are saved during training.")
            
        self.feature_scaler = joblib.load(feature_scaler_path)
        self.target_scaler = joblib.load(target_scaler_path)
        
        # Infer expected feature count from the fitted scaler
        self.num_features = self.feature_scaler.n_features_in_
        
        # 2. Re-instantiate Architecture & Load Checkpoint
        self.model = EnergyLSTM(
            input_size=self.num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=forecast_horizon
        ).to(self.device)
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")
            
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()  # Set to evaluation mode (disables dropout, etc.)

    def predict(self, raw_input_df: pd.DataFrame) -> np.ndarray:
        """
        Takes a DataFrame containing the raw past 24-hour sequence of features,
        preprocesses it, runs the LSTM model, and returns unscaled predictions.
        
        :param raw_input_df: pandas DataFrame with shape (24, num_features)
        :return: Numpy array containing unscaled predicted value(s) in original units.
        """
        # Validate Input Shape
        if len(raw_input_df) < 24:
            raise ValueError(f"Expected at least 24 past time steps, but got {len(raw_input_df)}.")
        
        # Select exact trailing 24 hours if more rows were provided
        input_data = raw_input_df.tail(24)
        
        if input_data.shape[1] != self.num_features:
            raise ValueError(
                f"Feature count mismatch! Scaler expects {self.num_features} columns, "
                f"but received {input_data.shape[1]}."
            )

        # 1. Scale Features
        scaled_features = self.feature_scaler.transform(input_data)
        
        # 2. Reshape to Tensor: (Batch_Size=1, Sequence_Length=24, Num_Features)
        input_tensor = torch.tensor(scaled_features, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # 3. Model Inference (Without gradient computation)
        with torch.no_grad():
            scaled_prediction = self.model(input_tensor)
            
        # Convert tensor prediction to numpy array on CPU
        scaled_prediction_np = scaled_prediction.cpu().numpy()
        
        # 4. Inverse Transform Prediction to Real-World Units
        # Reshape to 2D for scaler compatibility: (1, forecast_horizon)
        if scaled_prediction_np.ndim == 1:
            scaled_prediction_np = scaled_prediction_np.reshape(-1, 1)
            
        unscaled_prediction = self.target_scaler.inverse_transform(scaled_prediction_np)
        
        return unscaled_prediction.flatten()

# -------------------------------------------------------------
# Example Usage Entry Point
# -------------------------------------------------------------
if __name__ == "__main__":
    print("Initializing EnergyForecaster...")
    forecaster = EnergyForecaster()
    
    # Simulate receiving recent 24-hour raw feature data
    num_cols = forecaster.num_features
    dummy_past_24h = pd.DataFrame(
        np.random.rand(24, num_cols) * 100,
        columns=[f"feature_{i}" for i in range(num_cols)]
    )
    
    # Run Inference
    forecast = forecaster.predict(dummy_past_24h)
    print(f"\nNext-Hour Energy Forecast (Original Units): {forecast[0]:.2f}")