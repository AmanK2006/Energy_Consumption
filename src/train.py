import os
import sys
import torch
import torch.nn as nn
from torch.optim import Adam
import joblib

# Ensure src modules are discoverable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.features import prepare_train_val_test_splits  # Or custom split loader
from src.dataset import create_dataloaders
from src.models.lstm_model import EnergyLSTM

def train_model():
    # -------------------------------------------------------------
    # 1. Hyperparameters & Configuration
    # -------------------------------------------------------------
    LOOKBACK = 24
    FORECAST_HORIZON = 1
    BATCH_SIZE = 64
    EPOCHS = 20
    LEARNING_RATE = 0.001
    HIDDEN_SIZE = 64
    NUM_LAYERS = 2
    
    # Automatic GPU Acceleration (CUDA / MPS / CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using compute device: {device}")

    # Create directories for outputs
    os.makedirs('models', exist_ok=True)

    # -------------------------------------------------------------
    # 2. Prepare Data & Scalers
    # -------------------------------------------------------------
    print("Preparing datasets and scaling features...")
    # Expecting prepare_train_val_test_splits or similar logic
    train_scaled, val_scaled, test_scaled, target_scaler, feature_scaler = prepare_train_val_test_splits(
        df_path='data/processed/processed_aep.csv'
    )

    # Save scalers with joblib for future inference
    joblib.dump(target_scaler, 'models/target_scaler.pkl')
    joblib.dump(feature_scaler, 'models/feature_scaler.pkl')
    print("Saved scalers to models/ directory.")

    # Create PyTorch DataLoaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_scaled, val_scaled, test_scaled,
        lookback=LOOKBACK, forecast_horizon=FORECAST_HORIZON, batch_size=BATCH_SIZE
    )

    # -------------------------------------------------------------
    # 3. Model, Loss, & Optimizer Initialization
    # -------------------------------------------------------------
    num_features = train_scaled.shape[1]
    model = EnergyLSTM(
        input_size=num_features,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        output_size=FORECAST_HORIZON
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    # -------------------------------------------------------------
    # 4. Training Loop with Validation Checkpointing
    # -------------------------------------------------------------
    best_val_loss = float('inf')

    print("\nStarting Training Loop...")
    for epoch in range(1, EPOCHS + 1):
        # TRAINING PHASE
        model.train()
        train_loss = 0.0
        
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            # Forward pass
            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)

            # Backward pass & optimization
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x_batch.size(0)

        train_loss = train_loss / len(train_loader.dataset)

        # VALIDATION PHASE
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                y_pred = model(x_batch)
                loss = criterion(y_pred, y_batch)
                val_loss += loss.item() * x_batch.size(0)

        val_loss = val_loss / len(val_loader.dataset)

        # Print Epoch Stats
        print(f"Epoch [{epoch:02d}/{EPOCHS:02d}] | Train Loss (MSE): {train_loss:.6f} | Val Loss (MSE): {val_loss:.6f}")

        # Checkpoint Best Model Weights
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'models/best_lstm_model.pth')
            print(f"  --> Saved new best model weights (Val Loss: {val_loss:.6f})")

    print("\nTraining Complete! Best model checkpoint saved to 'models/best_lstm_model.pth'")

if __name__ == '__main__':
    train_model()