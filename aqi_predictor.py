"""
PM2.5 Prediction using LSTM Deep Learning Model
=================================================
Trains on air quality data (PM10, RH, Temperature, Wind Speed, time features)
from Bengaluru Jayanagar 5th Block KSPCB station.
User inputs a time of day (HH:MM) and gets a predicted PM2.5 value.
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import os
import warnings
warnings.filterwarnings('ignore')


# ─── 1. PM2.5 Category (Indian AQI PM2.5 Breakpoints) ─────────────────────────

def get_pm25_category(pm25):
    """Return PM2.5 category based on Indian AQI breakpoints for PM2.5."""
    if pm25 <= 30:
        return "Good", "Minimal impact on health"
    elif pm25 <= 60:
        return "Satisfactory", "Minor breathing discomfort for sensitive people"
    elif pm25 <= 90:
        return "Moderate", "Breathing discomfort for people with lung/heart disease"
    elif pm25 <= 120:
        return "Poor", "Breathing discomfort on prolonged exposure"
    elif pm25 <= 250:
        return "Very Poor", "Respiratory illness on prolonged exposure"
    else:
        return "Severe", "Affects healthy people; serious impact on sensitive people"


# ─── 2. Data Loading & Preprocessing ──────────────────────────────────────────

def load_and_preprocess(csv_paths):
    """Load CSVs, pivot parameters into columns, extract time features."""
    frames = []
    for path in csv_paths:
        df = pd.read_csv(path)
        frames.append(df)
    
    raw = pd.concat(frames, ignore_index=True)
    
    # Parse local datetime
    raw['datetime'] = pd.to_datetime(raw['datetimeLocal'], utc=False)
    raw['datetime'] = raw['datetime'].dt.tz_localize(None)  # strip tz info for simplicity
    
    # Pivot: rows = timestamps, columns = parameters
    pivoted = raw.pivot_table(index='datetime', columns='parameter', values='value', aggfunc='mean')
    pivoted.sort_index(inplace=True)
    pivoted.dropna(inplace=True)
    
    # Rename columns for clarity
    rename_map = {
        'pm10': 'PM10', 'pm25': 'PM2_5',
        'relativehumidity': 'RH', 'temperature': 'Temp',
        'wind_speed': 'WindSpeed'
    }
    pivoted.rename(columns=rename_map, inplace=True)
    
    # Time-of-day features (cyclical encoding)
    pivoted['hour'] = pivoted.index.hour
    pivoted['minute'] = pivoted.index.minute
    total_minutes = pivoted['hour'] * 60 + pivoted['minute']
    pivoted['time_sin'] = np.sin(2 * np.pi * total_minutes / 1440)
    pivoted['time_cos'] = np.cos(2 * np.pi * total_minutes / 1440)
    
    # Day-of-period (to distinguish day 1 vs day 2, etc.)
    min_date = pivoted.index.min().date()
    pivoted['day_num'] = (pivoted.index.date - min_date).astype('timedelta64[D]').astype(int)
    
    return pivoted


# ─── 3. LSTM Model Definition ─────────────────────────────────────────────────

class AQI_LSTM(nn.Module):
    """LSTM-based model for PM2.5 prediction from time + environmental features."""
    
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super(AQI_LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True, dropout=dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)
        # Use last time step output
        last_out = lstm_out[:, -1, :]
        return self.fc(last_out).squeeze(-1)


# ─── 4. Training Pipeline ─────────────────────────────────────────────────────

def create_sequences(features, targets, times, seq_len=4):
    """Create overlapping sequences, enforcing strict 15-min intervals."""
    X, y = [], []
    expected_delta = pd.Timedelta(minutes=15)
    
    for i in range(len(features) - seq_len):
        time_window = times[i : i + seq_len + 1]
        time_diffs = pd.Series(time_window).diff().dropna()
        
        if (time_diffs != expected_delta).any():
            continue
            
        X.append(features[i:i+seq_len])
        y.append(targets[i+seq_len])
        
    return np.array(X), np.array(y)


def train_model(data, seq_len=8, epochs=150, lr=0.001):
    """Train the LSTM model on the prepared data."""
    
    # Feature columns: time_sin, time_cos, PM10, PM2_5, RH, Temp, WindSpeed
    # PM2_5 is included so the model knows past PM2.5 trajectory
    feature_cols = ['time_sin', 'time_cos', 'PM10', 'PM2_5', 'RH', 'Temp', 'WindSpeed']
    target_col = 'PM2_5'
    
    features = data[feature_cols].values.astype(np.float32)
    targets = data[target_col].values.astype(np.float32)
    
    # Normalize features
    feat_mean = features.mean(axis=0)
    feat_std = features.std(axis=0) + 1e-8
    features_norm = (features - feat_mean) / feat_std
    
    # Normalize target
    tgt_mean = targets.mean()
    tgt_std = targets.std() + 1e-8
    targets_norm = (targets - tgt_mean) / tgt_std
    
    # Create sequences
    X, y = create_sequences(features_norm, targets_norm, data.index, seq_len)
    
    # Train/val split (80/20)
    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    
    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.FloatTensor(y_val)
    
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    
    # Model
    model = AQI_LSTM(input_size=len(feature_cols), hidden_size=64, num_layers=2, dropout=0.2)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.5)
    
    print("\n" + "="*60)
    print("  TRAINING LSTM MODEL FOR PM2.5 PREDICTION")
    print("="*60)
    print(f"  Training samples : {len(X_train)}")
    print(f"  Validation samples: {len(X_val)}")
    print(f"  Sequence length  : {seq_len}")
    print(f"  Features         : {feature_cols}")
    print("="*60 + "\n")
    
    best_val_loss = float('inf')
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            pred = model(batch_X)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_X.size(0)
        train_loss /= len(X_train)
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = criterion(val_pred, y_val_t).item()
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict().copy()
        
        if (epoch + 1) % 25 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:>4d}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
    
    # Load best model
    model.load_state_dict(best_state)
    
    # Final validation metrics
    model.eval()
    with torch.no_grad():
        val_pred = model(X_val_t).numpy()
        val_pred_pm25 = val_pred * tgt_std + tgt_mean
        val_actual_pm25 = y_val * tgt_std + tgt_mean
        mae = np.mean(np.abs(val_pred_pm25 - val_actual_pm25))
        rmse = np.sqrt(np.mean((val_pred_pm25 - val_actual_pm25)**2))
    
    print(f"\n  [OK] Best Validation Loss    : {best_val_loss:.6f}")
    print(f"  [OK] Validation MAE (PM2.5)  : {mae:.2f}")
    print(f"  [OK] Validation RMSE (PM2.5) : {rmse:.2f}")
    
    return model, feat_mean, feat_std, tgt_mean, tgt_std, data, feature_cols, seq_len


# ─── 5. Prediction Function ───────────────────────────────────────────────────

def predict_pm25(time_str, model, feat_mean, feat_std, tgt_mean, tgt_std, data, feature_cols, seq_len):
    """Predict PM2.5 for a given time of day (HH:MM format)."""
    
    # Parse input time
    parts = time_str.strip().split(':')
    hour = int(parts[0])
    minute = int(parts[1]) if len(parts) > 1 else 0
    
    total_minutes = hour * 60 + minute
    time_sin = np.sin(2 * np.pi * total_minutes / 1440)
    time_cos = np.cos(2 * np.pi * total_minutes / 1440)
    
    # Find closest time in data to get average environmental features at this time
    data_copy = data.copy()
    data_copy['time_total_min'] = data_copy['hour'] * 60 + data_copy['minute']
    
    # Get average features for the closest time window (±60 mins)
    time_diff = np.abs(data_copy['time_total_min'] - total_minutes)
    # Handle wrap-around (e.g., 23:45 close to 00:15)
    time_diff = np.minimum(time_diff, 1440 - time_diff)
    
    close_mask = time_diff <= 60  # within ±60 minutes
    if close_mask.sum() == 0:
        close_mask = time_diff <= 120
    if close_mask.sum() == 0:
        close_mask = pd.Series(True, index=data_copy.index)
    
    nearby = data_copy[close_mask]
    
    # Build a sequence from nearby data points
    avg_features = nearby[feature_cols].mean().values.astype(np.float32)
    # Override the time features with exact requested time
    avg_features[0] = time_sin
    avg_features[1] = time_cos
    
    # Create a synthetic sequence
    if len(nearby) >= seq_len:
        nearby_sorted = nearby.iloc[time_diff[close_mask].argsort().values[:seq_len]]
        seq_features = nearby_sorted[feature_cols].values.astype(np.float32)
    else:
        seq_features = np.tile(avg_features, (seq_len, 1))
    
    # Normalize
    seq_norm = (seq_features - feat_mean) / feat_std
    
    # Predict
    model.eval()
    with torch.no_grad():
        input_tensor = torch.FloatTensor(seq_norm).unsqueeze(0)  # (1, seq_len, features)
        pred_norm = model(input_tensor).item()
    
    # Denormalize
    pred_pm25 = pred_norm * tgt_std + tgt_mean
    pred_pm25 = max(0, pred_pm25)  # PM2.5 can't be negative
    
    return pred_pm25


# ─── 6. Main Entry Point ──────────────────────────────────────────────────────

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    csv_files = [
        os.path.join(script_dir, '1 Mar - 2 Mar.csv'),
        os.path.join(script_dir, '3 Mar - 4 Mar.csv'),
    ]
    
    # Check files exist
    for f in csv_files:
        if not os.path.exists(f):
            print(f"[ERROR] File not found: {f}")
            return
    
    print("\n[*] Loading and preprocessing air quality data...")
    data = load_and_preprocess(csv_files)
    print(f"   Total timestamps: {len(data)}")
    print(f"   Date range: {data.index.min()} → {data.index.max()}")
    print(f"   PM2.5 range: {data['PM2_5'].min():.1f} – {data['PM2_5'].max():.1f}")
    print(f"   Columns: {list(data.columns)}")
    
    # Train
    model, feat_mean, feat_std, tgt_mean, tgt_std, data, feature_cols, seq_len = train_model(data)
    
    # Interactive prediction loop
    print("\n" + "="*60)
    print("  PM2.5 PREDICTION SYSTEM (Bengaluru - Jayanagar)")
    print("="*60)
    print("  Enter a time (HH:MM in 24h format) to predict PM2.5.")
    print("  Type 'quit' or 'exit' to stop.\n")
    
    while True:
        try:
            user_input = input("  >> Enter time (HH:MM): ").strip()
            if user_input.lower() in ('quit', 'exit', 'q'):
                print("\n  Goodbye!\n")
                break
            
            # Basic validation
            parts = user_input.split(':')
            if len(parts) < 2:
                print("  [!] Please enter time in HH:MM format (e.g. 14:30)")
                continue
            h, m = int(parts[0]), int(parts[1])
            if not (0 <= h <= 23 and 0 <= m <= 59):
                print("  [!] Invalid time. Hours: 0-23, Minutes: 0-59")
                continue
            
            pred_pm25 = predict_pm25(user_input, model, feat_mean, feat_std,
                                     tgt_mean, tgt_std, data, feature_cols, seq_len)
            category, health = get_pm25_category(pred_pm25)
            
            print(f"\n  ┌─────────────────────────────────────────┐")
            print(f"  │  Time          : {h:02d}:{m:02d}                   │")
            print(f"  │  Predicted PM2.5: {pred_pm25:>6.1f} µg/m³          │")
            print(f"  │  Category  : {category:<28s}│")
            print(f"  │  Health    : {health:<28s}│")
            print(f"  └─────────────────────────────────────────┘\n")
            
        except KeyboardInterrupt:
            print("\n\n  Goodbye!\n")
            break
        except Exception as e:
            print(f"  [!] Error: {e}")
            continue


if __name__ == "__main__":
    main()
