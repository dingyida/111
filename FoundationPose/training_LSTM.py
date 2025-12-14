"""
LSTM Trajectory Prediction (Per-Object)
========================================
Trains a separate LSTM model for each object (actor_id).
Each object has different motion characteristics (accel_limit),
so they should be learned separately.

Usage:
    python lstm_per_object.py --csv poses.csv --epochs 200
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import argparse
import os
import json
import math
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Tuple, Optional, List


# ============================================================================
# DATASET
# ============================================================================

class ObjectTrajectoryDataset(Dataset):
    """Dataset for a single object's trajectories."""
    
    def __init__(self, sequences: np.ndarray, targets: np.ndarray,
                 normalize_params: Optional[Dict] = None):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        
        if normalize_params is not None:
            mean = torch.tensor(normalize_params['mean'], dtype=torch.float32)
            std = torch.tensor(normalize_params['std'], dtype=torch.float32)
            self.sequences = (self.sequences - mean) / (std + 1e-8)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


def load_object_data(csv_path: str, seq_len: int = 5,
                     features: List[str] = None) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """
    Load data and separate by object (block_idx/actor_id).
    
    Returns:
        Dict mapping object_id -> (sequences, targets)
    """
    if features is None:
        features = ['x', 'y', 'vx', 'vy', 'speed', 'accel_limit']
    
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")
    print(f"Objects: {sorted(df['block_idx'].unique())}")
    
    df = df.sort_values(['block_idx', 'frame']).reset_index(drop=True)
    
    object_data = {}
    
    for block_idx in df['block_idx'].unique():
        block_df = df[df['block_idx'] == block_idx].reset_index(drop=True)
        feat_array = block_df[features].values
        
        sequences = []
        targets = []
        
        # Create sequences for this object only
        for i in range(len(block_df) - seq_len):
            sequences.append(feat_array[i:i + seq_len])
            targets.append([
                block_df.iloc[i + seq_len]['x'],
                block_df.iloc[i + seq_len]['y']
            ])
        
        sequences = np.array(sequences, dtype=np.float32)
        targets = np.array(targets, dtype=np.float32)
        
        object_data[int(block_idx)] = (sequences, targets)
        
        print(f"  Object {block_idx}: {len(sequences)} sequences")
    
    return object_data


def split_data(sequences: np.ndarray, targets: np.ndarray,
               test_ratio: float = 0.15, val_ratio: float = 0.15,
               seed: int = 42) -> Tuple:
    """Split single object's data into train/val/test."""
    np.random.seed(seed)
    n = len(sequences)
    
    # Use sequential split to respect temporal order
    # (avoid using future data to predict past)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    n_train = n - n_test - n_val
    
    X_train = sequences[:n_train]
    y_train = targets[:n_train]
    X_val = sequences[n_train:n_train + n_val]
    y_val = targets[n_train:n_train + n_val]
    X_test = sequences[n_train + n_val:]
    y_test = targets[n_train + n_val:]
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def compute_norm_params(X_train: np.ndarray) -> Dict:
    """Compute normalization parameters from training data."""
    flat = X_train.reshape(-1, X_train.shape[-1])
    return {
        'mean': flat.mean(axis=0),
        'std': flat.std(axis=0)
    }


# ============================================================================
# LSTM MODEL
# ============================================================================

class BivarGaussianNLL(nn.Module):
    """Bivariate Gaussian Negative Log-Likelihood loss."""
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mu_x = pred[:, 0]
        mu_y = pred[:, 1]
        log_sx = pred[:, 2]
        log_sy = pred[:, 3]
        atanh_rho = pred[:, 4]
        
        sx = torch.exp(log_sx).clamp(min=1e-4, max=10)
        sy = torch.exp(log_sy).clamp(min=1e-4, max=10)
        rho = torch.tanh(atanh_rho) * 0.99
        
        x, y = target[:, 0], target[:, 1]
        dx = (x - mu_x) / sx
        dy = (y - mu_y) / sy
        
        one_minus_rho2 = (1 - rho ** 2).clamp(min=1e-6)
        z = (dx**2 + dy**2 - 2*rho*dx*dy) / one_minus_rho2
        
        nll = (math.log(2 * math.pi) + log_sx + log_sy +
               0.5 * torch.log(one_minus_rho2) + 0.5 * z)
        
        return nll.mean()


class LSTMPredictor(nn.Module):
    """
    LSTM model for trajectory prediction.
    Predicts bivariate Gaussian parameters.
    """
    
    def __init__(self, input_dim: int = 6, hidden_dim: int = 128,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        
        self.input_norm = nn.LayerNorm(input_dim)
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Output: ¦Ìx, ¦Ìy, log(¦Òx), log(¦Òy), atanh(¦Ñ)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 5)
        )
        
        # Initialize for reasonable ¦Ò (~0.5m) and ¦Ñ (~0)
        self.fc[-1].bias.data = torch.tensor([0., 0., -0.7, -0.7, 0.])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(x)
        lstm_out, _ = self.lstm(x)
        
        # Attention
        attn_weights = F.softmax(self.attention(lstm_out), dim=1)
        context = (lstm_out * attn_weights).sum(dim=1)
        
        return self.fc(context)


# ============================================================================
# TRAINING
# ============================================================================

def train_single_object(
    object_id: int,
    sequences: np.ndarray,
    targets: np.ndarray,
    epochs: int = 200,
    batch_size: int = 64,
    lr: float = 1e-3,
    hidden_dim: int = 128,
    num_layers: int = 2,
    device: str = 'cpu',
    verbose: bool = True
) -> Tuple[nn.Module, Dict, Dict]:
    """
    Train LSTM model for a single object.
    
    Returns:
        model, norm_params, metrics
    """
    # Split data (temporal split)
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(sequences, targets)
    
    if verbose:
        print(f"\n  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Compute normalization from training data only
    norm_params = compute_norm_params(X_train)
    
    # Create datasets
    train_ds = ObjectTrajectoryDataset(X_train, y_train, norm_params)
    val_ds = ObjectTrajectoryDataset(X_val, y_val, norm_params)
    test_ds = ObjectTrajectoryDataset(X_test, y_test, norm_params)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)
    
    # Create model
    input_dim = sequences.shape[-1]
    model = LSTMPredictor(input_dim, hidden_dim, num_layers).to(device)
    
    # Loss and optimizer
    loss_fn = BivarGaussianNLL()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=15, verbose=False
    )
    
    # Training loop
    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    max_patience = 30
    
    history = {'train_loss': [], 'val_loss': [], 'val_error': []}
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_losses = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())
        
        # Validate
        model.eval()
        val_losses, val_errors = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                val_losses.append(loss_fn(pred, y).item())
                
                mu_x, mu_y = pred[:, 0], pred[:, 1]
                error = torch.sqrt((mu_x - y[:, 0])**2 + (mu_y - y[:, 1])**2)
                val_errors.extend(error.cpu().numpy())
        
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        val_error = np.mean(val_errors)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_error'].append(val_error)
        
        scheduler.step(val_loss)
        
        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= max_patience:
            if verbose:
                print(f"  Early stopping at epoch {epoch + 1}")
            break
        
        # Print progress
        if verbose and ((epoch + 1) % 20 == 0 or epoch == 0):
            print(f"  Epoch {epoch+1:3d}/{epochs} | "
                  f"Train: {train_loss:.4f} | "
                  f"Val: {val_loss:.4f} | "
                  f"Error: {val_error:.4f}m")
    
    # Load best model
    model.load_state_dict(best_state)
    
    # Test evaluation
    model.eval()
    test_errors = []
    test_sigmas = []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            
            mu_x, mu_y = pred[:, 0], pred[:, 1]
            error = torch.sqrt((mu_x - y[:, 0])**2 + (mu_y - y[:, 1])**2)
            test_errors.extend(error.cpu().numpy())
            
            sigma_x = torch.exp(pred[:, 2])
            sigma_y = torch.exp(pred[:, 3])
            test_sigmas.extend(((sigma_x + sigma_y) / 2).cpu().numpy())
    
    metrics = {
        'test_error': float(np.mean(test_errors)),
        'test_sigma': float(np.mean(test_sigmas)),
        'best_val_loss': float(best_val_loss),
        'history': history
    }
    
    if verbose:
        print(f"  Test Error: {metrics['test_error']:.4f}m, ¦Ò: {metrics['test_sigma']:.4f}m")
    
    return model, norm_params, metrics


def train_all_objects(
    csv_path: str,
    epochs: int = 200,
    batch_size: int = 64,
    lr: float = 1e-3,
    hidden_dim: int = 128,
    num_layers: int = 2,
    seq_len: int = 5,
    device: str = 'auto',
    output_dir: str = 'models'
):
    """
    Train separate LSTM model for each object.
    """
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("=" * 60)
    print("Per-Object LSTM Training")
    print("=" * 60)
    print(f"Device: {device}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data separated by object
    object_data = load_object_data(csv_path, seq_len=seq_len)
    
    # Train model for each object
    results = {}
    
    for obj_id, (sequences, targets) in object_data.items():
        print(f"\n{'='*60}")
        print(f"Training Object {obj_id}")
        print("=" * 60)
        
        model, norm_params, metrics = train_single_object(
            object_id=obj_id,
            sequences=sequences,
            targets=targets,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            device=device
        )
        
        # Save model for this object
        save_path = os.path.join(output_dir, f'object_{obj_id}_model.pt')
        torch.save({
            'object_id': obj_id,
            'model_state_dict': model.state_dict(),
            'norm_params': norm_params,
            'metrics': metrics,
            'config': {
                'input_dim': sequences.shape[-1],
                'hidden_dim': hidden_dim,
                'num_layers': num_layers,
                'seq_len': seq_len
            }
        }, save_path)
        
        results[obj_id] = {
            'test_error': metrics['test_error'],
            'test_sigma': metrics['test_sigma'],
            'model_path': save_path
        }
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Object':<10} {'Test Error (m)':<15} {'Avg ¦Ò (m)':<15}")
    print("-" * 40)
    
    total_error = 0
    for obj_id, res in results.items():
        print(f"{obj_id:<10} {res['test_error']:<15.4f} {res['test_sigma']:<15.4f}")
        total_error += res['test_error']
    
    avg_error = total_error / len(results)
    print("-" * 40)
    print(f"{'Average':<10} {avg_error:<15.4f}")
    
    # Save summary
    summary_path = os.path.join(output_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nModels saved to: {output_dir}/")
    print(f"Summary saved to: {summary_path}")
    
    return results


# ============================================================================
# INFERENCE
# ============================================================================

class PerObjectPredictor:
    """
    Predictor that loads and uses per-object models.
    """
    
    def __init__(self, model_dir: str, device: str = 'auto'):
        """
        Load all object models from directory.
        
        Args:
            model_dir: Directory containing object_X_model.pt files
        """
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        self.models = {}
        self.norm_params = {}
        self.configs = {}
        
        # Load all object models
        for filename in os.listdir(model_dir):
            if filename.startswith('object_') and filename.endswith('_model.pt'):
                obj_id = int(filename.split('_')[1])
                path = os.path.join(model_dir, filename)
                
                checkpoint = torch.load(path, map_location=self.device)
                
                config = checkpoint['config']
                model = LSTMPredictor(
                    input_dim=config['input_dim'],
                    hidden_dim=config['hidden_dim'],
                    num_layers=config['num_layers']
                )
                model.load_state_dict(checkpoint['model_state_dict'])
                model = model.to(self.device)
                model.eval()
                
                self.models[obj_id] = model
                self.norm_params[obj_id] = checkpoint['norm_params']
                self.configs[obj_id] = config
        
        print(f"Loaded models for objects: {sorted(self.models.keys())}")
    
    def predict(self, object_id: int, past_frames: np.ndarray) -> Dict:
        """
        Predict next position for a specific object.
        
        Args:
            object_id: Which object (block_idx)
            past_frames: [seq_len, features] or [batch, seq_len, features]
        
        Returns:
            Dict with mu_x, mu_y, sigma_x, sigma_y, rho
        """
        if object_id not in self.models:
            raise ValueError(f"No model for object {object_id}. "
                           f"Available: {list(self.models.keys())}")
        
        model = self.models[object_id]
        norm = self.norm_params[object_id]
        
        # Handle dimensions
        if past_frames.ndim == 2:
            past_frames = past_frames[np.newaxis, ...]
        
        # Normalize
        x = (past_frames - norm['mean']) / (norm['std'] + 1e-8)
        x = torch.tensor(x, dtype=torch.float32).to(self.device)
        
        # Predict
        with torch.no_grad():
            pred = model(x).cpu().numpy()
        
        # Extract parameters
        result = {
            'mu_x': pred[:, 0],
            'mu_y': pred[:, 1],
            'sigma_x': np.exp(np.clip(pred[:, 2], -5, 3)),
            'sigma_y': np.exp(np.clip(pred[:, 3], -5, 3)),
            'rho': np.tanh(pred[:, 4])
        }
        
        # Squeeze if single sample
        if len(pred) == 1:
            result = {k: float(v[0]) for k, v in result.items()}
        
        return result
    
    def sample(self, object_id: int, past_frames: np.ndarray, 
               n_samples: int = 100) -> np.ndarray:
        """Sample positions from predicted distribution."""
        pred = self.predict(object_id, past_frames)
        
        mu = np.array([pred['mu_x'], pred['mu_y']])
        sx, sy, rho = pred['sigma_x'], pred['sigma_y'], pred['rho']
        
        cov = np.array([
            [sx**2, rho * sx * sy],
            [rho * sx * sy, sy**2]
        ])
        
        return np.random.multivariate_normal(mu, cov, size=n_samples)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Per-Object LSTM Training')
    parser.add_argument('--csv', type=str, required=True, help='Path to poses.csv')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--seq_len', type=int, default=5)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--output_dir', type=str, default='models')
    args = parser.parse_args()
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    train_all_objects(
        csv_path=args.csv,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        seq_len=args.seq_len,
        device=args.device,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
