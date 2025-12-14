#!/usr/bin/env python3
"""
Train variance predictor using MLP with embeddings (Approach 2).

Input: Previous 6-DoF pose + Object ID + Camera ID + Resolution ID (19D total)
Output: 6D variance (3D position std + 3D rotation std)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class VariancePredictor(nn.Module):
    """
    MLP with separate encoders for continuous (pose) and categorical (object/camera/resolution) features.
    """
    
    def __init__(self, pose_dim=6, n_objects=5, n_cameras=5, n_resolutions=3, 
                 pose_hidden=32, embed_dim_obj=16, embed_dim_cam=16, embed_dim_res=8,
                 hidden_dim=128, dropout=0.2):
        super().__init__()
        
        # Pose encoder (continuous features)
        self.pose_encoder = nn.Sequential(
            nn.Linear(pose_dim, pose_hidden),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Embeddings for categorical features
        self.object_embed = nn.Embedding(n_objects, embed_dim_obj)
        self.camera_embed = nn.Embedding(n_cameras, embed_dim_cam)
        self.resolution_embed = nn.Embedding(n_resolutions, embed_dim_res)
        
        # Combined feature dimension
        combined_dim = pose_hidden + embed_dim_obj + embed_dim_cam + embed_dim_res
        
        # Main network
        self.fc = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 6)  # Output: 6D variance
        )
    
    def forward(self, pose, obj_id, cam_id, res_id):
        """
        Forward pass.
        
        Args:
            pose: (batch, 6) - Previous 6-DoF pose
            obj_id: (batch,) - Object IDs
            cam_id: (batch,) - Camera IDs
            res_id: (batch,) - Resolution IDs
        
        Returns:
            (batch, 6) - Predicted variance (positive values)
        """
        # Encode pose
        pose_feat = self.pose_encoder(pose)
        
        # Embed categorical features
        obj_feat = self.object_embed(obj_id)
        cam_feat = self.camera_embed(cam_id)
        res_feat = self.resolution_embed(res_id)
        
        # Concatenate all features
        combined = torch.cat([pose_feat, obj_feat, cam_feat, res_feat], dim=1)
        
        # Main network
        output = self.fc(combined)
        
        # Ensure positive variance using exp
        return torch.exp(output)


def load_and_prepare_data(csv_file):
    """
    Load training data from CSV and prepare for training.
    
    Returns:
        X_train, X_test, y_train, y_test, pose_scaler
    """
    logging.info(f"Loading data from {csv_file}...")
    df = pd.read_csv(csv_file)
    
    logging.info(f"Loaded {len(df)} samples")
    logging.info(f"  Objects: {df['object_id'].nunique()}")
    logging.info(f"  Cameras: {df['camera_id'].nunique()}")
    logging.info(f"  Resolutions: {df['resolution_id'].nunique()}")
    
    # Extract features
    pose = df[['prev_tx', 'prev_ty', 'prev_tz', 'prev_rx', 'prev_ry', 'prev_rz']].values
    obj_ids = df['object_id'].values
    cam_ids = df['camera_id'].values
    res_ids = df['resolution_id'].values
    variance = df[['var_pos_x', 'var_pos_y', 'var_pos_z', 
                   'var_rot_x', 'var_rot_y', 'var_rot_z']].values
    
    # Normalize pose (important for neural networks!)
    pose_scaler = StandardScaler()
    pose_normalized = pose_scaler.fit_transform(pose)
    
    logging.info("\nPose normalization:")
    logging.info(f"  Mean: {pose_scaler.mean_}")
    logging.info(f"  Std:  {pose_scaler.scale_}")
    
    # Split data (stratified by object to ensure all objects in train/test)
    X_pose_train, X_pose_test, \
    obj_train, obj_test, \
    cam_train, cam_test, \
    res_train, res_test, \
    y_train, y_test = train_test_split(
        pose_normalized, obj_ids, cam_ids, res_ids, variance,
        test_size=0.2, random_state=42, stratify=obj_ids
    )
    
    logging.info(f"\nData split:")
    logging.info(f"  Train: {len(X_pose_train)} samples")
    logging.info(f"  Test:  {len(X_pose_test)} samples")
    
    return (X_pose_train, X_pose_test, 
            obj_train, obj_test, 
            cam_train, cam_test, 
            res_train, res_test, 
            y_train, y_test, 
            pose_scaler)


def train_model(model, train_loader, val_loader, n_epochs=100, lr=1e-3, device='cuda'):
    """
    Train the variance predictor model.
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    criterion = nn.MSELoss()
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    logging.info(f"\nStarting training on {device}...")
    
    for epoch in range(n_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for pose, obj, cam, res, target in train_loader:
            pose, obj, cam, res, target = pose.to(device), obj.to(device), cam.to(device), res.to(device), target.to(device)
            
            optimizer.zero_grad()
            pred = model(pose, obj, cam, res)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for pose, obj, cam, res, target in val_loader:
                pose, obj, cam, res, target = pose.to(device), obj.to(device), cam.to(device), res.to(device), target.to(device)
                pred = model(pose, obj, cam, res)
                loss = criterion(pred, target)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, 'best_variance_predictor.pth')
        
        # Logging
        if epoch % 10 == 0 or epoch == n_epochs - 1:
            logging.info(f"Epoch {epoch:3d}/{n_epochs}: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}, LR={optimizer.param_groups[0]['lr']:.6f}")
    
    logging.info(f"\nTraining complete! Best validation loss: {best_val_loss:.6f}")
    
    return train_losses, val_losses


def evaluate_model(model, test_loader, device='cuda'):
    """
    Evaluate model on test set and compute detailed metrics.
    """
    model = model.to(device)
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for pose, obj, cam, res, target in test_loader:
            pose, obj, cam, res = pose.to(device), obj.to(device), cam.to(device), res.to(device)
            pred = model(pose, obj, cam, res)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(target.numpy())
    
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    
    # Compute metrics
    mse = np.mean((all_preds - all_targets) ** 2, axis=0)
    mae = np.mean(np.abs(all_preds - all_targets), axis=0)
    relative_error = np.mean(np.abs(all_preds - all_targets) / (all_targets + 1e-8), axis=0)
    
    # R2 score
    from sklearn.metrics import r2_score
    r2_overall = r2_score(all_targets, all_preds)
    r2_per_dim = [r2_score(all_targets[:, i], all_preds[:, i]) for i in range(6)]
    
    logging.info("\n" + "="*80)
    logging.info("TEST SET EVALUATION")
    logging.info("="*80)
    
    labels = ['Pos X', 'Pos Y', 'Pos Z', 'Rot X', 'Rot Y', 'Rot Z']
    
    logging.info("\nMean Squared Error (MSE):")
    for i, label in enumerate(labels):
        unit = 'mm2' if i < 3 else 'rad2'
        val = mse[i] * 1e6 if i < 3 else mse[i]
        logging.info(f"  {label:8s}: {val:.6f} {unit}")
    
    logging.info("\nMean Absolute Error (MAE):")
    for i, label in enumerate(labels):
        unit = 'mm' if i < 3 else '¡ã'
        val = mae[i] * 1000 if i < 3 else np.rad2deg(mae[i])
        logging.info(f"  {label:8s}: {val:.4f} {unit}")
    
    logging.info("\nRelative Error (%):")
    for i, label in enumerate(labels):
        logging.info(f"  {label:8s}: {relative_error[i]*100:.2f}%")
    
    logging.info("\nR2 Score:")
    logging.info(f"  Overall:  {r2_overall:.4f}")
    for i, label in enumerate(labels):
        logging.info(f"  {label:8s}: {r2_per_dim[i]:.4f}")
    
    return all_preds, all_targets, mse, mae, r2_per_dim


def plot_results(train_losses, val_losses, predictions, targets):
    """
    Plot training curves and prediction vs actual.
    """
    fig = plt.figure(figsize=(18, 10))
    
    # Create grid: 2 rows, 4 columns
    # Row 0: Training curve + 3 position plots
    # Row 1: 3 rotation plots + empty
    
    # Training curves (top-left)
    ax_train = plt.subplot(2, 4, 1)
    ax_train.plot(train_losses, label='Train')
    ax_train.plot(val_losses, label='Validation')
    ax_train.set_xlabel('Epoch')
    ax_train.set_ylabel('Loss (MSE)')
    ax_train.set_title('Training History')
    ax_train.legend()
    ax_train.grid(True, alpha=0.3)
    
    # Prediction vs actual for each dimension
    labels = ['Pos X', 'Pos Y', 'Pos Z', 'Rot X', 'Rot Y', 'Rot Z']
    units = ['mm', 'mm', 'mm', '¡ã', '¡ã', '¡ã']
    
    # Position: row 0, columns 1-3 (plots 2,3,4)
    # Rotation: row 1, columns 0-2 (plots 5,6,7)
    subplot_positions = [2, 3, 4, 5, 6, 7]
    
    for i in range(6):
        ax = plt.subplot(2, 4, subplot_positions[i])
        
        pred = predictions[:, i]
        target = targets[:, i]
        
        # Convert units for display
        if i < 3:  # Position
            pred = pred * 1000
            target = target * 1000
        else:  # Rotation
            pred = np.rad2deg(pred)
            target = np.rad2deg(target)
        
        ax.scatter(target, pred, alpha=0.3, s=1)
        min_val = min(target.min(), pred.min())
        max_val = max(target.max(), pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect')
        ax.set_xlabel(f'Actual {labels[i]} ({units[i]})')
        ax.set_ylabel(f'Predicted {labels[i]} ({units[i]})')
        ax.set_title(labels[i])
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('variance_predictor_results.png', dpi=150, bbox_inches='tight')
    logging.info("\nSaved plot to variance_predictor_results.png")
    plt.close()


def main():
    """Main training pipeline."""
    
    # Configuration
    CSV_FILE = 'variance_training_data.csv'
    BATCH_SIZE = 256
    N_EPOCHS = 1000
    LEARNING_RATE = 1e-3
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    logging.info(f"Using device: {DEVICE}")
    
    # Load and prepare data
    (X_pose_train, X_pose_test, 
     obj_train, obj_test, 
     cam_train, cam_test, 
     res_train, res_test, 
     y_train, y_test, 
     pose_scaler) = load_and_prepare_data(CSV_FILE)
    
    # Create datasets
    train_dataset = TensorDataset(
        torch.FloatTensor(X_pose_train),
        torch.LongTensor(obj_train),
        torch.LongTensor(cam_train),
        torch.LongTensor(res_train),
        torch.FloatTensor(y_train)
    )
    
    test_dataset = TensorDataset(
        torch.FloatTensor(X_pose_test),
        torch.LongTensor(obj_test),
        torch.LongTensor(cam_test),
        torch.LongTensor(res_test),
        torch.FloatTensor(y_test)
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Create model
    model = VariancePredictor()
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    logging.info(f"\nModel has {n_params:,} parameters")
    
    # Train model
    train_losses, val_losses = train_model(
        model, train_loader, test_loader, 
        n_epochs=N_EPOCHS, lr=LEARNING_RATE, device=DEVICE
    )
    
    # Load best model
    checkpoint = torch.load('best_variance_predictor.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    logging.info(f"\nLoaded best model from epoch {checkpoint['epoch']} with val_loss={checkpoint['val_loss']:.6f}")
    
    # Evaluate
    predictions, targets, mse, mae, r2 = evaluate_model(model, test_loader, device=DEVICE)
    
    # Plot results
    plot_results(train_losses, val_losses, predictions, targets)
    
    # Save final model with scaler
    torch.save({
        'model_state_dict': model.state_dict(),
        'pose_scaler_mean': pose_scaler.mean_,
        'pose_scaler_scale': pose_scaler.scale_,
        'mse': mse,
        'mae': mae,
        'r2': r2,
    }, 'variance_predictor_final.pth')
    
    logging.info("\n" + "="*80)
    logging.info("Training complete!")
    logging.info("Saved models:")
    logging.info("  - best_variance_predictor.pth (best validation)")
    logging.info("  - variance_predictor_final.pth (with scaler and metrics)")
    logging.info("="*80)


if __name__ == '__main__':
    main()
