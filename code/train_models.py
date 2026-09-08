#!/usr/bin/env python3
"""
Train liver segmentation models using 5-fold cross-validation.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import nibabel as nib
import os
from pathlib import Path
import json
import logging
from datetime import datetime

# Setup logging
log_dir = './results'
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_file = os.path.join(log_dir, f'training_{timestamp}.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ─────────────────────────────────────────────────────────────────────────────
# MODEL ARCHITECTURE
# ─────────────────────────────────────────────────────────────────────────────
from torchvision.models.video import r3d_18

class ModelA(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = r3d_18(weights=None)
        self.encoder.stem[0] = nn.Conv3d(1,64,kernel_size=(3,7,7),stride=(1,2,2),padding=(1,3,3),bias=False)
        self.encoder.fc = nn.Identity()
        self.classifier = nn.Linear(512,2)
    def forward(self, x):
        return self.classifier(self.encoder(x))

class ModelB(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = r3d_18(weights=None)
        self.encoder.stem[0] = nn.Conv3d(1,64,kernel_size=(3,7,7),stride=(1,2,2),padding=(1,3,3),bias=False)
        self.encoder.fc = nn.Identity()
        self.projection = nn.Sequential(nn.Linear(512,1024), nn.ReLU(), nn.Linear(1024,2048))
        self.classifier = nn.Sequential(nn.Linear(512,256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256,2))
    def forward(self, x):
        return self.classifier(self.encoder(x))

# ─────────────────────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────────────────────
class CTDataset(Dataset):
    def __init__(self, ct_paths, seg_paths, labels, patch_shape=(64, 64, 64)):
        self.ct_paths = ct_paths
        self.seg_paths = seg_paths
        self.labels = labels
        self.patch_shape = patch_shape

    def __len__(self):
        return len(self.ct_paths)

    def __getitem__(self, idx):
        ct_data = nib.load(self.ct_paths[idx]).get_fdata()
        seg_data = nib.load(self.seg_paths[idx]).get_fdata()

        # Align dimensions
        d = min(ct_data.shape[2], seg_data.shape[2])
        ct_data = ct_data[:,:,:d]
        seg_data = seg_data[:,:,:d]

        # Apply mask
        img = ct_data * (seg_data > 0)
        if img.max() == 0:
            img = ct_data

        # Extract center patch
        patch = self._extract_patch(img)

        # Preprocess
        patch = np.clip(patch, -200, 300)
        patch = (patch - patch.mean()) / (patch.std() + 1e-8)

        patch_tensor = torch.FloatTensor(patch).unsqueeze(0)
        label_tensor = torch.LongTensor([self.labels[idx]])

        return patch_tensor, label_tensor[0]

    def _extract_patch(self, ct_data):
        d, h, w = ct_data.shape
        pd, ph, pw = self.patch_shape
        sd, sh, sw = max(0, (d - pd) // 2), max(0, (h - ph) // 2), max(0, (w - pw) // 2)
        ed, eh, ew = min(d, sd + pd), min(h, sh + ph), min(w, sw + pw)
        patch = ct_data[sd:ed, sh:eh, sw:ew]

        pad_d, pad_h, pad_w = max(0, pd - patch.shape[0]), max(0, ph - patch.shape[1]), max(0, pw - patch.shape[2])
        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            patch = np.pad(patch, ((0, pad_d), (0, pad_h), (0, pad_w)), mode='constant')

        return patch

# ─────────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────────
def train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=1e-3):
    """Train a single model."""
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        for ct_patches, labels in train_loader:
            ct_patches, labels = ct_patches.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(ct_patches)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for ct_patches, labels in val_loader:
                ct_patches, labels = ct_patches.to(device), labels.to(device)
                outputs = model(ct_patches)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total
        logging.info(f"Epoch {epoch+1}/{num_epochs} | Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc

    return model

# ─────────────────────────────────────────────────────────────────────────────
# MAIN TRAINING PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def main():
    logging.info("="*80)
    logging.info("TRAINING: Liver Segmentation Models")
    logging.info("="*80)

    # Configuration
    train_data_dir = './data/train_dataset'
    batch_size = 8
    num_epochs = 50
    num_folds = 5
    output_dir = './models'
    os.makedirs(output_dir, exist_ok=True)

    # Load training data
    ct_paths = []
    seg_paths = []
    labels = []

    for label_folder in ['Decomp', 'Nodecomp']:
        label_path = os.path.join(train_data_dir, label_folder)
        if not os.path.exists(label_path):
            logging.warning(f"Folder not found: {label_path}")
            continue

        for patient_dir in sorted(os.listdir(label_path)):
            patient_full_path = os.path.join(label_path, patient_dir)
            if not os.path.isdir(patient_full_path):
                continue

            ct_file = os.path.join(patient_full_path, 'image.nii.gz')
            seg_file = None

            for f in os.listdir(patient_full_path):
                if f.endswith('.nii.gz') and 'liver' in f.lower():
                    seg_file = os.path.join(patient_full_path, f)
                    break

            if ct_file is not None and seg_file is not None:
                label = 1 if label_folder == 'Decomp' else 0
                ct_paths.append(ct_file)
                seg_paths.append(seg_file)
                labels.append(label)

    logging.info(f"Loaded {len(ct_paths)} training cases")

    if len(ct_paths) == 0:
        logging.error("No training data found. Check data directory structure.")
        return

    # Implement k-fold cross-validation
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(ct_paths), 1):
        logging.info(f"\n{'='*80}")
        logging.info(f"FOLD {fold_idx}/{num_folds}")
        logging.info(f"{'='*80}")

        # Prepare data
        train_ct = [ct_paths[i] for i in train_idx]
        train_seg = [seg_paths[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]

        val_ct = [ct_paths[i] for i in val_idx]
        val_seg = [seg_paths[i] for i in val_idx]
        val_labels = [labels[i] for i in val_idx]

        # Create datasets
        train_dataset = CTDataset(train_ct, train_seg, train_labels, patch_shape=(64, 64, 64))
        val_dataset = CTDataset(val_ct, val_seg, val_labels, patch_shape=(64, 64, 64))

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Train Model A (32×64×64)
        logging.info("\nTraining Model A (32×64×64)...")
        model_a = ModelA().to(device)
        train_dataset_a = CTDataset(train_ct, train_seg, train_labels, patch_shape=(32, 64, 64))
        train_loader_a = DataLoader(train_dataset_a, batch_size=batch_size, shuffle=True)
        train_model(model_a, train_loader_a, val_loader, num_epochs=num_epochs)
        torch.save(model_a.state_dict(), f'{output_dir}/best_model_patch_fold_{fold_idx}.pth')

        # Train Model B (64×64×64)
        logging.info("\nTraining Model B (64×64×64)...")
        model_b = ModelB().to(device)
        train_model(model_b, train_loader, val_loader, num_epochs=num_epochs)
        torch.save(model_b.state_dict(), f'{output_dir}/best_model_paper_fold_{fold_idx}.pth')

    logging.info("\n" + "="*80)
    logging.info("TRAINING COMPLETE")
    logging.info(f"Models saved to: {output_dir}")
    logging.info("="*80)

if __name__ == '__main__':
    main()
