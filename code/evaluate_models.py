#!/usr/bin/env python3
"""
Evaluate pre-trained models on test dataset.
"""
import torch
import torch.nn as nn
import json
import numpy as np
import nibabel as nib
import os
from pathlib import Path
from sklearn.metrics import roc_curve, auc, confusion_matrix
import logging

# Setup logging
log_dir = './results'
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'evaluation.log')
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
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
logging.info("="*80)
logging.info("MODEL EVALUATION")
logging.info("="*80)

# Configuration
test_data_dir = './data/test_dataset'
labels_file = './data/test_labels.json'
model_a_dir = './models/model_a_weights'
model_b_dir = './models/model_b_weights'

# Load labels
with open(labels_file, 'r') as f:
    test_data = json.load(f)

ct_paths = []
seg_paths = []
labels = []
patient_ids = []

for label_folder in ['Decomp', 'Nodecomp']:
    label_path = os.path.join(test_data_dir, label_folder)
    if not os.path.exists(label_path):
        continue

    for patient in sorted(os.listdir(label_path)):
        patient_dir = os.path.join(label_path, patient)
        if not os.path.isdir(patient_dir):
            continue

        ct_file = os.path.join(patient_dir, 'image.nii.gz')
        seg_file = None

        for f in os.listdir(patient_dir):
            if f.endswith('.nii.gz') and 'liver' in f.lower():
                seg_file = os.path.join(patient_dir, f)
                break

        if ct_file is not None and seg_file is not None:
            label = 1 if label_folder == 'Decomp' else 0
            ct_paths.append(ct_file)
            seg_paths.append(seg_file)
            labels.append(label)
            patient_ids.append(patient)

logging.info(f"\nLoaded {len(ct_paths)} test cases")

# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────────────────────────────────────
logging.info(f"Device: {device}")
logging.info("\nLoading Model A...")

models_a = []
for fold_idx in range(1, 6):
    model = ModelA().to(device)
    weight_path = f'{model_a_dir}/best_model_patch_fold_{fold_idx}.pth'
    if os.path.exists(weight_path):
        state_dict = torch.load(weight_path, map_location=device)
        model_state = model.state_dict()
        filtered_state = {k:v for k,v in state_dict.items() if k in model_state and v.size()==model_state[k].size()}
        model.load_state_dict(filtered_state, strict=False)
    model.eval()
    models_a.append(model)

logging.info("Loading Model B...")

models_b = []
for fold_idx in range(1, 6):
    model = ModelB().to(device)
    weight_path = f'{model_b_dir}/best_model_paper_fold_{fold_idx}.pth'
    if os.path.exists(weight_path):
        state_dict = torch.load(weight_path, map_location=device)
        model_state = model.state_dict()
        filtered_state = {k:v for k,v in state_dict.items() if k in model_state and v.size()==model_state[k].size()}
        model.load_state_dict(filtered_state, strict=False)
    model.eval()
    models_b.append(model)

# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────────────────────────────────────
def extract_patch(ct_data, patch_shape):
    d, h, w = ct_data.shape
    pd, ph, pw = patch_shape
    sd, sh, sw = max(0, (d - pd) // 2), max(0, (h - ph) // 2), max(0, (w - pw) // 2)
    ed, eh, ew = min(d, sd + pd), min(h, sh + ph), min(w, sw + pw)
    patch = ct_data[sd:ed, sh:eh, sw:ew]
    pad_d, pad_h, pad_w = max(0, pd - patch.shape[0]), max(0, ph - patch.shape[1]), max(0, pw - patch.shape[2])
    if pad_d > 0 or pad_h > 0 or pad_w > 0:
        patch = np.pad(patch, ((0, pad_d), (0, pad_h), (0, pad_w)), mode='constant')
    return patch

logging.info("\nRunning inference...")

probs_a_all = []
probs_b_all = []
labels_all = []

for i, patient_id in enumerate(patient_ids):
    try:
        ct_data = nib.load(ct_paths[i]).get_fdata()
        seg_data = nib.load(seg_paths[i]).get_fdata()

        d = min(ct_data.shape[2], seg_data.shape[2])
        ct_data = ct_data[:,:,:d]
        seg_data = seg_data[:,:,:d]

        img = ct_data * (seg_data > 0)
        if img.max() == 0:
            img = ct_data

        # Model A inference
        patch_a = extract_patch(img, (32, 64, 64))
        patch_a = np.clip(patch_a, -200, 300)
        patch_a = (patch_a - patch_a.mean()) / (patch_a.std() + 1e-8)
        patch_a_tensor = torch.FloatTensor(patch_a).unsqueeze(0).unsqueeze(0).to(device)

        fold_probs_a = []
        with torch.no_grad():
            for model in models_a:
                logits = model(patch_a_tensor)
                prob = torch.softmax(logits, dim=1)[0, 1].item()
                fold_probs_a.append(prob)
        probs_a_all.append(np.mean(fold_probs_a))

        # Model B inference
        patch_b = extract_patch(img, (64, 64, 64))
        patch_b = np.clip(patch_b, -200, 300)
        patch_b = (patch_b - patch_b.mean()) / (patch_b.std() + 1e-8)
        patch_b_tensor = torch.FloatTensor(patch_b).unsqueeze(0).unsqueeze(0).to(device)

        fold_probs_b = []
        with torch.no_grad():
            for model in models_b:
                logits = model(patch_b_tensor)
                prob = torch.softmax(logits, dim=1)[0, 1].item()
                fold_probs_b.append(prob)
        probs_b_all.append(np.mean(fold_probs_b))

        labels_all.append(labels[i])
    except Exception as e:
        logging.error(f"Error processing {patient_id}: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# COMPUTE METRICS
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_probs):
    y_pred = (np.array(y_probs) >= 0.5).astype(int)
    y_true = np.array(y_true)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
    return {'auc': roc_auc, 'accuracy': accuracy, 'sensitivity': sensitivity,
            'specificity': specificity, 'f1': f1, 'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp}

metrics_a = compute_metrics(labels_all, probs_a_all)
metrics_b = compute_metrics(labels_all, probs_b_all)

# ─────────────────────────────────────────────────────────────────────────────
# RESULTS
# ─────────────────────────────────────────────────────────────────────────────
logging.info("\n" + "="*80)
logging.info("EVALUATION RESULTS")
logging.info("="*80)

logging.info("\nModel A (32×64×64)")
logging.info(f"  Accuracy:    {metrics_a['accuracy']*100:.2f}%")
logging.info(f"  Sensitivity: {metrics_a['sensitivity']*100:.2f}%")
logging.info(f"  Specificity: {metrics_a['specificity']*100:.2f}%")
logging.info(f"  Precision:   {metrics_a['precision']*100:.2f}%")
logging.info(f"  F1-Score:    {metrics_a['f1']:.4f}")
logging.info(f"  AUC:         {metrics_a['auc']:.4f}")

logging.info("\nModel B (64×64×64)")
logging.info(f"  Accuracy:    {metrics_b['accuracy']*100:.2f}%")
logging.info(f"  Sensitivity: {metrics_b['sensitivity']*100:.2f}%")
logging.info(f"  Specificity: {metrics_b['specificity']*100:.2f}%")
logging.info(f"  Precision:   {metrics_b['precision']*100:.2f}%")
logging.info(f"  F1-Score:    {metrics_b['f1']:.4f}")
logging.info(f"  AUC:         {metrics_b['auc']:.4f}")

logging.info("\n" + "="*80)
logging.info(f"Log saved to: {log_file}")
logging.info("="*80)
