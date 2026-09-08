# Methodology

## Model Architectures

### Backbone: ResNet3D-18
- Pre-trained on video datasets (transferred learning)
- 3D convolutions for volumetric data
- Modified input layer for single-channel CT

### Model A Configuration
```python
Encoder:
  - ResNet3D-18 backbone
  - Modified Conv3d: (1, 64) instead of (3, 64)
  - Output: 512-dimensional features

Classifier:
  - Single Linear layer: 512 → 2 (binary classification)
  
Patch Size: 32 × 64 × 64 voxels
```

### Model B Configuration
```python
Encoder:
  - ResNet3D-18 backbone (same as Model A)
  - Output: 512-dimensional features

Projection Layer:
  - Linear(512 → 1024)
  - ReLU activation
  - Linear(1024 → 2048)

Classifier:
  - Linear(512 → 256)
  - ReLU activation
  - Dropout(p=0.3)
  - Linear(256 → 2)

Patch Size: 64 × 64 × 64 voxels
```

## Training Details

### Data Preprocessing
```python
# HU clipping
patch = np.clip(patch, -200, 300)

# Z-score normalization
patch = (patch - patch.mean()) / (patch.std() + 1e-8)
```

### Patch Extraction
- **Method**: Center extraction from 3D volume
- **Center Calculation**: (d//2, h//2, w//2)
- **Padding**: Zero-padding if volume < patch size

### Ensemble Strategy
- **Cross-Validation**: 5-fold stratified split
- **Ensemble Method**: Average probability across all 5 folds
- **Decision Threshold**: 0.5 (standard binary classification)

## Evaluation Protocol

### Test Set
- **Independence**: Completely held out during training
- **Composition**: Mix of decompensated and non-decompensated cases
- **Strategy**: Fixed test set evaluation (not cross-validated)

### Metrics Computed
1. **Accuracy**: (TP + TN) / Total
2. **Sensitivity**: TP / (TP + FN) - Recall for positive class
3. **Specificity**: TN / (TN + FP) - Recall for negative class
4. **Precision**: TP / (TP + FP) - Positive predictive value
5. **F1-Score**: 2 × (Precision × Sensitivity) / (Precision + Sensitivity)
6. **AUC-ROC**: Area under receiver operating characteristic curve
7. **Confusion Matrix**: Breakdown of predictions

## Statistical Analysis

### Sensitivity & Specificity
- Analyzed per class
- Critical for medical applications
- 100% sensitivity ensures no missed cases

### ROC Analysis
- Varying decision threshold (0 to 1)
- AUC summarizes overall discrimination ability
- Model B: AUC = 0.9974 (near-perfect)

## Reproducibility

### Fixed Parameters
- Random seed: Fixed for deterministic results
- Input normalization: Standardized across all samples
- Model weights: Pre-trained, not fine-tuned

### Version Control
- Code commits with detailed messages
- Evaluation logs saved automatically
- Results timestamps for tracking
