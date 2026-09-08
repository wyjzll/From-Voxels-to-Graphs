# Liver Segmentation Model Evaluation on CT Scans

## Overview

This repository contains the code and results for liver segmentation models using 3D convolutional neural networks (ResNet3D-18). The project evaluates two model variants on a test set with complete liver segmentation masks, achieving **96.43% accuracy** (Model B) with **100% sensitivity** and **92.31% specificity**.

## Key Results

### Model Performance (Test Set)

| Model | Architecture | Accuracy | Sensitivity | Specificity | AUC | Precision |
|-------|--------------|----------|-------------|------------|-----|-----------|
| **Model A** | 32×64×64 patches | 92.86% | 100.0% | 84.62% | 0.9769 | 88.24% |
| **Model B** | 64×64×64 patches | **96.43%** | **100.0%** | **92.31%** | **0.9974** | **93.75%** |

### Clinical Significance
- **100% Sensitivity**: All decompensated cases correctly identified (no false negatives)
- **>92% Specificity**: Minimal false alarms with high specificity
- **Outstanding AUC**: Model B achieves near-perfect discrimination between classes (AUC 0.9974)

## Dataset

**Test Set Composition:**
- Both decompensated and non-decompensated cases
- All with complete CT images and liver segmentation masks

**Data split:**
- Original test cohort
- Additional validation cohort

**File Format:**
- CT images: NIfTI format (.nii.gz)
- Liver masks: NIfTI format (.nii.gz)
- Located in: `data/test_dataset/`

## Model Architecture

### Model A: Custom Parameters
```
Input → ResNet3D-18 Backbone → 512-dim features → Linear Classifier → 2 classes
- Patch Size: 32 × 64 × 64
- Classifier: Single Linear Layer (512 → 2)
- Ensemble: 5-fold cross-validation averaging
```

### Model B: Paper Parameters
```
Input → ResNet3D-18 Backbone → 512-dim features → Projection Layer → Multi-layer Classifier → 2 classes
- Patch Size: 64 × 64 × 64
- Projection: Linear (512 → 1024) + ReLU + Linear (1024 → 2048)
- Classifier: Linear (512 → 256) + ReLU + Dropout(0.3) + Linear (256 → 2)
- Ensemble: 5-fold cross-validation averaging
```

## Data Availability

**Important Privacy Note:** The actual patient CT images and liver segmentation masks are **NOT included** in this repository for privacy protection and regulatory compliance. 

The repository contains:
- ✓ Complete evaluation code
- ✓ Documentation and methods
- ✓ Publication-ready figures
- ✗ Patient images/masks (require separate access from authorized data sources)

To reproduce the evaluation, you will need to obtain the test dataset from appropriate sources with proper IRB approval and data use agreements.

See `docs/DATASET.md` for detailed data requirements and organization.

## Installation

### Requirements
- Python 3.8+
- PyTorch 2.0+
- CUDA 11.0+ (for GPU acceleration)
- TotalSegmentation (for automatic liver segmentation)

```bash
pip install torch torchvision nibabel scikit-learn totalsegmentator
```

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/liver-segmentation-evaluation.git
cd liver-segmentation-evaluation

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Preprocessing: Generate Liver Segmentations

```bash
# Generate liver segmentation masks using TotalSegmentation
python code/preprocess_with_totalsegmentation.py
```

Input structure:
```
./data/
├── raw_images/
│   ├── case1.nii.gz
│   ├── case2.nii.gz
│   └── ...
```

Output:
```
./data/
└── segmentations/
    ├── case1_liver.nii.gz
    ├── case2_liver.nii.gz
    └── ...
```

### 2. Training Models (Optional)

```bash
# Train models using 5-fold cross-validation
python code/train_models.py
```

Expected data structure:
```
./data/
└── train_dataset/
    ├── Decomp/
    │   ├── case1/
    │   │   ├── image.nii.gz
    │   │   └── liver.nii.gz
    │   └── ...
    └── Nodecomp/
        ├── case2/
        │   ├── image.nii.gz
        │   └── liver.nii.gz
        └── ...
```

### 3. Model Evaluation

```bash
# Evaluate both models on test dataset
python code/evaluate_models.py
```

### Expected Output
- Console logs with detailed metrics
- Log files saved in the `results/` directory
- Evaluation metrics printed to stdout

## Project Structure

```
liver-segmentation-evaluation/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── code/
│   ├── preprocess_with_totalsegmentation.py  # Generate liver masks
│   ├── train_models.py               # Training pipeline (5-fold CV)
│   └── evaluate_models.py            # Model evaluation script
├── data/
│   └── test_dataset/                         # Test dataset
│       ├── Decomp/          (Decompensated cases with CT + masks)
│       └── Nodecomp/        (Non-decompensated cases with CT + masks)
├── figures/
│   ├── confusion_matrices.png                # Confusion matrix heatmaps
│   ├── roc_curves.png                        # ROC curves for both models
│   ├── performance_metrics.png               # Performance comparison
│   └── probability_distribution.png          # Predicted probability distributions
├── docs/
│   ├── METRICS_FOR_PAPER_SUBMISSION.md       # Comprehensive metrics
│   ├── PAPER_SUBMISSION_TABLES.txt           # Publication tables
│   └── FINAL_ROOT_CAUSE_ANALYSIS.md          # Investigation details
└── results/
    └── [Generated evaluation logs]
```

## Data Preprocessing

### CT Image Processing
- **Normalization**: Hounsfield Unit (HU) clipping: -200 to 300 HU
- **Masking**: Liver segmentation mask applied to isolate region of interest
- **Standardization**: Per-image z-score normalization: `(patch - mean) / (std + 1e-8)`

### Patch Extraction
- **Method**: Center extraction from 3D CT volume
- **Padding**: Zero-padding applied if volume smaller than patch size
- **Consistency**: Identical extraction method for both training and testing

## Model Training

Models were trained using 5-fold cross-validation on a separate training set. The evaluation uses the trained weights:

- **Model A**: `best_model_patch_fold_*.pth` (custom parameters)
- **Model B**: `best_model_paper_fold_*.pth` (paper parameters)

Predictions are obtained by averaging across all 5 folds for robustness.

## Evaluation Metrics

### Metrics Computed
1. **Accuracy**: (TP + TN) / (TP + TN + FP + FN)
2. **Sensitivity (Recall)**: TP / (TP + FN) - True positive rate
3. **Specificity**: TN / (TN + FP) - True negative rate
4. **Precision**: TP / (TP + FP) - Positive predictive value
5. **F1-Score**: 2 × (Precision × Sensitivity) / (Precision + Sensitivity)
6. **AUC-ROC**: Area under receiver operating characteristic curve
7. **Confusion Matrix**: TP, TN, FP, FN counts

### Classification Threshold
- **Decision Threshold**: 0.5 (standard probability threshold)
- **Ensemble**: Average probability across 5 folds

## Results Visualization

### Figure 1: Confusion Matrices
Shows the distribution of true positives, false positives, true negatives, and false negatives for both models. Model B demonstrates superior specificity with only 1 false positive.

### Figure 2: ROC Curves
Receiver operating characteristic curves demonstrate excellent discrimination ability for both models, with Model B achieving exceptional performance (AUC = 0.9974).

### Figure 3: Performance Metrics
Bar chart comparison of accuracy, sensitivity, specificity, and AUC for both models side-by-side.

### Figure 4: Probability Distribution
Distribution of predicted probabilities separated by true class labels, showing clear separation between classes for both models.

## Reproducibility

All evaluation scripts are fully documented and reproducible:

- **Fixed Random Seed**: Set for consistent results
- **Complete Documentation**: Detailed comments in all code
- **Parameter Logging**: All hyperparameters logged to output
- **Independent Test Set**: Completely held out during training
- **Evaluation Log**: Detailed log file with all metrics saved

To reproduce the results:
```bash
python code/eval_collab_on_cleaned_dataset.py
# Check results in console output and results/ directory
```


## Publications and Citations

If you use this code or dataset, please cite:

```bibtex
@article{yourpaper2026,
  title={High-Accuracy Liver Segmentation Using 3D Convolutional Neural Networks},
  author={Your Name and Collaborators},
  journal={Medical Image Analysis},
  year={2026}
}
```

## Data Availability

The test dataset with CT images and liver segmentation masks is available in the `data/test_dataset/` directory. Dataset consists of:
- CT images for all test cases
- Corresponding liver segmentation masks
- All data in NIfTI format (.nii.gz)

## Model Weights

Pre-trained model weights are required to run the evaluation:
- **Model A**: `best_model_patch_fold_*.pth` (5 folds)
- **Model B**: `best_model_paper_fold_*.pth` (5 folds)

Pre-trained model weights are required to run the evaluation. Update the paths in the evaluation scripts to point to your model directory.

## License

[MIT License / Your License Here]

## Contact

For questions or issues, please:
1. Open an issue on GitHub
2. Contact the authors directly
3. See the paper supplementary materials for additional details

## Acknowledgments

We thank the institutions that provided the medical imaging data and the support for this research.

---

**Last Updated**: September 8, 2026  
**Status**: ✓ Ready for peer review and publication
