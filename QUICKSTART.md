# Quick Start Guide

## 5-Minute Setup

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/liver-segmentation-evaluation.git
cd liver-segmentation-evaluation
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download Model Weights
The evaluation requires pre-trained model weights. These are referenced in the evaluation scripts:
- Model A weights: `best_model_patch_fold_*.pth`
- Model B weights: `best_model_paper_fold_*.pth`

Update the paths in the evaluation scripts to point to your model weights location.

### 4. Run Evaluation
```bash
python code/evaluate_models.py
```

Expected output:
```
EVALUATION RESULTS

Model A (32×64×64)
  Accuracy:    92.86%
  Sensitivity: 100.00%
  Specificity: 84.62%
  Precision:   88.24%
  F1-Score:    0.9375
  AUC:         0.9769

Model B (64×64×64)
  Accuracy:    96.43%
  Sensitivity: 100.00%
  Specificity: 92.31%
  Precision:   93.75%
  F1-Score:    0.9677
  AUC:         0.9974
```

## File Organization

```
liver-segmentation-evaluation/
├── code/                          # Analysis and training scripts
│   ├── preprocess_with_totalsegmentation.py
│   ├── train_models.py
│   └── evaluate_models.py
├── data/
│   ├── raw_images/               # Raw CT images
│   ├── segmentations/             # Generated liver masks
│   └── test_dataset/              # Organized test dataset
├── figures/                        # Publication-ready figures
│   ├── confusion_matrices.png
│   ├── roc_curves.png
│   ├── performance_metrics.png
│   └── probability_distribution_fixed.png
├── docs/                          # Documentation
│   ├── DATASET.md
│   ├── METHODS.md
│   ├── RESULTS.md
│   └── QUICKSTART.md (this file)
└── README.md
```

## Understanding the Code

### preprocess_with_totalsegmentation.py
Generates automatic liver segmentation masks from CT images.

Key functions:
- `preprocess_with_totalsegmentation()`: Runs TotalSegmentation on all CT images
- `organize_by_class()`: Organizes files into Decomp/Nodecomp folder structure

### train_models.py
Trains ModelA and ModelB using 5-fold cross-validation.

Key components:
1. **Model Architecture** (lines 15-50): ModelA and ModelB classes
2. **CTDataset** (lines 73-130): Data loader for CT patches
3. **Training Loop** (lines 172-210): 5-fold CV training with validation
4. **Main Pipeline** (lines 254-310): Full training workflow

### evaluate_models.py
General-purpose evaluation script for both models.

Key sections:
1. **Model Architecture** (lines 15-50): Define ModelA and ModelB classes
2. **Data Loading** (lines 55-85): Load CT images and segmentation masks
3. **Model Loading** (lines 90-120): Load pre-trained weights from 5 folds
4. **Inference** (lines 145-190): Run predictions and ensemble averaging
5. **Metrics** (lines 193-205): Compute evaluation metrics

## Key Results at a Glance

| Metric | Model A | Model B |
|--------|---------|---------|
| Accuracy | 92.86% | **96.43%** ✓ |
| Sensitivity | 100% | **100%** ✓ |
| Specificity | 84.62% | **92.31%** ✓ |
| AUC | 0.9769 | **0.9974** ✓ |

**Recommendation**: Use Model B for best performance

## Common Questions

### Q: How do I use my own test data?
A: Modify the data paths in the evaluation scripts. Follow the same folder structure:
```
your_data/
├── Decomp/
│   ├── PatientID/
│   │   ├── image.nii.gz
│   │   └── liver.nii.gz
└── Nodecomp/
    └── ...
```

### Q: How do I access the model weights?
A: The weights are not included in the repository. You need to:
1. Train your own models using the training code
2. Or, contact the authors for the pre-trained weights

### Q: Can I use different patch sizes?
A: Yes, modify the patch size in the `extract_patch()` calls:
```python
# For Model A: change 32 to your desired size
patch = extract_patch(img, (32, 64, 64))

# For Model B: change 64 to your desired size
patch = extract_patch(img, (64, 64, 64))
```

### Q: How do I interpret the confusion matrix?
A:
- **TP (True Positives)**: Correctly identified Decomp cases
- **TN (True Negatives)**: Correctly identified NonDecomp cases
- **FP (False Positives)**: NonDecomp cases incorrectly predicted as Decomp
- **FN (False Negatives)**: Decomp cases incorrectly predicted as NonDecomp

### Q: What does AUC mean?
A: AUC (Area Under the ROC Curve) measures overall discrimination ability:
- 0.5 = Random guessing
- 1.0 = Perfect classification
- Our models: 0.9769-0.9974 = Excellent

## Troubleshooting

### Import Error: nibabel not found
```bash
pip install nibabel
```

### CUDA Out of Memory
The models run on GPU by default. If you don't have a GPU:
```python
# Change this line in the scripts:
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# It will automatically fall back to CPU
```

### Path Not Found Error
Update the paths in the scripts to match your system:
```python
test_data_dir = '/your/path/to/test_dataset'
model_a_dir = '/your/path/to/model/weights'
```

## Next Steps

1. **Read the Paper**: Check the published paper for detailed methodology
2. **Review Results**: See `docs/RESULTS.md` for detailed analysis
3. **Check Methods**: See `docs/METHODS.md` for technical details
4. **Explore Data**: See `docs/DATASET.md` for dataset information

## Citation

If you use this code or results, please cite:

```bibtex
@article{yourpaper2026,
  title={High-Accuracy Liver Segmentation Using 3D Convolutional Neural Networks},
  author={Your Name and Co-authors},
  journal={Your Journal},
  year={2026}
}
```

## Support

- **Issues**: Open a GitHub issue for bugs or questions
- **Discussions**: Use GitHub Discussions for general questions
- **Email**: Contact the authors for detailed inquiries

---

Happy analyzing! 🎉
