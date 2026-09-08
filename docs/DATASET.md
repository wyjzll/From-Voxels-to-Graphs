# Dataset Documentation

## Overview
The test dataset consists of CT scans with corresponding liver segmentation masks, split into two cohorts:
- **Original cohort**: Primary test cases
- **Additional validation cohort**: Validation cases

## Class Distribution
- **Decompensated (positive)**: Multiple cases
- **Non-Decompensated (negative)**: Multiple cases
- Dataset is relatively balanced between classes

## Data Format
- **CT Images**: 3D volumes in NIfTI format (.nii.gz)
- **Segmentation Masks**: Binary masks indicating liver region (.nii.gz)
- **Organization**: Organized by class (Decomp/Nodecomp)

## File Structure
```
test_dataset/
├── Decomp/                    (Decompensated cases)
│   ├── patient_001/
│   │   ├── image.nii.gz      (CT volume)
│   │   └── liver.nii.gz      (Segmentation mask)
│   ├── patient_002/
│   └── ... (additional decompensated cases)
│
└── Nodecomp/                 (Non-Decompensated cases)
    ├── patient_101/
    │   ├── image.nii.gz
    │   └── liver.nii.gz
    ├── patient_102/
    └── ... (additional non-decompensated cases)
```

## Image Specifications
- **Modality**: Computed Tomography (CT)
- **Format**: NIfTI (.nii.gz)
- **Spatial Resolution**: Varies by patient (typical: 512×512×~150 slices)
- **HU Range**: -500 to +500 (clipped to -200 to +300 for processing)

## Preprocessing Applied
1. **HU Clipping**: -200 to 300 HU
2. **Masking**: Liver mask applied to isolate region
3. **Normalization**: Z-score per image (mean=0, std=1)

## Data Access

**Important:** The actual patient CT images and segmentation masks are **NOT included** in this repository for privacy and compliance reasons.

To run the evaluation:

1. **Obtain the dataset** from the original data source
2. **Organize your data** according to the file structure shown above
3. **Place your data** in: `./data/test_dataset/`
4. **Update paths** in the evaluation scripts to point to your data location

Example data organization:
```
./data/
└── test_dataset/
    ├── Decomp/
    │   ├── patient_001/
    │   │   ├── image.nii.gz
    │   │   └── liver.nii.gz
    │   └── ... (more patients)
    └── Nodecomp/
        ├── patient_101/
        │   ├── image.nii.gz
        │   └── liver.nii.gz
        └── ... (more patients)
```

## Ethical Considerations
- De-identified patient data
- IRB approved for research use
- Restrictions on commercial use apply

## Citation
If using this dataset, please cite:
```
@dataset{liverct2026,
  title={Liver CT Segmentation Test Dataset},
  author={Your Name},
  year={2026}
}
```
