# Environment and Dependencies

## System Requirements
- **OS**: Linux (tested on Linux 4.18.0)
- **Python**: 3.10+
- **CUDA**: 11.8 (optional, for GPU acceleration)
- **RAM**: 8GB minimum (16GB+ recommended)
- **GPU**: NVIDIA GPU with CUDA support (CPU fallback available)

## Package Versions

### Core Deep Learning Framework
| Package | Version | Purpose |
|---------|---------|---------|
| torch | 2.7.1 | Neural network training and inference |
| torchvision | 0.22.1 | Computer vision utilities |
| torchaudio | 2.7.1 | Audio processing (included for completeness) |

### Scientific Computing
| Package | Version | Purpose |
|---------|---------|---------|
| numpy | 2.2.6 | Numerical computing |
| scipy | 1.15.3 | Scientific computing functions |
| pandas | 2.3.3 | Data manipulation and analysis |
| scikit-learn | 1.7.2 | Machine learning utilities (metrics, cross-validation) |
| scikit-image | 0.25.2 | Image processing |

### Medical Imaging
| Package | Version | Purpose |
|---------|---------|---------|
| nibabel | 5.3.3 | NIfTI file I/O |
| SimpleITK | 2.5.3 | Image registration and processing |
| monai | 1.5.2 | Medical imaging deep learning |
| TotalSegmentator | 2.12.0 | Automatic organ segmentation |

### Visualization and Utilities
| Package | Version | Purpose |
|---------|---------|---------|
| matplotlib | 3.10.8 | Plotting and visualization |
| seaborn | 0.13.2 | Statistical visualization |
| pillow | 12.0.0 | Image processing |
| tqdm | 4.67.3 | Progress bars |
| PyYAML | 6.0.3 | YAML file handling |

### GPU Support (CUDA 11.8)
| Package | Version |
|---------|---------|
| nvidia-cublas-cu11 | 11.11.3.6 |
| nvidia-cudnn-cu11 | 9.1.0.70 |
| nvidia-cuda-runtime-cu11 | 11.8.89 |

## Installation Methods

### Option 1: Using requirements.txt (Pip)
```bash
pip install -r requirements.txt
```

### Option 2: Using environment.yml (Conda)
```bash
conda env create -f environment.yml
conda activate liver-segmentation
```

### Option 3: Manual installation
```bash
# Install PyTorch with CUDA support
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt
```

## Reproducibility

All models in this repository were trained and evaluated using the exact package versions listed in `environment.yml`. To ensure reproducibility:

1. Use Python 3.10+
2. Install packages from `requirements.txt` or `environment.yml`
3. All results should be reproducible within floating-point precision
4. Random seeds are fixed in the code (numpy, torch)

## Optional Packages

Some packages are optional depending on your workflow:

- **TotalSegmentator**: Only needed if generating segmentation masks from scratch
- **monai**: Useful for additional medical imaging utilities
- **jupyter**: For interactive exploration of results

If you only need to run inference with pre-trained models, you can use a minimal setup:

```bash
pip install torch torchvision nibabel numpy scikit-learn matplotlib
```

## Version History

| Date | Python | PyTorch | CUDA | Notes |
|------|--------|---------|------|-------|
| 2026-09-08 | 3.10 | 2.7.1 | 11.8 | Current stable version |

## Known Issues and Workarounds

### Out of Memory
If you encounter CUDA out of memory errors:
- Reduce batch size in training scripts
- Use CPU mode: Set `device = torch.device('cpu')`
- Reduce patch size (32×64×64 uses less memory than 64×64×64)

### M1/M2 Mac Users
TotalSegmentator may not work natively on Apple Silicon. Consider:
- Using Docker
- Running on CPU only
- Using a cloud GPU instance

### Windows
Some packages may have compatibility issues. Consider:
- Using Windows Subsystem for Linux (WSL2)
- Using Docker
- Installing dependencies one by one with troubleshooting

## References

- [PyTorch Official](https://pytorch.org/)
- [NiBabel Documentation](https://nipy.org/nibabel/)
- [TotalSegmentator](https://github.com/wasserth/TotalSegmentator)
- [MONAI](https://monai.io/)
