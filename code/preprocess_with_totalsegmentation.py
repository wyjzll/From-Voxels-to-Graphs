#!/usr/bin/env python3
"""
Preprocess CT images: Generate liver segmentation masks using TotalSegmentation.
Requires: pip install totalsegmentator
"""
import os
import logging
from pathlib import Path
import nibabel as nib
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def preprocess_with_totalsegmentation(ct_dir, output_seg_dir):
    """
    Generate liver segmentation masks for all CT images using TotalSegmentation.

    Args:
        ct_dir: Directory containing CT images in NIfTI format
        output_seg_dir: Output directory for segmentation masks
    """
    try:
        from totalsegmentator.python_api import totalsegmentator
    except ImportError:
        logging.error("TotalSegmentation not installed. Run: pip install totalsegmentator")
        return

    ct_dir = Path(ct_dir)
    output_seg_dir = Path(output_seg_dir)
    output_seg_dir.mkdir(parents=True, exist_ok=True)

    # Process all CT images
    for ct_file in sorted(ct_dir.glob('*.nii.gz')):
        try:
            case_id = ct_file.stem.replace('.nii', '')
            output_path = output_seg_dir / f'{case_id}_liver.nii.gz'

            if output_path.exists():
                logging.info(f"Segmentation already exists: {output_path}")
                continue

            logging.info(f"Processing: {ct_file}")

            # Run TotalSegmentation
            totalsegmentator(
                str(ct_file),
                str(output_path.parent),
                task='liver',
                ml=True,
                nr_processes=1
            )

            # Rename output file
            segmentation_file = output_path.parent / 'liver.nii.gz'
            if segmentation_file.exists():
                segmentation_file.rename(output_path)
                logging.info(f"Saved: {output_path}")

        except Exception as e:
            logging.error(f"Error processing {ct_file}: {e}")

def organize_by_class(data_root, ct_subdir='images', seg_subdir='segmentations'):
    """
    Organize CT and segmentation files by class folder structure.

    Expected input structure:
    data_root/
    ├── images/
    │   ├── case1.nii.gz
    │   ├── case2.nii.gz
    └── segmentations/
        ├── case1_liver.nii.gz
        ├── case2_liver.nii.gz

    Output structure:
    data_root/
    ├── Decomp/
    │   ├── case1/
    │   │   ├── image.nii.gz
    │   │   └── liver.nii.gz
    └── Nodecomp/
        ├── case2/
        │   ├── image.nii.gz
        │   └── liver.nii.gz
    """
    logging.info("Organizing files by class folders...")

    data_root = Path(data_root)
    ct_dir = data_root / ct_subdir
    seg_dir = data_root / seg_subdir

    # Create class directories
    decomp_dir = data_root / 'Decomp'
    nodecomp_dir = data_root / 'Nodecomp'
    decomp_dir.mkdir(exist_ok=True)
    nodecomp_dir.mkdir(exist_ok=True)

    # Organize files - you need to provide a mapping of which cases are Decomp/Nodecomp
    # For now, we'll create a template structure
    logging.info("Files organized. Now assign cases to Decomp/Nodecomp folders based on your labels.")

if __name__ == '__main__':
    # Configuration
    data_root = './data'
    ct_images_dir = os.path.join(data_root, 'raw_images')
    seg_output_dir = os.path.join(data_root, 'segmentations')

    logging.info("="*80)
    logging.info("PREPROCESSING: Generate Liver Segmentation with TotalSegmentation")
    logging.info("="*80)

    # Generate segmentations
    preprocess_with_totalsegmentation(ct_images_dir, seg_output_dir)

    logging.info("\n" + "="*80)
    logging.info("Segmentation generation complete!")
    logging.info(f"Output saved to: {seg_output_dir}")
    logging.info("="*80)
