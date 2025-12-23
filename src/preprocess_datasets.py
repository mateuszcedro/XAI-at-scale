#!/usr/bin/env python3
"""
Medical Image Dataset Preprocessing Script

Each dataset is downloaded from Google Drive and preprocessed separately.
Preprocessing includes: resizing, normalization, histogram equalization, and quality checks.

Author: Mateusz Cedro
"""

import os
import sys
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Optional, Dict
import shutil
import warnings
from tqdm import tqdm
from PIL import Image
import logging
import torch
import torch.nn as nn
from torchvision import transforms

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore")


# Configuration class
class Config:
    """Configuration for image preprocessing"""
    IMG_SIZE = 224  # Standard image size for medical imaging
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_transforms():
    """Create train and test data transformations using torchvision.transforms."""
    transforms_train = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.Grayscale(num_output_channels=1),
        transforms.CenterCrop(Config.IMG_SIZE),
        transforms.RandomRotation(degrees=15),
        transforms.GaussianBlur(kernel_size=3),
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
    ])

    transforms_test = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.Grayscale(num_output_channels=1),
        transforms.CenterCrop(Config.IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
    ])
    
    return transforms_train, transforms_test


def get_augmentation_transforms():
    """Create augmented transformations for data augmentation."""
    transforms_aug = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.Grayscale(num_output_channels=1),
        transforms.CenterCrop(Config.IMG_SIZE),
        transforms.RandomRotation(degrees=15),
        transforms.GaussianBlur(kernel_size=3),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    return transforms_aug



class DatasetPreprocessor:
    """Base class for dataset preprocessing"""
    
    def __init__(self, dataset_name: str, raw_dir: str, processed_dir: str, target_size: Tuple[int, int] = (224, 224),
                 apply_augmentation: bool = False, augmentation_factor: int = 2):
        """
        Initialize the preprocessor.
        
        Args:
            dataset_name: Name of the dataset
            raw_dir: Directory containing raw images
            processed_dir: Directory to save processed images
            target_size: Target image size for resizing
            apply_augmentation: Whether to apply data augmentation
            augmentation_factor: Number of augmented variants per image
        """
        self.dataset_name = dataset_name
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.target_size = target_size
        self.apply_augmentation = apply_augmentation
        self.augmentation_factor = augmentation_factor
        
        # Get transforms from the get_transforms() function
        self.train_transform, self.test_transform = get_transforms()
        self.augment_transform = get_augmentation_transforms()
        
        # Create processed directory if it doesn't exist
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized preprocessor for {dataset_name}")
        logger.info(f"Raw directory: {self.raw_dir}")
        logger.info(f"Processed directory: {self.processed_dir}")
        logger.info(f"Augmentation enabled: {self.apply_augmentation}")
    
    def load_image(self, image_path: str) -> Optional[np.ndarray]:
        """
        Load an image from file.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Image as numpy array or None if loading fails
        """
        try:
            img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                logger.warning(f"Failed to load image: {image_path}")
                return None
            return img
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            return None
    
    def resize_image(self, img: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        """Resize image to target size"""
        return cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)
    
    def normalize_image(self, img: np.ndarray) -> np.ndarray:
        """Normalize image to [0, 1] range"""
        img = img.astype(np.float32)
        return img / 255.0
    
    def apply_clahe(self, img: np.ndarray, clip_limit: float = 2.0, tile_size: int = 8) -> np.ndarray:
        """
        Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).
        
        Args:
            img: Input image (grayscale)
            clip_limit: Threshold for contrast limiting
            tile_size: Size of grid tiles
            
        Returns:
            Enhanced image
        """
        img_uint8 = (img * 255).astype(np.uint8) if img.max() <= 1 else img.astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
        return clahe.apply(img_uint8).astype(np.float32) / 255.0
    
    def apply_gaussian_blur(self, img: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """Apply Gaussian blur for noise reduction"""
        img_uint8 = (img * 255).astype(np.uint8) if img.max() <= 1 else img.astype(np.uint8)
        blurred = cv2.GaussianBlur(img_uint8, (kernel_size, kernel_size), 0)
        return blurred.astype(np.float32) / 255.0
    
    def preprocess_image(self, img: np.ndarray) -> np.ndarray:
        """
        Apply preprocessing pipeline to image using torchvision transforms.
        
        Args:
            img: Input image (numpy array)
            
        Returns:
            Preprocessed image as tensor (converted back to numpy for compatibility)
        """
        # Convert numpy array to PIL Image
        img_pil = Image.fromarray(img)
        
        # Apply test transform (no augmentation in base preprocessing)
        img_tensor = self.test_transform(img_pil)
        
        # Convert back to numpy array for compatibility
        img_processed = img_tensor.numpy().squeeze()
        
        return img_processed
    
    def save_image(self, img: np.ndarray, output_path: str) -> bool:
        """
        Save preprocessed image.
        
        Args:
            img: Image to save
            output_path: Path to save the image
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Convert to uint8 for saving
            img_uint8 = (img * 255).astype(np.uint8) if img.max() <= 1 else img.astype(np.uint8)
            cv2.imwrite(str(output_path), img_uint8)
            return True
        except Exception as e:
            logger.error(f"Error saving image {output_path}: {e}")
            return False
    
    def process_dataset(self, organize_by_class: bool = False) -> Dict[str, any]:
        """
        Process all images in the dataset.
        
        Args:
            organize_by_class: If True, organize images in class subdirectories
            
        Returns:
            Dictionary with processing statistics
        """
        stats = {
            'total': 0,
            'processed': 0,
            'failed': 0,
            'skipped': 0
        }
        
        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp'}
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(self.raw_dir.rglob(f'*{ext}'))
            image_files.extend(self.raw_dir.rglob(f'*{ext.upper()}'))
        
        stats['total'] = len(image_files)
        logger.info(f"Found {stats['total']} images in {self.dataset_name}")
        
        if stats['total'] == 0:
            logger.warning(f"No images found in {self.raw_dir}")
            return stats
        
        # Process images
        for image_path in tqdm(image_files, desc=f"Processing {self.dataset_name}"):
            try:
                # Load image
                img = self.load_image(str(image_path))
                if img is None:
                    stats['failed'] += 1
                    continue
                
                # Skip if image is too small
                if img.shape[0] < 50 or img.shape[1] < 50:
                    logger.warning(f"Skipping too small image: {image_path}")
                    stats['skipped'] += 1
                    continue
                
                # Preprocess
                processed_img = self.preprocess_image(img)
                
                # Determine output path
                if organize_by_class:
                    class_name = image_path.parent.name
                    output_dir = self.processed_dir / class_name
                    output_dir.mkdir(parents=True, exist_ok=True)
                else:
                    output_dir = self.processed_dir
                
                # Save original processed image
                output_path = output_dir / image_path.name
                
                if self.save_image(processed_img, str(output_path)):
                    stats['processed'] += 1
                else:
                    stats['failed'] += 1
                
                # Apply augmentation if enabled
                if self.apply_augmentation:
                    augmented_variants = self.augment_image(processed_img)
                    
                    # Save augmented variants
                    for idx, (aug_img, aug_type) in enumerate(augmented_variants):
                        # Create filename with augmentation type
                        name_parts = image_path.stem.split('.')
                        aug_filename = f"{name_parts[0]}_aug_{aug_type}_{idx}{image_path.suffix}"
                        aug_output_path = output_dir / aug_filename
                        
                        if self.save_image(aug_img, str(aug_output_path)):
                            stats['processed'] += 1
                        else:
                            stats['failed'] += 1
                    
            except Exception as e:
                logger.error(f"Error processing {image_path}: {e}")
                stats['failed'] += 1
        
        return stats


class PetPreprocessor(DatasetPreprocessor):
    """Preprocessor for PET dataset"""
    
    def __init__(self, raw_dir: str, processed_dir: str):
        super().__init__("PET", raw_dir, processed_dir)
    
    def preprocess_image(self, img: np.ndarray) -> np.ndarray:
        """PET specific preprocessing using torchvision transforms"""
        # Convert numpy array to PIL Image
        img_pil = Image.fromarray(img)
        
        # Apply test transform (includes resize, grayscale, normalization)
        img_tensor = self.test_transform(img_pil)
        
        # Convert back to numpy array
        img_processed = img_tensor.numpy().squeeze()
        
        return img_processed


# class PetAnimalPreprocessor(DatasetPreprocessor):
#     """Preprocessor for Pet/Animal dataset (cats and dogs)"""
    
#     def __init__(self, raw_dir: str, processed_dir: str):
#         super().__init__("Pet/Animal", raw_dir, processed_dir)
    
#     def preprocess_image(self, img: np.ndarray) -> np.ndarray:
#         """Pet/Animal specific preprocessing using torchvision transforms"""
#         # Convert numpy array to PIL Image
#         img_pil = Image.fromarray(img)
        
#         # Apply test transform (includes resize, grayscale, normalization)
#         img_tensor = self.test_transform(img_pil)
        
#         # Convert back to numpy array
#         img_processed = img_tensor.numpy().squeeze()
        
#         return img_processed


def preprocess_all_datasets(base_raw_dir: str, base_processed_dir: str, apply_augmentation: bool = False, dataset_type: str = "pet") -> Dict[str, Dict]:
    """
    Preprocess datasets based on type.
    
    Args:
        base_raw_dir: Base directory containing raw datasets
        base_processed_dir: Base directory for processed datasets
        apply_augmentation: Whether to apply data augmentation
        dataset_type: Type of dataset ("covid", "pet")
        
    Returns:
        Dictionary with statistics for each dataset
    """
    results = {}
    
    
    # Pet (default)
    logger.info("=" * 50)
    logger.info("Processing Pet Dataset")
    logger.info("=" * 50)
    pet_raw = os.path.join(base_raw_dir, "pet")
    pet_processed = os.path.join(base_processed_dir, "pet")
    preprocessor = PetPreprocessor(pet_raw, pet_processed)
    preprocessor.apply_augmentation = apply_augmentation
    results['pet'] = preprocessor.process_dataset(organize_by_class=True)
    
    return results


def print_statistics(results: Dict[str, Dict]) -> None:
    """Print preprocessing statistics"""
    logger.info("\n" + "=" * 50)
    logger.info("PREPROCESSING SUMMARY")
    logger.info("=" * 50)
    
    for dataset_name, stats in results.items():
        logger.info(f"\n{dataset_name}:")
        logger.info(f"  Total images: {stats['total']}")
        logger.info(f"  Processed: {stats['processed']}")
        logger.info(f"  Failed: {stats['failed']}")
        logger.info(f"  Skipped: {stats['skipped']}")
        if stats['total'] > 0:
            success_rate = (stats['processed'] / stats['total']) * 100
            logger.info(f"  Success rate: {success_rate:.2f}%")


if __name__ == "__main__":
    # Configuration
    BASE_RAW_DIR = os.path.expanduser("~/data/medical_imaging/raw")
    BASE_PROCESSED_DIR = os.path.expanduser("~/data/medical_imaging/processed")
    
    # Enable/disable augmentation
    APPLY_AUGMENTATION = True  # Set to False to disable augmentation
    
    # You can customize these paths based on your setup
    # For example, if data is in current directory:
    # BASE_RAW_DIR = "./data/raw"
    # BASE_PROCESSED_DIR = "./data/processed"
    
    logger.info(f"Raw data directory: {BASE_RAW_DIR}")
    logger.info(f"Processed data directory: {BASE_PROCESSED_DIR}")
    logger.info(f"Augmentation enabled: {APPLY_AUGMENTATION}")
    
    # Preprocess all datasets
    results = preprocess_all_datasets(BASE_RAW_DIR, BASE_PROCESSED_DIR, apply_augmentation=APPLY_AUGMENTATION)
    
    # Print statistics
    print_statistics(results)
    
    logger.info("\nPreprocessing complete!")
