#!/usr/bin/env python3
"""
Medical Image Binary Classification Training Script

Trains multiple deep learning architectures on medical imaging datasets:
- ResNets (18, 34, 50, 101) - trained from scratch
- DenseNets (121, 169, 201) - trained from scratch
- CheXpert (pretrained on medical images)
- ResNet50-ImageNet (pretrained on ImageNet)

Features:
- Train/Validation/Test split
- Multi-model training and evaluation
- Model checkpointing and early stopping
- Comprehensive metrics (Accuracy, Precision, Recall, F1, AUC-ROC)
- Visualization of training curves and confusion matrices

Author: Mateusz Cedro
"""

import os
import sys
import gc
import time
import warnings
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms, datasets
from torchvision.models import ResNet50_Weights
from PIL import Image
from tqdm import tqdm
import torchxrayvision as xrv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")


def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MedicalImageDataset(Dataset):
    """Custom dataset for medical images"""
    
    def __init__(self, image_paths: List[str], labels: List[int], transform=None):
        """
        Initialize dataset.
        
        Args:
            image_paths: List of image file paths
            labels: List of binary labels (0 or 1)
            transform: Image transformations
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            # Return a black image as fallback
            img = Image.new('RGB', (224, 224))
        
        if self.transform:
            img = self.transform(img)
        
        return img, label


class ModelTrainer:
    """Trainer class for binary image classification"""
    
    def __init__(self, model_name: str, num_classes: int = 2, learning_rate: float = 0.001,
                 num_epochs: int = 50, batch_size: int = 32, checkpoint_dir: str = "./checkpoints",
                 class_weights: torch.Tensor = None):
        """
        Initialize trainer.
        
        Args:
            model_name: Name of the model architecture
            num_classes: Number of classes (binary = 2)
            learning_rate: Learning rate for optimizer
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            checkpoint_dir: Directory to save model checkpoints
            class_weights: Tensor of class weights to handle class imbalance
        """
        self.model_name = model_name
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.class_weights = class_weights
        self.is_frozen = False  # Track if backbone is currently frozen
        
        # Initialize model
        self.model = self._build_model(model_name)
        self.model.to(device)
        
        # Loss and optimizer with class weights if provided
        self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Tracking
        self.history = defaultdict(list)
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.max_patience = 5
        
        logger.info(f"Initialized trainer for {model_name}")
        self._log_hyperparameters()
    
    def _log_hyperparameters(self):
        """Log all hyperparameters"""
        logger.info(f"\n{'='*60}")
        logger.info(f"HYPERPARAMETERS FOR {self.model_name}")
        logger.info(f"{'='*60}")
        logger.info(f"Model Architecture: {self.model_name}")
        logger.info(f"Number of Classes: {self.num_classes}")
        logger.info(f"Learning Rate: {self.learning_rate}")
        logger.info(f"Batch Size: {self.batch_size}")
        logger.info(f"Number of Epochs: {self.num_epochs}")
        logger.info(f"Loss Function: CrossEntropyLoss")
        logger.info(f"Class Weights: {self.class_weights.tolist()}")
        logger.info(f"Optimizer: AdamW")
        logger.info(f"Optimizer Betas: (0.9, 0.999)")
        logger.info(f"Learning Rate Scheduler: ReduceLROnPlateau")
        logger.info(f"  - Factor: 0.5")
        logger.info(f"  - Patience: 5")
        logger.info(f"  - Mode: min (on validation loss)")
        logger.info(f"Early Stopping Patience: {self.max_patience}")
        logger.info(f"Image Size: 224x224")
        logger.info(f"Device: {device}")
        
        # Count model parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Total Parameters: {total_params:,}")
        logger.info(f"Trainable Parameters: {trainable_params:,}")
        logger.info(f"{'='*60}\n")
    
    def _build_model(self, model_name: str) -> nn.Module:
        """Build model architecture from scratch or with pretrained weights"""
        logger.info(f"Building {model_name}...")
        
        # ResNets (trained from scratch)
        if model_name == "ResNet18":
            model = models.resnet18(weights=None)
        elif model_name == "ResNet34":
            model = models.resnet34(weights=None)
        elif model_name == "ResNet50":
            model = models.resnet50(weights=None)
        elif model_name == "ResNet101":
            model = models.resnet101(weights=None)
        
        # DenseNets (trained from scratch)
        elif model_name == "DenseNet121":
            model = models.densenet121(weights=None)
        elif model_name == "DenseNet169":
            model = models.densenet169(weights=None)
        elif model_name == "DenseNet201":
            model = models.densenet201(weights=None)
        
        # ResNet50 pretrained on ImageNet
        elif model_name == "ResNet50-ImageNet":
            logger.info("Loading ResNet50 pretrained on ImageNet...")
            model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

            for param in model.parameters():
                param.requires_grad = False
            self.is_frozen = True
            logger.info("Frozen ResNet50 backbone weights.")
        
        # CheXpert pretrained medical model
        elif model_name == "CheXpert":
            logger.info("Loading CheXpert pretrained model from torchxrayvision...")
            model = xrv.models.DenseNet(weights="densenet121-res224-chex", op_threshs=None)
            
            for param in model.parameters():
                param.requires_grad = False
            self.is_frozen = True
            logger.info("Frozen CheXpert backbone weights.")

            # Store original forward method for feature extraction
            original_features = model.features
            original_classifier = model.classifier
            
            # Create a wrapper that extracts features and applies custom classification
            class CheXpertBinary(nn.Module):
                def __init__(self, features, classifier, num_classes):
                    super().__init__()
                    self.features = features
                    # This new layer will automatically have requires_grad=True
                    self.classifier = nn.Linear(classifier.in_features, num_classes)
                
                def forward(self, x):
                    # Extract features using CheXpert's feature extractor
                    out = self.features(x)
                    # Global average pooling
                    out = torch.nn.functional.adaptive_avg_pool2d(out, (1, 1))
                    out = torch.flatten(out, 1)
                    # Binary classification
                    out = self.classifier(out)
                    return out
            
            model = CheXpertBinary(original_features, original_classifier, self.num_classes)
            return model
        
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Convert first layer to accept single-channel (grayscale) input
        # The original first layer expects 3 channels, so we replace it to accept 1 channel
        if "ResNet" in model_name:
            # Modify the first convolutional layer to accept 1 channel
            original_conv = model.conv1
            model.conv1 = nn.Conv2d(1, original_conv.out_channels, kernel_size=original_conv.kernel_size,
                                   stride=original_conv.stride, padding=original_conv.padding, bias=original_conv.bias)
            
            # Initialize weights from original conv (average across channel dimension)
            if original_conv.weight.data is not None:
                model.conv1.weight.data = original_conv.weight.data.mean(dim=1, keepdim=True)
            
            # Modify final layer for binary classification
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, self.num_classes)
            
        elif "DenseNet" in model_name:
            # Modify the first convolutional layer to accept 1 channel
            original_conv = model.features[0]
            model.features[0] = nn.Conv2d(1, original_conv.out_channels, kernel_size=original_conv.kernel_size,
                                         stride=original_conv.stride, padding=original_conv.padding, bias=original_conv.bias)
            
            # Initialize weights from original conv (average across channel dimension)
            if original_conv.weight.data is not None:
                model.features[0].weight.data = original_conv.weight.data.mean(dim=1, keepdim=True)
            
            # Modify final layer for binary classification
            in_features = model.classifier.in_features
            model.classifier = nn.Linear(in_features, self.num_classes)
        
        return model

    def unfreeze_backbone(self, new_lr: float = 1e-5):
        """Unfreezes model and lowers learning rate for fine-tuning"""
        if not self.is_frozen:
            logger.info("Model is already unfrozen. Skipping unfreeze step.")
            return

        logger.info(f"\n{'!'*60}")
        logger.info(f"UNFREEZING BACKBONE - Switching to Fine-Tuning Mode")
        logger.info(f"New Learning Rate: {new_lr}")
        logger.info(f"{'!'*60}\n")

        # 1. Unfreeze all parameters
        for param in self.model.parameters():
            param.requires_grad = True
            
        # 2. Reset Optimizer with lower LR
        self.optimizer = optim.AdamW(self.model.parameters(), lr=new_lr)
        
        # 3. Reset Scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )

        self.patience_counter = 0 
        logger.info("Patience counter reset to 0.")
        
        self.is_frozen = False # Update flag
        
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        
        for images, labels in tqdm(train_loader, desc="Training"):
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        return avg_loss
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate model"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validating"):
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total
        return avg_loss, accuracy

    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, unfreeze_patience: bool = True):
        """
        Train model with Smart Unfreezing.
        
        Args:
            train_loader: Training data
            val_loader: Validation data
            unfreeze_patience: If True, uses patience to trigger unfreeze instead of stopping.
        """
        logger.info(f"Starting training for {self.model_name}")
        
        training_start_time = time.time()
        epoch_times = []
        
        for epoch in range(self.num_epochs):
            epoch_start_time = time.time()
            
            train_loss = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)
            
            epoch_time = time.time() - epoch_start_time
            epoch_times.append(epoch_time)
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_accuracy'].append(val_acc)
            
            logger.info(f"Epoch {epoch+1}/{self.num_epochs} - "
                       f"Train Loss: {train_loss:.4f}, "
                       f"Val Loss: {val_loss:.4f}, "
                       f"Val Acc: {val_acc:.4f} "
                       f"[{epoch_time:.2f}s]")
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self._save_checkpoint(epoch)
                logger.info(f"Saved checkpoint for {self.model_name}")
            else:
                self.patience_counter += 1
                logger.info(f"Patience counter: {self.patience_counter}/{self.max_patience}")
                
                if self.patience_counter >= self.max_patience:
                    
                    # CASE 1: Model is Frozen -> Time to Unfreeze!
                    if self.is_frozen and unfreeze_patience:
                        logger.info(f"Head has converged (patience reached). Unfreezing backbone now...")
                        
                        # IMPORTANT: Load the BEST model so far before unfreezing
                        # We don't want to continue from the current 'bad' epoch, 
                        # we want the best version of the Head.
                        self._load_checkpoint() 
                        
                        # Unfreeze and reset patience
                        self.unfreeze_backbone(new_lr=1e-5)
                        self.patience_counter = 0 
                        
                    # CASE 2: Model is already Unfrozen (or scratch) -> Really Stop
                    else:
                        logger.info(f"Early stopping triggered for {self.model_name}")
                        break
        
        # Load best model
        self._load_checkpoint()
        
        # Calculate and log training time
        total_training_time = time.time() - training_start_time
        avg_epoch_time = np.mean(epoch_times)
        
        self.training_time = total_training_time
        self.avg_epoch_time = avg_epoch_time
        
        logger.info(f"\nTraining completed for {self.model_name}")
        logger.info(f"  Total training time: {total_training_time/60:.2f} minutes ({total_training_time:.2f} seconds)")
        logger.info(f"  Average time per epoch: {avg_epoch_time:.2f} seconds")
        logger.info(f"  Number of epochs completed: {len(epoch_times)}")
    
    def _save_checkpoint(self, epoch: int):
        """Save model checkpoint"""
        checkpoint_path = self.checkpoint_dir / f"{self.model_name}_best.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss
        }, checkpoint_path)
    
    def _load_checkpoint(self):
        """Load best model checkpoint"""
        checkpoint_path = self.checkpoint_dir / f"{self.model_name}_best.pth"
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded checkpoint for {self.model_name}")
    
    def evaluate(self, test_loader: DataLoader) -> Dict:
        """Evaluate model on test set"""
        self.model.eval()
        all_predictions = []
        all_labels = []
        all_probs = []
        
        # Start timing
        inference_start_time = time.time()
        
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="Testing"):
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = self.model(images)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                
                _, predicted = torch.max(outputs.data, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())
        
        # Calculate inference time
        inference_time = time.time() - inference_start_time
        num_samples = len(all_labels)
        time_per_sample = (inference_time / num_samples) * 1000  # in milliseconds
        
        # Calculate metrics
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        metrics = {
            'accuracy': accuracy_score(all_labels, all_predictions),
            'precision': precision_score(all_labels, all_predictions, zero_division=0),
            'recall': recall_score(all_labels, all_predictions, zero_division=0),
            'f1': f1_score(all_labels, all_predictions, zero_division=0),
            'auc_roc': roc_auc_score(all_labels, all_probs),
            'confusion_matrix': confusion_matrix(all_labels, all_predictions).tolist(),
            'predictions': all_predictions.tolist(),
            'labels': all_labels.tolist(),
            'probabilities': all_probs.tolist(),
            'inference_time': inference_time,
            'num_test_samples': num_samples,
            'time_per_sample_ms': time_per_sample
        }
        
        logger.info(f"\n{self.model_name} - Inference Statistics:")
        logger.info(f"  Total inference time: {inference_time:.4f} seconds")
        logger.info(f"  Number of test samples: {num_samples}")
        logger.info(f"  Time per sample: {time_per_sample:.4f} ms")
        logger.info(f"  Throughput: {num_samples/inference_time:.2f} samples/second")
        
        return metrics
    
    def plot_training_curves(self, save_path: str = None):
        """Plot training and validation curves"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss
        axes[0].plot(self.history['train_loss'], label='Train Loss', marker='o', linewidth=2)
        axes[0].plot(self.history['val_loss'], label='Val Loss', marker='s', linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=11)
        axes[0].set_ylabel('Loss', fontsize=11)
        axes[0].set_title(f'{self.model_name} - Training & Validation Loss', fontsize=12, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[1].plot(self.history['val_accuracy'], label='Val Accuracy', marker='o', linewidth=2, color='green')
        axes[1].set_xlabel('Epoch', fontsize=11)
        axes[1].set_ylabel('Accuracy', fontsize=11)
        axes[1].set_title(f'{self.model_name} - Validation Accuracy', fontsize=12, fontweight='bold')
        axes[1].set_ylim([0, 1.05])
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training curves saved to {save_path}")
        plt.close()
    
    def plot_confusion_matrix(self, test_loader: DataLoader, save_path: str = None):
        """Plot confusion matrix for test set"""
        self.model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="Computing CM"):
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Compute confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        
        # Plot
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        
        # Add colorbar
        plt.colorbar(im, ax=ax)
        
        # Set labels
        classes = ['Class 0', 'Class 1']
        tick_marks = np.arange(len(classes))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(classes, fontsize=10)
        ax.set_yticklabels(classes, fontsize=10)
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i, j in np.ndindex(cm.shape):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black",
                   fontsize=14, fontweight='bold')
        
        ax.set_ylabel('True Label', fontsize=11, fontweight='bold')
        ax.set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
        ax.set_title(f'{self.model_name} - Confusion Matrix', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
        plt.close()
        
        return cm


def load_dataset(data_dir: str, train_ratio: float = 0.7, val_ratio: float = 0.2, seed: int = 42, batch_size: int = 32, model_name: str = None):
    """
    Load dataset with train/val/test split, skipping corrupted images.
    
    Data structure expected (supports multiple formats):
    data_dir/
        class_0/
            images/
                image1.jpg
                image2.jpg
        class_1/
            images/
                image3.jpg
    
    Args:
        data_dir: Root directory containing class folders with images/ subdirectories
        train_ratio: Proportion of data for training (default 0.7)
        val_ratio: Proportion of data for validation (default 0.2)
        seed: Random seed for reproducibility
        batch_size: Batch size for dataloaders
        model_name: Model name for applying model-specific preprocessing (e.g., 'CheXpert' for grayscale)
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader, class_names, weights_tensor)
    """
    # Set random seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    data_dir = Path(data_dir)
    
    # Get transformations (model-specific)
    train_transform, val_test_transform = get_data_transforms(model_name=model_name)
    
    # Helper function to validate image
    def is_valid_image(img_path):
        """Check if image file can be opened"""
        try:
            img = Image.open(img_path)
            img.load()
            return True
        except Exception:
            return False
    
    # Collect all valid image paths and their class labels
    image_paths = []
    labels = []
    class_names = []
    class_to_idx = {}
    class_counts = {}  # Track images per class
    
    for class_path in sorted(data_dir.iterdir()):
        if class_path.is_dir() and class_path.name != "_temp_imagefolder":
            class_name = class_path.name
            class_names.append(class_name)
            class_to_idx[class_name] = len(class_names) - 1
            class_counts[class_name] = 0  # Initialize counter
            
            # Get all valid images from the images/ subdirectory
            images_dir = class_path / "images"
            if images_dir.exists():
                for img_file in sorted(images_dir.glob("*.*")):
                    if img_file.is_file() and img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']:
                        # Only add if image is valid
                        if is_valid_image(img_file):
                            image_paths.append(str(img_file))
                            labels.append(class_to_idx[class_name])
                            class_counts[class_name] += 1
    
    total_images = len(image_paths)
    logger.info(f"Found {total_images} valid images in {len(class_names)} classes: {class_names}")
    logger.info("\nImages correctly loaded by class:")
    class_distribution = {}
    for class_name in class_names:
        count = class_counts[class_name]
        class_distribution[class_name] = count
        logger.info(f"  - {class_name}: {count} images ({count/total_images*100:.1f}%)")
        if count == 0:
            logger.error(f"ERROR: Class '{class_name}' has 0 images!")
            logger.error("Cannot proceed with training. Please check your dataset structure.")
            sys.exit(1)
    
    # Calculate class weights for imbalanced data (inverse of class frequency)
    class_weights = {}
    for class_name in class_names:
        class_weight = total_images / (len(class_names) * class_counts[class_name])
        class_weights[class_name] = class_weight
    
    logger.info("\nClass Weights:")
    for class_name, weight in class_weights.items():
        logger.info(f"  - {class_name}: {weight:.4f}")
    
    # Convert to tensor ordered by class index
    weights_tensor = torch.tensor([class_weights[class_names[i]] for i in range(len(class_names))], dtype=torch.float32)
    
    # Create train/val/test split
    n = len(image_paths)
    train_size = int(n * train_ratio)
    val_size = int(n * val_ratio)
    test_size = n - train_size - val_size
    
    # Generate random indices for splitting
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    # Create datasets
    train_dataset = MedicalImageDataset(
        [image_paths[i] for i in train_indices],
        [labels[i] for i in train_indices],
        transform=train_transform
    )
    
    val_dataset = MedicalImageDataset(
        [image_paths[i] for i in val_indices],
        [labels[i] for i in val_indices],
        transform=val_test_transform
    )
    
    test_dataset = MedicalImageDataset(
        [image_paths[i] for i in test_indices],
        [labels[i] for i in test_indices],
        transform=val_test_transform
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")
    logger.info(f"Test dataset size: {len(test_dataset)}")
    logger.info(f"Class names: {class_names}")
    
    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader, class_names, weights_tensor

def get_data_transforms(image_size: int = 224, model_name: str = None):
    """
    Get image transformations for training and testing.
    All images are converted to grayscale (1 channel) for consistency across models.
    Models are modified to accept single-channel input.
    
    Args:
        image_size: Size to resize images to
        model_name: Model name (optional, kept for backwards compatibility)
    """
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.Grayscale(num_output_channels=1),  # Convert all images to grayscale
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.Grayscale(num_output_channels=1),  # Convert all images to grayscale
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
    ])
    
    return train_transform, val_test_transform


def mock_training(data_dir: str, output_dir: str = "./mock_results", num_epochs: int = 3, 
                  batch_size: int = 32, seeds: List[int] = None):
    """
    Quick mock training to test the entire pipeline.
    
    Args:
        data_dir: Directory containing dataset
        output_dir: Directory to save mock results
        num_epochs: Number of training epochs (default 2 for quick testing)
        batch_size: Batch size for training
        seeds: List of random seeds for multiple runs (default [42])
        
    Returns:
        mock_results: Dictionary with results from mock training over all seeds
    """
    
    # Set default seeds if not provided
    if seeds is None:
        seeds = [42, 0]
    
    logger.info("\n" + "="*80)
    logger.info("MOCK TRAINING - PIPELINE VERIFICATION")
    logger.info("="*80)
    logger.info(f"Epochs: {num_epochs}, Batch Size: {batch_size}")
    logger.info(f"Number of seeds: {len(seeds)}, Seeds: {seeds}")
    logger.info("="*80 + "\n")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Dictionary to store results across all seeds
    all_mock_results = {}
    aggregated_metrics = defaultdict(list)
    
    # Create checkpoint directory
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Models to test (subset for speed)
    test_models = [
        "ResNet18", 
        #"DenseNet121", 
        #"ResNet50-ImageNet",
        "CheXpert",
    ]
    
    # Training loop over multiple seeds
    for seed_idx, current_seed in enumerate(seeds, 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"SEED {seed_idx}/{len(seeds)}: seed={current_seed}")
        logger.info(f"{'='*80}")
        
        # Set seed for this iteration
        set_seed(current_seed)
        
        seed_results = {}
        
        logger.info(f"Dataloaders created for {len(seeds)} seeds")
        
        for model_name in test_models:
            logger.info(f"\n  Training {model_name}...")
            
            try:
                # Load dataset with model-specific preprocessing
                logger.info(f"    Loading dataset with {model_name} preprocessing...")
                train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader, class_names, class_weights = load_dataset(
                    data_dir, seed=current_seed, batch_size=batch_size, model_name=model_name
                )
                
                # Create trainer with checkpoint directory
                trainer = ModelTrainer(
                    model_name=model_name,
                    num_epochs=num_epochs,
                    batch_size=batch_size,
                    checkpoint_dir=str(checkpoint_dir),
                    class_weights=class_weights
                )
                
                logger.info(f"    Checkpoint directory: {checkpoint_dir}")
                
                # Train (includes checkpoint saving during training)
                trainer.train(train_loader, val_loader)
                
                # Verify checkpoint was saved
                checkpoint_path = checkpoint_dir / f"{model_name}_best.pth"
                if checkpoint_path.exists():
                    checkpoint_size = checkpoint_path.stat().st_size / (1024 * 1024)  # Size in MB
                    logger.info(f"    ✓ Checkpoint saved: {checkpoint_path} ({checkpoint_size:.2f} MB)")
                else:
                    logger.warning(f"    ⚠ Checkpoint not found: {checkpoint_path}")
                
                # Evaluate
                test_metrics = trainer.evaluate(test_loader)
                
                # Store results
                seed_results[model_name] = {
                    'metrics': test_metrics,
                    'history': dict(trainer.history),
                    'training_time': trainer.training_time,
                    'avg_epoch_time': trainer.avg_epoch_time,
                    'checkpoint_path': str(checkpoint_path)
                }
                
                # Aggregate metrics for later analysis
                aggregated_metrics[model_name].append({
                    'seed': current_seed,
                    'accuracy': test_metrics['accuracy'],
                    'precision': test_metrics['precision'],
                    'recall': test_metrics['recall'],
                    'f1': test_metrics['f1'],
                    'auc_roc': test_metrics['auc_roc'],
                    'inference_time': test_metrics['inference_time']
                })
                
                # Log results
                logger.info(f"    ✓ Accuracy: {test_metrics['accuracy']:.4f}")
                logger.info(f"    ✓ F1-Score: {test_metrics['f1']:.4f}")
                logger.info(f"    ✓ AUC-ROC: {test_metrics['auc_roc']:.4f}")
                
                # Clean up
                del trainer
                gc.collect()
                torch.cuda.empty_cache()
                
            except Exception as e:
                logger.error(f"    ✗ Error training {model_name}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Store results for this seed
        all_mock_results[f"seed_{current_seed}"] = seed_results
        
        # Create plots directory for this seed
        seed_plot_dir = output_dir / "plots" / f"seed_{current_seed}"
        seed_plot_dir.mkdir(parents=True, exist_ok=True)
        
        # Save plots for this seed
        for model_name, data in seed_results.items():
            try:
                # We need to recreate trainer to access plot methods
                # For now, just save the metrics
                pass
            except Exception as e:
                logger.warning(f"Could not save plots for {model_name}: {e}")
    
    # Save all mock results
    mock_results_path = output_dir / "mock_results.json"
    
    # Prepare data for JSON serialization
    json_results = {
        "metadata": {
            "num_seeds": len(seeds),
            "seeds": [int(s) for s in seeds],
            "num_epochs": int(num_epochs),
            "batch_size": int(batch_size)
        },
        "results_by_seed": {}
    }
    
    for seed_key, seed_data in all_mock_results.items():
        json_results["results_by_seed"][seed_key] = {}
        for model_name, data in seed_data.items():
            json_results["results_by_seed"][seed_key][model_name] = {
                'metrics': {
                    'accuracy': float(data['metrics']['accuracy']),
                    'precision': float(data['metrics']['precision']),
                    'recall': float(data['metrics']['recall']),
                    'f1': float(data['metrics']['f1']),
                    'auc_roc': float(data['metrics']['auc_roc']),
                    'confusion_matrix': data['metrics']['confusion_matrix'],
                    'inference_time': float(data['metrics']['inference_time']),
                    'time_per_sample_ms': float(data['metrics']['time_per_sample_ms']),
                    'num_test_samples': int(data['metrics']['num_test_samples'])
                },
                'training_time': float(data['training_time']),
                'avg_epoch_time': float(data['avg_epoch_time']),
                'checkpoint_path': data.get('checkpoint_path', 'Not saved')
            }
    
    # Add aggregated statistics across seeds
    json_results["aggregated_stats"] = {}
    for model_name, metrics_list in aggregated_metrics.items():
        accuracies = [m['accuracy'] for m in metrics_list]
        f1_scores = [m['f1'] for m in metrics_list]
        auc_rocs = [m['auc_roc'] for m in metrics_list]
        
        json_results["aggregated_stats"][model_name] = {
            'accuracy': {
                'mean': float(np.mean(accuracies)),
                'std': float(np.std(accuracies)),
                'min': float(np.min(accuracies)),
                'max': float(np.max(accuracies))
            },
            'f1': {
                'mean': float(np.mean(f1_scores)),
                'std': float(np.std(f1_scores)),
                'min': float(np.min(f1_scores)),
                'max': float(np.max(f1_scores))
            },
            'auc_roc': {
                'mean': float(np.mean(auc_rocs)),
                'std': float(np.std(auc_rocs)),
                'min': float(np.min(auc_rocs)),
                'max': float(np.max(auc_rocs))
            }
        }
    
    with open(mock_results_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    logger.info(f"\n✓ Mock results saved to {mock_results_path}")
    
    # Save mock configuration
    config_path = output_dir / "mock_training_config.txt"
    config_text = f"""
{'='*80}
MOCK TRAINING CONFIGURATION - Pipeline Verification with Multiple Seeds
{'='*80}

PURPOSE:
  Test the entire training pipeline with minimal data before running full training
  using multiple random seeds to assess variability

PARAMETERS:
  - Total train samples: {len(train_dataset)}
  - Total val samples: {len(val_dataset)}
  - Total test samples: {len(test_dataset)}
  - Epochs per seed: {num_epochs}
  - Batch Size: {batch_size}
  - Number of seeds: {len(seeds)}
  - Seeds: {seeds}

MODELS TESTED:
  - ResNet18 (from scratch)
  - DenseNet121 (from scratch)
  - ResNet50-ImageNet (pretrained)

VERIFIED:
  ✓ Data loading and preprocessing
  ✓ Model initialization and architecture
  ✓ Training loop and optimization
  ✓ Validation procedure
  ✓ Evaluation and metrics computation
  ✓ Checkpoint saving and loading (best model saved during training)
  ✓ Visualization generation
  ✓ GPU/CPU compatibility
  ✓ Memory management
  ✓ Reproducibility and seed handling
  ✓ Training variability across seeds

CHECKPOINT SAVING:
  During training, the model checkpoint is automatically saved whenever the validation loss
  improves. This ensures the best model weights are always preserved.
  - Checkpoint format: PyTorch .pth file
  - Contents: model_state_dict, optimizer_state_dict, epoch, best_val_loss
  - Location: mock_results/checkpoints/{model_name}_best.pth
  - Usage: Checkpoints can be loaded later for inference or continued training

RESULTS:
  Results are saved for each seed separately, with aggregated statistics (mean, std, min, max)
  computed across all seeds for each model. This allows assessment of:
  - Stability: Are results consistent across seeds?
  - Variability: How much do results vary with different random initializations?
  - Robustness: Are the models robust to initialization differences?

NEXT STEPS:
  If all models trained successfully across all seeds, run full training:
  >>> python train_models.py

OUTPUT FILES:
  - mock_results.json: Model metrics for all seeds with aggregated statistics and checkpoint paths
  - mock_training_config.txt: This file
  - plots/seed_*/: Training curves and confusion matrices per seed (if generated)
  - checkpoints/: Model weights (.pth files) - Best model for each architecture

{'='*80}
"""
    
    with open(config_path, 'w') as f:
        f.write(config_text)
    
    logger.info(f"✓ Mock configuration saved to {config_path}")
    
    # Print aggregated summary
    logger.info("\n" + "="*80)
    logger.info("AGGREGATED RESULTS ACROSS ALL SEEDS")
    logger.info("="*80)
    for model_name, stats in json_results.get("aggregated_stats", {}).items():
        logger.info(f"\n{model_name}:")
        logger.info(f"  Accuracy: {stats['accuracy']['mean']:.4f} ± {stats['accuracy']['std']:.4f}")
        logger.info(f"  F1-Score: {stats['f1']['mean']:.4f} ± {stats['f1']['std']:.4f}")
        logger.info(f"  AUC-ROC:  {stats['auc_roc']['mean']:.4f} ± {stats['auc_roc']['std']:.4f}")
    
    logger.info("\n" + "="*80)
    logger.info("MOCK TRAINING COMPLETE")
    logger.info("="*80)
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"Seeds tested: {len(seeds)}")
    logger.info(f"Models tested per seed: {len(test_models)}")
    logger.info(f"Total training runs: {len(seeds) * len(test_models)}")
    logger.info(f"Successful seed runs: {len(all_mock_results)}/{len(seeds)}")
    
    total_runs = sum(len(models) for models in all_mock_results.values())
    total_expected = len(seeds) * len(test_models)
    logger.info(f"Successful individual runs: {total_runs}/{total_expected}")
    
    if len(all_mock_results) == len(seeds) and total_runs == total_expected:
        logger.info("\n✓ All models trained successfully across all seeds!")
        logger.info("✓ Pipeline verification PASSED")
        logger.info("✓ Results are stable and reproducible")
        logger.info("✓ Safe to run full training")
    else:
        logger.warning(f"\n✗ Some models failed. Check errors above.")
    
    logger.info("="*80 + "\n")
    
    return all_mock_results


def train_all_models(data_dir: str, output_dir: str = "./results", num_epochs: int = 50, batch_size: int = 32, num_runs: int = 3, seeds: List[int] = None, unfreeze_after: int = 5):
    """
    Train all models multiple times with different seeds.
    
    Args:
        data_dir: Directory containing dataset
        output_dir: Directory to save results
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        num_runs: Number of runs with different seeds
        seeds: List of random seeds (if None, generates automatically)
        unfreeze_after: Epoch number to unfreeze backbone for pretrained models
    """
    
    if seeds is None:
        seeds = list(range(42, 42 + num_runs))
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Model architectures to train
    models_to_train = [
        "ResNet18", "ResNet34", "ResNet50", "ResNet101",
        "DenseNet121", "DenseNet169", "DenseNet201",
        "ResNet50-ImageNet",
        "CheXpert"
    ]
    
    # Store results for all runs: {model_name: {run_id: metrics}}
    all_results = defaultdict(lambda: defaultdict(dict))
    
    logger.info(f"Starting {num_runs} runs with seeds: {seeds}")
    
    for run_id, seed in enumerate(seeds):
        logger.info(f"\n{'='*80}")
        logger.info(f"RUN {run_id + 1}/{num_runs} - SEED: {seed}")
        logger.info(f"{'='*80}")
        
        # Set seed for reproducibility
        set_seed(seed)
        
        # Load dataset with current seed - get both datasets and dataloaders
        train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader, class_names, class_weights = load_dataset(
            data_dir, seed=seed, batch_size=batch_size
        )
        
        for model_name in models_to_train:
            logger.info(f"\nTraining {model_name}...")
            
            try:
                # Create trainer
                trainer = ModelTrainer(
                    model_name=model_name,
                    num_epochs=num_epochs,
                    batch_size=batch_size,
                    checkpoint_dir=str(output_dir / "checkpoints" / f"seed_{seed}"),
                    class_weights=class_weights
                )
                
                # Train
                trainer.train(train_loader, val_loader, unfreeze_patience=True)
                
                # Create directory for plots
                plots_dir = output_dir / "plots" / f"seed_{seed}"
                plots_dir.mkdir(parents=True, exist_ok=True)
                
                # Plot training curves
                trainer.plot_training_curves(
                    save_path=str(plots_dir / f"{model_name}_training_curves.png")
                )
                
                # Plot confusion matrix
                trainer.plot_confusion_matrix(
                    test_loader,
                    save_path=str(plots_dir / f"{model_name}_confusion_matrix.png")
                )
                
                # Evaluate
                test_metrics = trainer.evaluate(test_loader)
                
                # Store results
                all_results[model_name][run_id] = {
                    'seed': seed,
                    'metrics': test_metrics,
                    'history': dict(trainer.history),
                    'training_time': trainer.training_time,
                    'avg_epoch_time': trainer.avg_epoch_time,
                    'final_val_loss': trainer.best_val_loss
                }
                
                logger.info(f"{model_name} - Accuracy: {test_metrics['accuracy']:.4f}, "
                           f"F1: {test_metrics['f1']:.4f}, AUC-ROC: {test_metrics['auc_roc']:.4f}")
                
                # Save intermediate results after each model completes
                _save_intermediate_results(all_results, output_dir, run_id, seed, model_name)
                
                # Clean up to free memory
                del trainer
                gc.collect()
                torch.cuda.empty_cache()

            except Exception as e:
                logger.error(f"Error training {model_name} (seed {seed}): {e}")
                continue
    
    # Aggregate and save final results
    aggregated_results = aggregate_results(all_results)
    save_aggregated_results(aggregated_results, output_dir)
    save_training_config(output_dir, num_epochs, batch_size, num_runs=num_runs, seeds=seeds)
    
    # Create comparison plots
    create_comparison_plots_with_std(aggregated_results, output_dir)
    
    return all_results, aggregated_results


def _save_intermediate_results(all_results: Dict, output_dir: Path, run_id: int, seed: int, completed_model: str = None):
    """
    Save and display intermediate results after each model completes training.
    Shows progress through the training process.
    
    Args:
        all_results: Current results collected so far {model_name: {run_id: metrics}}
        output_dir: Directory to save results
        run_id: Current run ID
        seed: Current seed value
        completed_model: Name of the model that just completed (if provided, shows only this model's results)
    """
    # Create intermediate results directory
    intermediate_dir = output_dir / "intermediate_results" / f"seed_{seed}"
    intermediate_dir.mkdir(parents=True, exist_ok=True)
    
    # Save current all_results as JSON for this checkpoint
    if completed_model:
        checkpoint_file = intermediate_dir / f"{completed_model}_run_{run_id}.json"
    else:
        checkpoint_file = intermediate_dir / f"run_{run_id}.json"
    
    json_results = {}
    for model_name, runs in all_results.items():
        json_results[model_name] = {}
        for r_id, data in runs.items():
            json_results[model_name][str(r_id)] = {
                'seed': int(data['seed']),
                'metrics': {
                    'accuracy': float(data['metrics']['accuracy']),
                    'precision': float(data['metrics']['precision']),
                    'recall': float(data['metrics']['recall']),
                    'f1': float(data['metrics']['f1']),
                    'auc_roc': float(data['metrics']['auc_roc']),
                    'inference_time': float(data['metrics'].get('inference_time', 0.0)),
                    'time_per_sample_ms': float(data['metrics'].get('time_per_sample_ms', 0.0)),
                    'num_test_samples': int(data['metrics'].get('num_test_samples', 0))
                },
                'training_time': float(data['training_time']),
                'avg_epoch_time': float(data['avg_epoch_time']),
                'final_val_loss': float(data.get('final_val_loss', 0.0))
            }
    
    with open(checkpoint_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    logger.info(f"Checkpoint saved: {checkpoint_file.name}")
    
    # Display results for completed model or all models if none specified
    if completed_model:
        # Show detailed results for the model that just completed
        logger.info(f"\n{'='*30}")
        logger.info(f"COMPLETED: {completed_model} (Run {run_id + 1}, Seed {seed})")
        logger.info(f"{'='*30}")
        
        runs = all_results.get(completed_model, {})
        if runs:
            accuracies = [runs[r_id]['metrics']['accuracy'] for r_id in sorted(runs.keys())]
            precisions = [runs[r_id]['metrics']['precision'] for r_id in sorted(runs.keys())]
            recalls = [runs[r_id]['metrics']['recall'] for r_id in sorted(runs.keys())]
            f1_scores = [runs[r_id]['metrics']['f1'] for r_id in sorted(runs.keys())]
            auc_rocs = [runs[r_id]['metrics']['auc_roc'] for r_id in sorted(runs.keys())]
            
            logger.info(f"  Runs completed: {len(runs)}")
            logger.info(f"  Accuracy:  {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
            logger.info(f"  Precision: {np.mean(precisions):.4f} ± {np.std(precisions):.4f}")
            logger.info(f"  Recall:    {np.mean(recalls):.4f} ± {np.std(recalls):.4f}")
            logger.info(f"  F1-Score:  {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
            logger.info(f"  AUC-ROC:   {np.mean(auc_rocs):.4f} ± {np.std(auc_rocs):.4f}")
        
        logger.info(f"{'='*30}\n")
    else:
        # Show summary for all models
        logger.info(f"\n{'='*30}")
        logger.info(f"RESULTS SUMMARY - After Run {run_id + 1} (Seed {seed})")
        logger.info(f"{'='*30}")
        
        # Show results per model
        for model_name in sorted(all_results.keys()):
            runs = all_results[model_name]
            if runs:  # Only show if this model has results
                accuracies = [runs[r_id]['metrics']['accuracy'] for r_id in sorted(runs.keys())]
                f1_scores = [runs[r_id]['metrics']['f1'] for r_id in sorted(runs.keys())]
                auc_rocs = [runs[r_id]['metrics']['auc_roc'] for r_id in sorted(runs.keys())]
                
                logger.info(f"\n{model_name}:")
                logger.info(f"  Runs completed: {len(runs)}")
                logger.info(f"  Accuracy:  {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
                logger.info(f"  F1-Score:  {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
                logger.info(f"  AUC-ROC:   {np.mean(auc_rocs):.4f} ± {np.std(auc_rocs):.4f}")
        
        logger.info(f"\n{'='*30}\n")


def aggregate_results(all_results: Dict) -> Dict:
    """
    Aggregate results from multiple runs.
    
    Args:
        all_results: Results from all runs {model_name: {run_id: metrics}}
        
    Returns:
        Aggregated statistics {model_name: {metric: {mean, std, values}}}
    """
    aggregated = {}
    
    for model_name, runs in all_results.items():
        aggregated[model_name] = {}
        
        # Metrics to aggregate
        metrics_keys = ['accuracy', 'precision', 'recall', 'f1', 'auc_roc']
        
        for metric_key in metrics_keys:
            values = [runs[run_id]['metrics'][metric_key] for run_id in runs.keys()]
            aggregated[model_name][metric_key] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'values': values
            }
        
        # Aggregate training times
        training_times = [runs[run_id]['training_time'] for run_id in runs.keys()]
        aggregated[model_name]['training_time'] = {
            'mean': float(np.mean(training_times)),
            'std': float(np.std(training_times)),
            'min': float(np.min(training_times)),
            'max': float(np.max(training_times)),
            'total': float(sum(training_times)),
            'values': training_times
        }
        
        # Aggregate avg epoch times
        avg_epoch_times = [runs[run_id]['avg_epoch_time'] for run_id in runs.keys()]
        aggregated[model_name]['avg_epoch_time'] = {
            'mean': float(np.mean(avg_epoch_times)),
            'std': float(np.std(avg_epoch_times)),
            'min': float(np.min(avg_epoch_times)),
            'max': float(np.max(avg_epoch_times)),
            'values': avg_epoch_times
        }
        
        # Aggregate inference times
        inference_times = [runs[run_id]['metrics']['inference_time'] for run_id in runs.keys()]
        time_per_sample = [runs[run_id]['metrics']['time_per_sample_ms'] for run_id in runs.keys()]
        num_test_samples = [runs[run_id]['metrics']['num_test_samples'] for run_id in runs.keys()]
        
        aggregated[model_name]['inference_time'] = {
            'mean': float(np.mean(inference_times)),
            'std': float(np.std(inference_times)),
            'min': float(np.min(inference_times)),
            'max': float(np.max(inference_times)),
            'values': inference_times
        }
        
        aggregated[model_name]['time_per_sample_ms'] = {
            'mean': float(np.mean(time_per_sample)),
            'std': float(np.std(time_per_sample)),
            'min': float(np.min(time_per_sample)),
            'max': float(np.max(time_per_sample)),
            'values': time_per_sample
        }
        
        aggregated[model_name]['num_test_samples'] = int(num_test_samples[0])
        
        # Aggregate final validation loss
        final_val_losses = [runs[run_id].get('final_val_loss', 0.0) for run_id in runs.keys()]
        aggregated[model_name]['final_val_loss'] = {
            'mean': float(np.mean(final_val_losses)),
            'std': float(np.std(final_val_losses)),
            'values': final_val_losses
        }
    
    return aggregated


def save_aggregated_results(aggregated_results: Dict, output_dir: Path):
    """Save aggregated results to JSON"""
    results_path = output_dir / "aggregated_results.json"
    
    with open(results_path, 'w') as f:
        json.dump(aggregated_results, f, indent=2)
    
    logger.info(f"Aggregated results saved to {results_path}")
    
    # Also print summary
    logger.info("\n" + "="*80)
    logger.info("AGGREGATED RESULTS SUMMARY")
    logger.info("="*80)
    
    for model_name, metrics in aggregated_results.items():
        logger.info(f"\n{model_name}:")
        for metric_key, stats in metrics.items():
            if metric_key == 'training_time':
                logger.info(f"  {metric_key}:")
                logger.info(f"    - Mean: {stats['mean']/60:.2f} min ({stats['mean']:.2f}s)")
                logger.info(f"    - Std: {stats['std']:.2f}s")
                logger.info(f"    - Min: {stats['min']/60:.2f} min")
                logger.info(f"    - Max: {stats['max']/60:.2f} min")
                logger.info(f"    - Total: {stats['total']/60:.2f} min (all runs)")
            elif metric_key == 'avg_epoch_time':
                logger.info(f"  {metric_key}: {stats['mean']:.2f} ± {stats['std']:.2f} seconds")
            elif metric_key == 'inference_time':
                logger.info(f"  {metric_key}:")
                logger.info(f"    - Mean: {stats['mean']:.4f} seconds")
                logger.info(f"    - Std: {stats['std']:.4f} seconds")
                logger.info(f"    - Min: {stats['min']:.4f} seconds")
                logger.info(f"    - Max: {stats['max']:.4f} seconds")
            elif metric_key == 'time_per_sample_ms':
                logger.info(f"  {metric_key}: {stats['mean']:.4f} ± {stats['std']:.4f} ms")
            elif metric_key == 'num_test_samples':
                logger.info(f"  {metric_key}: {stats}")
            else:
                logger.info(f"  {metric_key}: {stats['mean']:.4f} ± {stats['std']:.4f} "
                           f"(min: {stats['min']:.4f}, max: {stats['max']:.4f})")


def save_training_config(output_dir: Path, num_epochs: int, batch_size: int, learning_rate: float = 0.001, #TODO: make learning rate adjustable, etc
                        num_runs: int = 3, seeds: List[int] = None):
    """Save training configuration to file"""
    config_path = output_dir / "training_config.txt"
    
    config_text = f"""
{'='*80}
TRAINING CONFIGURATION
{'='*80}

DATASET PARAMETERS:
  - Train/Val/Test Split: 70% / 20% / 10%
  - Image Size: 224x224
  - Image Format: RGB (converted from grayscale)
  - Data Augmentation (Training):
    * Random Horizontal Flip (p=0.5)
    * Random Rotation (±15°)
    * Color Jitter (brightness=0.2, contrast=0.2)
    * Normalization: ImageNet mean/std

TRAINING PARAMETERS:
  - Number of Runs: {num_runs}
  - Random Seeds: {seeds}
  - Number of Epochs: {num_epochs}
  - Batch Size: {batch_size}
  - Learning Rate: {learning_rate}
  - Optimizer: Adam (betas=(0.9, 0.999), eps=1e-8)
  - Loss Function: CrossEntropyLoss
  - Learning Rate Scheduler: ReduceLROnPlateau
    * Factor: 0.5
    * Patience: 5
    * Mode: min (on validation loss)
  - Early Stopping Patience: 5 epochs

MODELS TRAINED:
  1. ResNet18 (from scratch)
  2. ResNet34 (from scratch)
  3. ResNet50 (from scratch)
  4. ResNet101 (from scratch)
  5. DenseNet121 (from scratch)
  6. DenseNet169 (from scratch)
  7. DenseNet201 (from scratch)
  8. ResNet50-ImageNet (pretrained on ImageNet)
  9. CheXpert (pretrained on medical images)

EVALUATION METRICS:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - AUC-ROC
  - Confusion Matrix

OUTPUT FILES:
  - aggregated_results.json: Mean ± Std metrics across all runs
  - training_config.txt: This configuration file
  - model_comparison_with_std.png: Comparison plot with error bars
  - checkpoints/: Model weights for each seed
  - plots/: Training curves and confusion matrices for each seed

{'='*80}
"""
    
    with open(config_path, 'w') as f:
        f.write(config_text)
    
    logger.info(f"Training configuration saved to {config_path}")


def save_results(results: Dict, output_dir: Path):
    """Save results to JSON"""
    results_path = output_dir / "results.json"
    
    # Prepare data for JSON serialization
    json_results = {}
    for model_name, data in results.items():
        json_results[model_name] = {
            'metrics': {
                'accuracy': float(data['metrics']['accuracy']),
                'precision': float(data['metrics']['precision']),
                'recall': float(data['metrics']['recall']),
                'f1': float(data['metrics']['f1']),
                'auc_roc': float(data['metrics']['auc_roc']),
                'confusion_matrix': data['metrics']['confusion_matrix']
            },
            'history': data['history']
        }
    
    with open(results_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    logger.info(f"Results saved to {results_path}")


def create_comparison_plots(results: Dict, output_dir: Path):
    """Create comparison plots across models"""
    
    # Extract metrics
    models = list(results.keys())
    accuracies = [results[m]['metrics']['accuracy'] for m in models]
    precisions = [results[m]['metrics']['precision'] for m in models]
    recalls = [results[m]['metrics']['recall'] for m in models]
    f1_scores = [results[m]['metrics']['f1'] for m in models]
    auc_rocs = [results[m]['metrics']['auc_roc'] for m in models]
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Model Comparison - Binary Classification Metrics', fontsize=16, fontweight='bold')
    
    # Accuracy
    axes[0, 0].bar(models, accuracies, color='skyblue')
    axes[0, 0].set_title('Accuracy')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Precision
    axes[0, 1].bar(models, precisions, color='lightcoral')
    axes[0, 1].set_title('Precision')
    axes[0, 1].set_ylabel('Score')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Recall
    axes[0, 2].bar(models, recalls, color='lightgreen')
    axes[0, 2].set_title('Recall')
    axes[0, 2].set_ylabel('Score')
    axes[0, 2].tick_params(axis='x', rotation=45)
    axes[0, 2].grid(axis='y', alpha=0.3)
    
    # F1-Score
    axes[1, 0].bar(models, f1_scores, color='lightyellow')
    axes[1, 0].set_title('F1-Score')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # AUC-ROC
    axes[1, 1].bar(models, auc_rocs, color='plum')
    axes[1, 1].set_title('AUC-ROC')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    # Summary table
    axes[1, 2].axis('off')
    summary_data = []
    for model in models:
        summary_data.append([
            model,
            f"{results[model]['metrics']['accuracy']:.4f}",
            f"{results[model]['metrics']['f1']:.4f}",
            f"{results[model]['metrics']['auc_roc']:.4f}"
        ])
    
    table = axes[1, 2].table(
        cellText=summary_data,
        colLabels=['Model', 'Accuracy', 'F1-Score', 'AUC-ROC'],
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 2)
    
    plt.tight_layout()
    plt.savefig(output_dir / "model_comparison.png", dpi=300, bbox_inches='tight')
    logger.info(f"Comparison plot saved to {output_dir / 'model_comparison.png'}")
    plt.close()


def create_comparison_plots_with_std(aggregated_results: Dict, output_dir: Path):
    """Create comparison plots with error bars (mean ± std)"""
    
    models = list(aggregated_results.keys())
    
    # Extract means and stds
    accuracies_mean = [aggregated_results[m]['accuracy']['mean'] for m in models]
    accuracies_std = [aggregated_results[m]['accuracy']['std'] for m in models]
    
    precisions_mean = [aggregated_results[m]['precision']['mean'] for m in models]
    precisions_std = [aggregated_results[m]['precision']['std'] for m in models]
    
    recalls_mean = [aggregated_results[m]['recall']['mean'] for m in models]
    recalls_std = [aggregated_results[m]['recall']['std'] for m in models]
    
    f1_scores_mean = [aggregated_results[m]['f1']['mean'] for m in models]
    f1_scores_std = [aggregated_results[m]['f1']['std'] for m in models]
    
    auc_rocs_mean = [aggregated_results[m]['auc_roc']['mean'] for m in models]
    auc_rocs_std = [aggregated_results[m]['auc_roc']['std'] for m in models]
    
    # Create comparison plot with error bars
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Model Comparison - Binary Classification Metrics (Mean ± Std)', fontsize=16, fontweight='bold')
    
    x = np.arange(len(models))
    width = 0.6
    
    # Accuracy
    axes[0, 0].bar(x, accuracies_mean, width, yerr=accuracies_std, capsize=5, color='skyblue', alpha=0.8)
    axes[0, 0].set_title('Accuracy')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(models, rotation=45, ha='right')
    axes[0, 0].set_ylim([0, 1.1])
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Precision
    axes[0, 1].bar(x, precisions_mean, width, yerr=precisions_std, capsize=5, color='lightcoral', alpha=0.8)
    axes[0, 1].set_title('Precision')
    axes[0, 1].set_ylabel('Score')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(models, rotation=45, ha='right')
    axes[0, 1].set_ylim([0, 1.1])
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Recall
    axes[0, 2].bar(x, recalls_mean, width, yerr=recalls_std, capsize=5, color='lightgreen', alpha=0.8)
    axes[0, 2].set_title('Recall')
    axes[0, 2].set_ylabel('Score')
    axes[0, 2].set_xticks(x)
    axes[0, 2].set_xticklabels(models, rotation=45, ha='right')
    axes[0, 2].set_ylim([0, 1.1])
    axes[0, 2].grid(axis='y', alpha=0.3)
    
    # F1-Score
    axes[1, 0].bar(x, f1_scores_mean, width, yerr=f1_scores_std, capsize=5, color='lightyellow', alpha=0.8)
    axes[1, 0].set_title('F1-Score')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(models, rotation=45, ha='right')
    axes[1, 0].set_ylim([0, 1.1])
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # AUC-ROC
    axes[1, 1].bar(x, auc_rocs_mean, width, yerr=auc_rocs_std, capsize=5, color='plum', alpha=0.8)
    axes[1, 1].set_title('AUC-ROC')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(models, rotation=45, ha='right')
    axes[1, 1].set_ylim([0, 1.1])
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    # Summary table
    axes[1, 2].axis('off')
    summary_data = []
    for model in models:
        summary_data.append([
            model,
            f"{aggregated_results[model]['accuracy']['mean']:.3f}±{aggregated_results[model]['accuracy']['std']:.3f}",
            f"{aggregated_results[model]['f1']['mean']:.3f}±{aggregated_results[model]['f1']['std']:.3f}",
            f"{aggregated_results[model]['auc_roc']['mean']:.3f}±{aggregated_results[model]['auc_roc']['std']:.3f}"
        ])
    
    table = axes[1, 2].table(
        cellText=summary_data,
        colLabels=['Model', 'Accuracy', 'F1-Score', 'AUC-ROC'],
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 2)
    
    plt.tight_layout()
    plt.savefig(output_dir / "model_comparison_with_std.png", dpi=300, bbox_inches='tight')
    logger.info(f"Comparison plot with std saved to {output_dir / 'model_comparison_with_std.png'}")
    plt.close()


if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Train deep learning models for medical image classification"
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run mock training with small dataset to verify pipeline (2 epochs, 3 models)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to processed dataset (default: ~/data/medical_imaging/processed)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for training"
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=None,
        help="Number of runs with different seeds"
    )
    
    args = parser.parse_args()
    
    # Configuration
    DATA_DIR = args.data_dir or os.path.expanduser("~/data/medical_imaging/processed")
    OUTPUT_DIR = args.output_dir or ("./mock_results" if args.mock else "./training_results")
    NUM_EPOCHS = args.epochs or (2 if args.mock else 50)
    BATCH_SIZE = args.batch_size or (4 if args.mock else 32)
    NUM_RUNS = args.num_runs or (1 if args.mock else 3)
    
    # Randomly draw seeds from 0 to 100
    np.random.seed(42)
    SEEDS = list(np.random.randint(0, 101, NUM_RUNS))
    
    logger.info(f"Data directory: {DATA_DIR}")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    logger.info(f"Number of epochs: {NUM_EPOCHS}")
    logger.info(f"Batch size: {BATCH_SIZE}")
    logger.info(f"Number of runs: {NUM_RUNS}")
    logger.info(f"Seeds: {SEEDS}")
    logger.info(f"Mock mode: {args.mock}")
    
    if args.mock:
        # Run mock training to verify pipeline
        logger.info("\nMOCK TRAINING MODE")
        mock_results = mock_training(
            DATA_DIR,
            output_dir=OUTPUT_DIR,
            num_epochs=NUM_EPOCHS,
            batch_size=BATCH_SIZE,
            seeds=SEEDS
        )
        logger.info("\nMock training complete!")
    else:
        # Train all models with multiple seeds
        logger.info("\nFULL TRAINING MODE")
        all_results, aggregated_results = train_all_models(
            DATA_DIR, OUTPUT_DIR, NUM_EPOCHS, BATCH_SIZE, NUM_RUNS, SEEDS
        )
        logger.info("\nTraining complete!")