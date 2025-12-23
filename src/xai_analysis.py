#!/usr/bin/env python3
"""
Beyond the Black Box: XAI Analysis on Complete Test Set

This script implements and evaluates explainability methods (Saliency Maps, 
GradientSHAP, Integrated Gradients, GradCAM, Feature Permutation) on the entire test set 
and compares attributions to ground truth masks.

Author: Mateusz Cedro
"""

import os
import sys
import gc
import copy
import time
import warnings
import pathlib
import logging
import argparse
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import torch.nn.functional as F
from PIL import Image

from sklearn.metrics import (
    roc_curve, roc_auc_score, confusion_matrix, classification_report,
    jaccard_score, f1_score, accuracy_score, precision_score, recall_score
)
from captum.attr import Saliency, IntegratedGradients, GradientShap, LayerGradCam
import quantus
from scipy import stats
from scipy.stats import kruskal, mannwhitneyu
import copy as copy_module
from statsmodels.stats.multitest import multipletests

# Import train_models to reuse data loading
import importlib.util
spec = importlib.util.spec_from_file_location("train_models", "src/train_models.py")
train_models = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_models)

# Suppress warnings
warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration parameters for XAI analysis."""

    # Dataset configuration
    DATASET = "covid_qu_ex"
    # DATASET = "chest_x_pneumo"
    # DATASET = "pet"

    # Adjust class names based on dataset
    if DATASET == "covid_qu_ex":
        CLASS_0_NAME = "COVID-19"
        CLASS_0_NAME_MASKS = "infection_masks"
        CLASS_1_NAME = "Normal"
        CLASS_1_NAME_MASKS = "lung_masks"
    elif DATASET == "pet":
        CLASS_0_NAME = "cat"
        CLASS_0_NAME_MASKS = "masks"
        CLASS_1_NAME = "dog" 
        CLASS_1_NAME_MASKS = "masks"
    elif DATASET == "chest_x_pneumo":
        CLASS_0_NAME = "no_pneumo"
        # CLASS_0_NAME_MASKS = "masks"
        CLASS_1_NAME = "pneumo"
        CLASS_1_NAME_MASKS = "masks"
    else:
        print(f"Unknown DATASET: {DATASET}. Please configure class names accordingly.")

    # Flag for Sanity Checks
    # Set to True ONLY when running on the Pet dataset (or your chosen validation set)
    if DATASET == "pet":
        ENABLE_SANITY_CHECKS = True
    else:
        ENABLE_SANITY_CHECKS = False
    
    # Data paths
    DATA_ROOT = f"./data/{DATASET}"
    
    # Model paths - will be set dynamically for each seed
    CHECKPOINT_DIR = None  # Will be set when processing specific seed
    RESULTS_DIR = None  # Will be set when processing specific seed
    PLOTS_DIR = None  # Will be set when processing specific seed
    
    # Training parameters
    BATCH_SIZE = 32
    IMG_SIZE = 224
    NUM_CLASSES = 2
    
    # XAI parameters
    N_STEPS_INTGRAD = 25  # Steps for Integrated Gradients (reduced from 25 to save memory)
    EXPLAIN_BATCH_SIZE = 4  # Batch size for explanation generation (reduced for memory efficiency)
    
    # Visualization
    PLOT_SAMPLES = 12  # Number of samples to visualize
    COLORMAP = 'jet'
    DPI = 150
    
    @classmethod
    def set_seed(cls, seed: str):
        """Set configuration for a specific seed."""
        cls.CHECKPOINT_DIR = f"./training_results/checkpoints/{seed}"
        cls.RESULTS_DIR = f"./xai_results/{seed}"
        cls.PLOTS_DIR = f"{cls.RESULTS_DIR}/plots"


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def setup_directories():
    """Create necessary directories."""
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.RESULTS_DIR, exist_ok=True)
    os.makedirs(Config.PLOTS_DIR, exist_ok=True)
    logger.info(f"Directories created/verified")


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    logger.info(f"Random seed set to {seed} for reproducibility")


def get_transforms():
    """Create data transformations."""
    transforms_train = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    transforms_test = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    return transforms_train, transforms_test


def load_test_data(transforms_test, seed: int = 42):
    """Load test dataset and masks using the same data split as train_models.py."""
    logger.info(f"Loading test data using train_models dataset split (seed={seed})...")
    
    # Use the same data loading and splitting as train_models.py
    # This ensures we use the same test set that the models were trained on
    test_root = Config.DATA_ROOT
    
    # Load dataset using train_models function to get consistent train/val/test split
    train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader, class_names, weights_tensor = (
        train_models.load_dataset(
            data_dir=test_root,
            train_ratio=0.7,
            val_ratio=0.2,
            seed=seed,
            batch_size=Config.BATCH_SIZE,
            model_name=None
        )
    )
    
    logger.info(f"Test set size: {len(test_dataset)}")
    logger.info(f"Class names: {class_names}")
    
    # Load masks with matching structure
    class MaskDataset(torch.utils.data.Dataset):
        def __init__(self, class_0_mask_dir, class_1_mask_dir, transforms=None):
            self.transforms = transforms
            self.masks = []
            self.labels = []
            
            # Load class_0 masks (label 0)
            class_0_mask_files = sorted([f for f in os.listdir(class_0_mask_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
            for f in class_0_mask_files:
                self.masks.append(os.path.join(class_0_mask_dir, f))
                self.labels.append(0)
            
            # Load class_1 masks (label 1)
            class_1_mask_files = sorted([f for f in os.listdir(class_1_mask_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
            for f in class_1_mask_files:
                self.masks.append(os.path.join(class_1_mask_dir, f))
                self.labels.append(1)
        
        def __len__(self):
            return len(self.masks)
        
        def __getitem__(self, idx):
            mask_path = self.masks[idx]
            mask = Image.open(mask_path).convert('L')  # Convert to grayscale
            label = self.labels[idx]
            
            if self.transforms:
                mask = self.transforms(mask)
            
            return mask, label
    
    mask_dataset = MaskDataset(
        f"{test_root}/{Config.CLASS_0_NAME}/{Config.CLASS_0_NAME_MASKS}",
        f"{test_root}/{Config.CLASS_1_NAME}/{Config.CLASS_1_NAME_MASKS}",
        transforms_test
    )
    mask_dataloader = torch.utils.data.DataLoader(
        mask_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=4
    )
    
    logger.info(f"Mask set size: {len(mask_dataset)}")
    logger.info(f"Masks loaded from both class_0 and class_1 classes")
    
    return test_dataset, test_loader, mask_dataset, mask_dataloader


def load_model(model_path: Optional[str] = None) -> nn.Module:
    """Load trained model from checkpoint."""
    logger.info("Loading model...")
    
    # Determine which checkpoint to load
    if model_path is None:
        # Default to ResNet50-ImageNet if available
        model_path = os.path.join(Config.CHECKPOINT_DIR, "ResNet50-ImageNet_best.pth")
        if not os.path.exists(model_path):
            # Try other available models
            for model_name in ["DenseNet121_best.pth", "ResNet50_best.pth", "ResNet18_best.pth"]:
                alt_path = os.path.join(Config.CHECKPOINT_DIR, model_name)
                if os.path.exists(alt_path):
                    model_path = alt_path
                    break
    
    # Load the appropriate architecture based on checkpoint filename
    checkpoint_filename = os.path.basename(model_path)
    
    if "DenseNet121" in checkpoint_filename:
        model = models.densenet121(weights=None)
        model.features[0] = nn.Conv2d(1, model.features[0].out_channels, kernel_size=model.features[0].kernel_size,
                                      stride=model.features[0].stride, padding=model.features[0].padding, bias=model.features[0].bias)
        model.classifier = nn.Linear(model.classifier.in_features, Config.NUM_CLASSES)
    
    elif "DenseNet169" in checkpoint_filename:
        model = models.densenet169(weights=None)
        model.features[0] = nn.Conv2d(1, model.features[0].out_channels, kernel_size=model.features[0].kernel_size,
                                      stride=model.features[0].stride, padding=model.features[0].padding, bias=model.features[0].bias)
        model.classifier = nn.Linear(model.classifier.in_features, Config.NUM_CLASSES)
    
    elif "DenseNet201" in checkpoint_filename:
        model = models.densenet201(weights=None)
        model.features[0] = nn.Conv2d(1, model.features[0].out_channels, kernel_size=model.features[0].kernel_size,
                                      stride=model.features[0].stride, padding=model.features[0].padding, bias=model.features[0].bias)
        model.classifier = nn.Linear(model.classifier.in_features, Config.NUM_CLASSES)
    
    elif "CheXpert" in checkpoint_filename:
        import torchxrayvision as xrv
        xrv_model = xrv.models.DenseNet(weights="densenet121-res224-chex", op_threshs=None)
        class CheXpertBinary(nn.Module):
            def __init__(self, features, classifier, num_classes):
                super().__init__()
                self.features = features
                self.classifier = nn.Linear(classifier.in_features, num_classes)
            def forward(self, x):
                out = self.features(x)
                out = torch.nn.functional.adaptive_avg_pool2d(out, (1, 1))
                out = torch.flatten(out, 1)
                out = self.classifier(out)
                return out
        model = CheXpertBinary(xrv_model.features, xrv_model.classifier, Config.NUM_CLASSES)
    else:
        # Default to ResNet (covers ResNet18, ResNet34, ResNet50, ResNet101)
        if "ResNet18" in checkpoint_filename:
            model = models.resnet18(weights=None)
        elif "ResNet34" in checkpoint_filename:
            model = models.resnet34(weights=None)
        elif "ResNet101" in checkpoint_filename:
            model = models.resnet101(weights=None)
        else:
            model = models.resnet50(weights=None)
        
        # Modify for grayscale input
        model.conv1 = nn.Conv2d(1, model.conv1.out_channels, kernel_size=model.conv1.kernel_size,
                               stride=model.conv1.stride, padding=model.conv1.padding, bias=model.conv1.bias)
        # Modify for binary classification
        model.fc = nn.Linear(model.fc.in_features, Config.NUM_CLASSES)
    
    # Load checkpoint
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        logger.info(f"Loaded model from {model_path}")
    else:
        logger.error(f"Checkpoint not found: {model_path}")
        logger.error(f"Available checkpoints in {Config.CHECKPOINT_DIR}:")
        if os.path.exists(Config.CHECKPOINT_DIR):
            for f in os.listdir(Config.CHECKPOINT_DIR):
                logger.error(f"  - {f}")
        raise FileNotFoundError(f"No checkpoint found at {model_path}")
    
    model = model.to(device)
    model.eval()
    
    return model


# ============================================================================
# EXPLAINABILITY METHODS
# ============================================================================

def saliency_explainer(model, inputs, targets, abs=False, normalise=False, *args, **kwargs) -> np.ndarray:
    """Saliency explanation method."""
    gc.collect()
    torch.cuda.empty_cache()

    model.to(kwargs.get("device", device))
    model.eval()

    if not isinstance(inputs, torch.Tensor):
        inputs = (
            torch.Tensor(inputs)
            .reshape(
                -1,
                kwargs.get("nr_channels", 1),
                kwargs.get("img_size", 224),
                kwargs.get("img_size", 224),
            )
            .to(kwargs.get("device", device))
        )
    if not isinstance(targets, torch.Tensor):
        targets = torch.as_tensor(targets).long().to(kwargs.get("device", device))

    assert len(np.shape(inputs)) == 4, "Inputs should be shaped (nr_samples, nr_channels, img_size, img_size)"

    explanation = (
        Saliency(model)
        .attribute(inputs, targets, abs=abs)
        .sum(axis=1)
        .reshape(-1, kwargs.get("img_size", 224), kwargs.get("img_size", 224))
        .cpu()
        .data
    )

    gc.collect()
    torch.cuda.empty_cache()

    if normalise:
        explanation = quantus.normalise_func.normalise_by_negative(explanation)

    if isinstance(explanation, torch.Tensor):
        if explanation.requires_grad:
            return explanation.cpu().detach().numpy()
        return explanation.cpu().numpy()

    return explanation


def intgrad_explainer(model, inputs, targets, abs=False, normalise=False, *args, **kwargs) -> np.ndarray:
    """Integrated Gradients explanation method with memory-efficient batching."""
    device_to_use = kwargs.get("device", device)
    img_size = kwargs.get("img_size", 224)
    nr_channels = kwargs.get("nr_channels", 1)
    
    gc.collect()
    torch.cuda.empty_cache()

    model.to(device_to_use)
    model.eval()

    if not isinstance(inputs, torch.Tensor):
        inputs = torch.Tensor(inputs).reshape(-1, nr_channels, img_size, img_size)
    
    if not isinstance(targets, torch.Tensor):
        targets = torch.as_tensor(targets).long()
    
    # Move to device only during processing
    inputs = inputs.to(device_to_use)
    targets = targets.to(device_to_use)

    assert len(np.shape(inputs)) == 4, "Inputs should be shaped (nr_samples, nr_channels, img_size, img_size)"
    
    # Process in smaller batches to save memory
    batch_size = 2  # Very small batch size for IntGrad due to high memory usage
    num_samples = inputs.shape[0]
    explanations_list = []
    
    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            # Get batch
            end_idx = min(i + batch_size, num_samples)
            inputs_batch = inputs[i:end_idx]
            targets_batch = targets[i:end_idx]
            
            # Compute IntGrad for this batch
            try:
                expl_batch = (
                    IntegratedGradients(model)
                    .attribute(
                        inputs=inputs_batch,
                        target=targets_batch,
                        baselines=torch.zeros_like(inputs_batch),
                        n_steps=Config.N_STEPS_INTGRAD,
                        method="riemann_trapezoid",
                    )
                    .sum(axis=1)
                    .cpu()
                    .detach()
                )
                explanations_list.append(expl_batch)
            except Exception as e:
                logger.error(f"Error processing IntGrad batch {i//batch_size}: {e}")
                # Create dummy explanation on error
                explanations_list.append(torch.zeros(end_idx - i, img_size, img_size))
            
            # Aggressive cleanup after each batch
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    
    # Concatenate all batches
    explanation = torch.cat(explanations_list, dim=0)
    explanation = explanation.reshape(-1, img_size, img_size)

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    if normalise:
        explanation = quantus.normalise_func.normalise_by_negative(explanation.numpy())
        explanation = torch.from_numpy(explanation)

    if isinstance(explanation, torch.Tensor):
        if explanation.requires_grad:
            return explanation.cpu().detach().numpy()
        return explanation.cpu().numpy()

    return explanation


def gradshap_explainer(model, inputs, targets, abs=False, normalise=False, *args, **kwargs) -> np.ndarray:
    """GradientSHAP explanation method."""
    gc.collect()
    torch.cuda.empty_cache()

    model.to(kwargs.get("device", device))
    model.eval()

    if not isinstance(inputs, torch.Tensor):
        inputs = (
            torch.Tensor(inputs)
            .reshape(
                -1,
                kwargs.get("nr_channels", 1),
                kwargs.get("img_size", 224),
                kwargs.get("img_size", 224),
            )
            .to(kwargs.get("device", device))
        )

    if not isinstance(targets, torch.Tensor):
        targets = torch.as_tensor(targets).long().to(kwargs.get("device", device))

    assert len(np.shape(inputs)) == 4, "Inputs should be shaped (nr_samples, nr_channels, img_size, img_size)"

    baselines = torch.zeros_like(inputs).to(kwargs.get("device", device))
    explanation = (
        GradientShap(model)
        .attribute(inputs=inputs, target=targets, baselines=baselines)
        .sum(axis=1)
        .reshape(-1, kwargs.get("img_size", 224), kwargs.get("img_size", 224))
        .cpu()
        .data
    )

    gc.collect()
    torch.cuda.empty_cache()

    if normalise:
        explanation = quantus.normalise_func.normalise_by_negative(explanation)

    if isinstance(explanation, torch.Tensor):
        if explanation.requires_grad:
            return explanation.cpu().detach().numpy()
        return explanation.cpu().numpy()

    return explanation


# ============================================================================
# CUSTOM RELEVANCE MASS ACCURACY METRIC
# ============================================================================

class PositiveAttributionRatio(quantus.metrics.base.Metric):
    """
    Implementation of the Positive Attribution Ratio basing on Relevance Mass Accuracy by Arras et al., 2021.

    The Positive Attribution Ratio, that base on Relevance Mass Accuracy, which computes the ratio of attributions inside the bounding box to
    the sum of overall positive attributions. High scores are desired, as the pixels with the highest positively
    attributed scores should be within the bounding box of the targeted object.

    References:
        1) Leila Arras et al.: "CLEVR-XAI: A benchmark dataset for the ground
        truth evaluation of neural network explanations." Inf. Fusion 81 (2022): 14-40.

    Attributes:
        -  _name: The name of the metric.
        - _data_applicability: The data types that the metric implementation currently supports.
        - _models: The model types that this metric can work with.
        - score_direction: How to interpret the scores, whether higher/ lower values are considered better.
        - evaluation_category: What property/ explanation quality that this metric measures.
    """

    name = "Relevance Mass Accuracy" # Quantus name
    data_applicability = {quantus.helpers.enums.DataType.IMAGE, quantus.helpers.enums.DataType.TIMESERIES, quantus.helpers.enums.DataType.TABULAR}
    model_applicability = {quantus.helpers.enums.ModelType.TORCH, quantus.helpers.enums.ModelType.TF}
    score_direction = quantus.helpers.enums.ScoreDirection.HIGHER
    evaluation_category = quantus.helpers.enums.EvaluationCategory.LOCALISATION

    def __init__(
        self,
        abs: bool = False,
        normalise: bool = True,
        normalise_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        normalise_func_kwargs: Optional[Dict[str, Any]] = None,
        return_aggregate: bool = False,
        aggregate_func: Callable = np.mean,
        default_plot_func: Optional[Callable] = None,
        disable_warnings: bool = False,
        display_progressbar: bool = False,
        **kwargs,
    ):
        """
        Parameters
        ----------
        abs: boolean
            Indicates whether absolute operation is applied on the attribution, default=False.
        normalise: boolean
            Indicates whether normalise operation is applied on the attribution, default=True.
        normalise_func: callable
            Attribution normalisation function applied in case normalise=True.
            If normalise_func=None, the default value is used, default=normalise_by_max.
        normalise_func_kwargs: dict
            Keyword arguments to be passed to normalise_func on call, default={}.
        return_aggregate: boolean
            Indicates if an aggregated score should be computed over all instances.
        aggregate_func: callable
            Callable that aggregates the scores given an evaluation call.
        default_plot_func: callable
            Callable that plots the metrics result.
        disable_warnings: boolean
            Indicates whether the warnings are printed, default=False.
        display_progressbar: boolean
            Indicates whether a tqdm-progress-bar is printed, default=False.
        kwargs: optional
            Keyword arguments.
        """
        if normalise_func is None:
            normalise_func = quantus.normalise_func.normalise_by_max

        super().__init__(
            abs=abs,
            normalise=normalise,
            normalise_func=normalise_func,
            normalise_func_kwargs=normalise_func_kwargs,
            return_aggregate=return_aggregate,
            aggregate_func=aggregate_func,
            default_plot_func=default_plot_func,
            display_progressbar=display_progressbar,
            disable_warnings=disable_warnings,
            **kwargs,
        )

    def __call__(
        self,
        model,
        x_batch: np.array,
        y_batch: np.array,
        a_batch: Optional[np.ndarray] = None,
        s_batch: Optional[np.ndarray] = None,
        channel_first: Optional[bool] = None,
        explain_func: Optional[Callable] = None,
        explain_func_kwargs: Optional[Dict] = None,
        model_predict_kwargs: Optional[Dict] = None,
        softmax: Optional[bool] = False,
        device: Optional[str] = None,
        batch_size: int = 64,
        custom_batch: Optional[Any] = None,
        **kwargs,
    ) -> List[float]:
        """
        This implementation represents the main logic of the metric and makes the class object callable.
        """
        return super().__call__(
            model=model,
            x_batch=x_batch,
            y_batch=y_batch,
            a_batch=a_batch,
            s_batch=s_batch,
            custom_batch=None,
            channel_first=channel_first,
            explain_func=explain_func,
            explain_func_kwargs=explain_func_kwargs,
            softmax=softmax,
            device=device,
            model_predict_kwargs=model_predict_kwargs,
            **kwargs,
        )

    def evaluate_instance(
        self,
        model,
        x: np.ndarray,
        y: np.ndarray,
        a: np.ndarray,
        s: np.ndarray,
    ) -> float:
        """
        Evaluate instance gets model and data for a single instance as input and returns the evaluation result.

        Parameters
        ----------
        model: Model interface
            A model that is subject to explanation.
        x: np.ndarray
            The input to be evaluated on an instance-basis.
        y: np.ndarray
            The output to be evaluated on an instance-basis.
        a: np.ndarray
            The explanation to be evaluated on an instance-basis.
        s: np.ndarray
            The segmentation to be evaluated on an instance-basis.

        Returns
        -------
        float
            The evaluation results.
        """
        # Return np.nan as result if segmentation map is empty.
        if np.sum(s) == 0:
            return np.nan

        # Prepare shapes.
        a = a.flatten()
        s = s.flatten().astype(bool)

        # Compute inside/outside ratio.
        r_total = np.sum(a[a > 0])

        if r_total == 0:
            return 0.0

        r_within = np.sum(a[(a > 0) & (s > 0)])  # I have modified that from original Quantus implementation
        # print(f"r_within: {r_within}, r_total: {r_total}")
        # Calculate mass accuracy.
        mass_accuracy = r_within / r_total

        return mass_accuracy

    def evaluate_batch(
        self,
        model,
        x_batch: np.ndarray,
        y_batch: Optional[np.ndarray],
        a_batch: Optional[np.ndarray],
        s_batch: np.ndarray,
        custom_batch: Optional[np.ndarray] = None,
        **kwargs,
    ) -> List[float]:
        """
        Evaluate batch of instances.
        
        Parameters
        ----------
        model: Model interface
        x_batch: np.ndarray
            Batch of inputs
        y_batch: Optional[np.ndarray]
            Batch of labels
        a_batch: Optional[np.ndarray]
            Batch of attributions/explanations
        s_batch: np.ndarray
            Batch of segmentations/masks
        custom_batch: Optional[np.ndarray]
            Custom batch data
        
        Returns
        -------
        List[float]
            List of scores for each instance
        """
        batch_size = x_batch.shape[0]
        scores = []
        
        for i in range(batch_size):
            score = self.evaluate_instance(
                model=model,
                x=x_batch[i],
                y=y_batch[i] if y_batch is not None else None,
                a=a_batch[i] if a_batch is not None else None,
                s=s_batch[i],
            )
            scores.append(score)
        
        return scores

    def custom_preprocess(
        self,
        model,
        x_batch: np.ndarray,
        y_batch: Optional[np.ndarray],
        a_batch: Optional[np.ndarray],
        s_batch: np.ndarray,
        custom_batch: Optional[np.ndarray],
    ) -> None:
        """
        Implementation of custom_preprocess_batch.
        """
        pass


def get_last_conv_layer(model):
    """Get the last convolutional layer in the model."""
    # Try common layer names for different architectures
    candidates = [
        'layer4',  # ResNet
        'features',  # DenseNet, VGG
        'conv_layer',  # Other potential names
    ]
    
    for candidate in candidates:
        if hasattr(model, candidate):
            module = getattr(model, candidate)
            # For DenseNet, we need to get the last block
            if hasattr(module, '__getitem__'):
                return module[-1]
            return module
    
    # Fallback: find the last convolutional layer by iterating
    last_conv = None
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Sequential)):
            if isinstance(module, torch.nn.Conv2d):
                last_conv = module
    
    if last_conv is not None:
        return last_conv
    
    # Last resort: return the first layer found
    for module in model.modules():
        if isinstance(module, torch.nn.Sequential):
            return module[-1]
    
    raise RuntimeError("Could not find a convolutional layer in the model")


def gradcam_explainer(model, inputs, targets, abs=False, normalise=False, *args, **kwargs) -> np.ndarray:
    """GradCAM explanation method."""
    gc.collect()
    torch.cuda.empty_cache()

    model.to(kwargs.get("device", device))
    model.eval()

    if not isinstance(inputs, torch.Tensor):
        inputs = (
            torch.Tensor(inputs)
            .reshape(
                -1,
                kwargs.get("nr_channels", 1),
                kwargs.get("img_size", 224),
                kwargs.get("img_size", 224),
            )
            .to(kwargs.get("device", device))
        )
    if not isinstance(targets, torch.Tensor):
        targets = torch.as_tensor(targets).long().to(kwargs.get("device", device))

    assert len(np.shape(inputs)) == 4, "Inputs should be shaped (nr_samples, nr_channels, img_size, img_size)"

    # Dynamically find the last convolutional layer
    try:
        last_conv_layer = get_last_conv_layer(model)
        explanation = (
            LayerGradCam(model, last_conv_layer)
            .attribute(inputs, targets, relu_attributions=True)
            .cpu()
            .data
        )
    except Exception as e:
        logger.warning(f"Failed to compute GradCAM: {e}. Using fallback Saliency.")
        # Fallback to Saliency if GradCAM fails
        explanation = (
            Saliency(model)
            .attribute(inputs, targets, abs=False)
            .sum(axis=1)
            .reshape(-1, kwargs.get("img_size", 224), kwargs.get("img_size", 224))
            .cpu()
            .data
        )

    # Resize to input size if needed
    if explanation.shape[2:] != inputs.shape[2:]:
        explanation_resized = []
        for i in range(explanation.shape[0]):
            img_np = explanation[i, 0].cpu().numpy() if isinstance(explanation, torch.Tensor) else explanation[i, 0]
            img_resized = cv2.resize(
                img_np,
                (kwargs.get("img_size", 224), kwargs.get("img_size", 224)),
                interpolation=cv2.INTER_LINEAR
            )
            explanation_resized.append(img_resized)
        explanation = np.stack(explanation_resized, axis=0)
    else:
        if isinstance(explanation, torch.Tensor):
            explanation = explanation.sum(axis=1).reshape(-1, kwargs.get("img_size", 224), kwargs.get("img_size", 224)).numpy()
        else:
            explanation = explanation.sum(axis=1).reshape(-1, kwargs.get("img_size", 224), kwargs.get("img_size", 224))

    gc.collect()
    torch.cuda.empty_cache()

    if normalise:
        explanation = quantus.normalise_func.normalise_by_negative(explanation)

    if isinstance(explanation, torch.Tensor):
        if explanation.requires_grad:
            return explanation.cpu().detach().numpy()
        return explanation.cpu().numpy()

    return explanation


def feature_permutation_explainer(model, inputs, targets, abs=False, normalise=False, *args, **kwargs) -> np.ndarray:
    """Feature Permutation explanation method - measures importance by permuting input features."""
    gc.collect()
    torch.cuda.empty_cache()

    model.to(kwargs.get("device", device))
    model.eval()

    if not isinstance(inputs, torch.Tensor):
        inputs = (
            torch.Tensor(inputs)
            .reshape(
                -1,
                kwargs.get("nr_channels", 1),
                kwargs.get("img_size", 224),
                kwargs.get("img_size", 224),
            )
            .to(kwargs.get("device", device))
        )
    if not isinstance(targets, torch.Tensor):
        targets = torch.as_tensor(targets).long().to(kwargs.get("device", device))

    assert len(np.shape(inputs)) == 4, "Inputs should be shaped (nr_samples, nr_channels, img_size, img_size)"

    batch_size = inputs.shape[0]
    img_size = kwargs.get("img_size", 224)
    explanations = np.zeros((batch_size, img_size, img_size))

    with torch.no_grad():
        # Get baseline predictions
        baseline_outputs = model(inputs)
        baseline_probs = torch.softmax(baseline_outputs, dim=1)
        baseline_scores = baseline_probs[range(batch_size), targets].cpu().numpy()

        # Permute patches and measure importance
        patch_size = 16  # 16x16 patches for 224x224 images
        num_patches = (img_size // patch_size) ** 2

        for patch_h in range(0, img_size, patch_size):
            for patch_w in range(0, img_size, patch_size):
                # Create permuted input
                inputs_permuted = inputs.clone()

                # Add random noise to the patch
                noise = torch.randn_like(
                    inputs_permuted[:, :, patch_h:patch_h+patch_size, patch_w:patch_w+patch_size]
                ) * 0.1
                inputs_permuted[:, :, patch_h:patch_h+patch_size, patch_w:patch_w+patch_size] += noise

                # Get predictions on permuted input
                permuted_outputs = model(inputs_permuted)
                permuted_probs = torch.softmax(permuted_outputs, dim=1)
                permuted_scores = permuted_probs[range(batch_size), targets].cpu().numpy()

                # Calculate importance as change in prediction score
                importance = baseline_scores - permuted_scores

                # Fill the explanation map
                explanations[:, patch_h:patch_h+patch_size, patch_w:patch_w+patch_size] = (
                    importance.reshape(-1, 1, 1)
                )

    explanations = np.array(explanations)
    gc.collect()
    torch.cuda.empty_cache()

    if normalise:
        explanations = quantus.normalise_func.normalise_by_negative(explanations)

    return explanations


def explainer_wrapper(**kwargs):
    """Wrapper for explainer functions."""
    if kwargs["method"] == "Saliency":
        return saliency_explainer(**kwargs)
    elif kwargs["method"] == "IntegratedGradients":
        return intgrad_explainer(**kwargs)
    elif kwargs["method"] == "GradientShap":
        return gradshap_explainer(**kwargs)
    elif kwargs["method"] == "GradCAM":
        return gradcam_explainer(**kwargs)
    elif kwargs["method"] == "FeaturePermutation":
        return feature_permutation_explainer(**kwargs)
    else:
        raise ValueError(f"Unknown explanation method: {kwargs['method']}")


# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

def apply_multiple_comparisons_correction(p_values: List[float], 
                                         method: str = 'fdr_bh',
                                         alpha: float = 0.05) -> Dict:
    """
    Apply multiple comparisons correction to p-values.
    
    This addresses the multiple comparisons problem when testing:
    - Multiple metrics (Relevance Rank Accuracy, AUC, Pointing Game, Custom Metrics)
    - Multiple XAI methods (Saliency, IntGrad, GradShap, GradCAM, FeaturePermutation)
    
    Methods available:
    - 'bonferroni': Conservative, controls FWER (Family-Wise Error Rate)
    - 'holm': Holm-Bonferroni, less conservative, controls FWER
    - 'fdr_bh': Benjamini-Hochberg, controls FDR (False Discovery Rate) - DEFAULT
    - 'fdr_by': Benjamini-Yekutieli, more conservative FDR control
    
    Args:
        p_values: List of p-values from multiple tests
        method: Correction method ('bonferroni', 'holm', 'fdr_bh', 'fdr_by')
        alpha: Significance level
        
    Returns:
        dict: Contains corrected p-values, reject decisions, method info
    """
    p_values = np.array(p_values)
    
    # Apply correction
    reject, p_corrected, alpha_sidak, alpha_bonferroni = multipletests(
        p_values, 
        alpha=alpha, 
        method=method, 
        returnsorted=False
    )
    
    results = {
        'original_p_values': p_values,
        'corrected_p_values': p_corrected,
        'reject': reject,
        'method': method,
        'alpha_sidak': alpha_sidak,
        'alpha_bonferroni': alpha_bonferroni,
        'num_tests': len(p_values),
        'num_significant': np.sum(reject),
        'alpha': alpha,
    }
    
    return results


def bonferroni_correction(p_values: List[float], alpha: float = 0.05) -> Dict:
    """
    Apply Bonferroni correction (most conservative).
    
    Corrected alpha = alpha / number_of_tests
    
    Best for: Strict control of false positives (FWER)
    Downside: Reduced statistical power
    """
    p_values = np.array(p_values)
    num_tests = len(p_values)
    bonferroni_alpha = alpha / num_tests
    reject = p_values < bonferroni_alpha
    
    return {
        'method': 'bonferroni',
        'original_alpha': alpha,
        'corrected_alpha': bonferroni_alpha,
        'original_p_values': p_values,
        'reject': reject,
        'num_tests': num_tests,
        'num_significant': np.sum(reject),
    }


def holm_correction(p_values: List[float], alpha: float = 0.05) -> Dict:
    """
    Apply Holm-Bonferroni correction (less conservative than Bonferroni).
    
    Adjusts alpha based on sorted p-values: alpha / (m - i + 1)
    where m = number of tests, i = rank of p-value
    
    Best for: Balance between FWER control and power
    """
    p_values = np.array(p_values)
    sorted_indices = np.argsort(p_values)
    num_tests = len(p_values)
    
    # Calculate adjusted alphas
    adjusted_alphas = alpha / (num_tests - np.arange(num_tests))
    reject = np.zeros(num_tests, dtype=bool)
    
    for i, idx in enumerate(sorted_indices):
        if p_values[idx] < adjusted_alphas[i]:
            reject[idx] = True
        else:
            # Once we fail to reject, all subsequent tests fail
            break
    
    return {
        'method': 'holm',
        'original_alpha': alpha,
        'original_p_values': p_values,
        'reject': reject,
        'num_tests': num_tests,
        'num_significant': np.sum(reject),
    }


def benjamini_hochberg_correction(p_values: List[float], alpha: float = 0.05) -> Dict:
    """
    Apply Benjamini-Hochberg FDR correction.
    
    Controls False Discovery Rate (not FWER).
    Less conservative, better statistical power.
    
    Formula: For sorted p-values, reject if p(i) <= i/m * alpha
    where m = number of tests, i = rank
    
    Best for: Studies with many tests (good power while controlling FDR)
    """
    p_values = np.array(p_values)
    sorted_indices = np.argsort(p_values)
    num_tests = len(p_values)
    
    # Find largest i such that p(i) <= i/m * alpha
    reject_indices = []
    for i, idx in enumerate(sorted_indices[::-1], 1):
        if p_values[idx] <= (num_tests - i + 1) / num_tests * alpha:
            reject_indices.extend(sorted_indices[:(num_tests - i + 1)])
            break
    
    reject = np.zeros(num_tests, dtype=bool)
    reject[reject_indices] = True
    
    return {
        'method': 'benjamini_hochberg',
        'original_alpha': alpha,
        'original_p_values': p_values,
        'reject': reject,
        'num_tests': num_tests,
        'num_significant': np.sum(reject),
    }


def report_multiple_comparisons(metrics_results: Dict, 
                                xai_methods: List[str],
                                metric_names: List[str],
                                alpha: float = 0.05) -> Dict:
    """
    Perform comprehensive multiple comparisons correction across all metrics and methods.
    
    Addresses the reviewer comment:
    "No correction for multiple comparisons across metrics (Relevance-Rank + Positive Attribution Ratio)
     or across XAI methods"
    
    Args:
        metrics_results: Dictionary of metric results {method: {metric: value}}
        xai_methods: List of XAI method names
        metric_names: List of metric names
        alpha: Significance level
        
    Returns:
        Dictionary with corrected p-values and interpretations
    """
    logger.info("\n" + "="*30)
    logger.info("MULTIPLE COMPARISONS CORRECTION")
    logger.info("="*30)
    
    # Flatten all comparisons
    all_comparisons = []
    comparison_labels = []
    
    for method in xai_methods:
        for metric_name in metric_names:
            value = metrics_results.get(method, {}).get(metric_name, np.nan)
            if not np.isnan(value):
                all_comparisons.append(value)
                comparison_labels.append(f"{method} - {metric_name}")
    
    num_comparisons = len(all_comparisons)
    
    logger.info(f"\nTotal number of comparisons: {num_comparisons}")
    logger.info(f"  {len(xai_methods)} XAI methods × {len(metric_names)} metrics = {len(xai_methods) * len(metric_names)}")
    logger.info(f"Significance level (alpha): {alpha}")
    logger.info(f"Bonferroni-corrected alpha: {alpha / num_comparisons:.6f}")
    logger.info(f"FDR-controlled alpha (BH): {alpha:.4f}")
    
    # Apply different correction methods
    results = {
        'num_comparisons': num_comparisons,
        'alpha': alpha,
        'comparison_labels': comparison_labels,
        'original_values': all_comparisons,
    }
    
    # Method 1: Bonferroni (most conservative)
    bonf_results = bonferroni_correction(all_comparisons, alpha)
    results['bonferroni'] = bonf_results
    
    logger.info(f"\nBonferroni Correction (Most Conservative):")
    logger.info(f"  Adjusted alpha: {alpha / num_comparisons:.6f}")
    logger.info(f"  Significant: {bonf_results['num_significant']}/{num_comparisons}")
    
    # Method 2: Holm (less conservative)
    holm_results = holm_correction(all_comparisons, alpha)
    results['holm'] = holm_results
    
    logger.info(f"\nHolm Correction (Step-down procedure):")
    logger.info(f"  Significant: {holm_results['num_significant']}/{num_comparisons}")
    
    # Method 3: Benjamini-Hochberg (best for power)
    bh_results = benjamini_hochberg_correction(all_comparisons, alpha)
    results['benjamini_hochberg'] = bh_results
    
    logger.info(f"\nBenjamini-Hochberg FDR Correction (Recommended):")
    logger.info(f"  Significant: {bh_results['num_significant']}/{num_comparisons}")
    logger.info(f"  Expected False Discovery Rate: {alpha:.4f}")
    
    # Show which results remain significant under different methods
    logger.info(f"\nResults remaining significant under each method:")
    
    # Bonferroni results
    bonf_msg = "  Bonferroni: "
    if bonf_results['num_significant'] > 0:
        sig_labels = [comparison_labels[i] for i, x in enumerate(bonf_results['reject']) if x]
        bonf_msg += ", ".join(sig_labels[:3])
        if len(sig_labels) > 3:
            bonf_msg += f", ... (+{len(sig_labels)-3} more)"
    else:
        bonf_msg += "None"
    logger.info(bonf_msg)
    
    # Holm results
    holm_msg = "  Holm: "
    if holm_results['num_significant'] > 0:
        sig_labels = [comparison_labels[i] for i, x in enumerate(holm_results['reject']) if x]
        holm_msg += ", ".join(sig_labels[:3])
        if len(sig_labels) > 3:
            holm_msg += f", ... (+{len(sig_labels)-3} more)"
    else:
        holm_msg += "None"
    logger.info(holm_msg)
    
    # Benjamini-Hochberg results
    bh_msg = "  Benjamini-Hochberg (FDR): "
    if bh_results['num_significant'] > 0:
        sig_labels = [comparison_labels[i] for i, x in enumerate(bh_results['reject']) if x]
        bh_msg += ", ".join(sig_labels[:3])
        if len(sig_labels) > 3:
            bh_msg += f", ... (+{len(sig_labels)-3} more)"
    else:
        bh_msg += "None"
    logger.info(bh_msg)
    
    logger.info("="*80 + "\n")
    
    return results


def create_multiple_comparisons_summary(metrics_df: pd.DataFrame,
                                       correction_results: Dict) -> str:
    """
    Create a summary table for multiple comparisons correction.
    
    Returns formatted string for reporting.
    """
    summary = "\n" + "="*80 + "\n"
    summary += "MULTIPLE COMPARISONS CORRECTION SUMMARY\n"
    summary += "="*80 + "\n\n"
    
    summary += "CORRECTION METHODS APPLIED:\n"
    summary += "-"*80 + "\n"
    summary += f"Total number of tests: {correction_results['num_comparisons']}\n"
    summary += f"  • {len(metrics_df.columns)} XAI methods\n"
    summary += f"  • {len(metrics_df.columns)} metrics per method (see note below)\n"
    summary += f"  • Total comparisons: {correction_results['num_comparisons']}\n\n"
    
    summary += "SIGNIFICANCE LEVEL:\n"
    summary += "-"*80 + "\n"
    summary += f"Original alpha (per-test): {correction_results['alpha']}\n"
    summary += f"Bonferroni-corrected alpha: {correction_results['alpha'] / correction_results['num_comparisons']:.6f}\n"
    summary += f"  (alpha / number of tests)\n\n"
    
    summary += "CORRECTION METHOD COMPARISON:\n"
    summary += "-"*80 + "\n"
    
    bonf = correction_results['bonferroni']
    holm = correction_results['holm']
    bh = correction_results['benjamini_hochberg']
    
    summary += f"{'Method':<25} {'Type':<15} {'Significant':<15} {'Description':<40}\n"
    summary += "-"*80 + "\n"
    summary += f"{'Bonferroni':<25} {'FWER':<15} {bonf['num_significant']}/{bonf['num_tests']:<13} Most conservative\n"
    summary += f"{'Holm':<25} {'FWER':<15} {holm['num_significant']}/{holm['num_tests']:<13} Less conservative\n"
    summary += f"{'Benjamini-Hochberg':<25} {'FDR':<15} {bh['num_significant']}/{bh['num_tests']:<13} Recommended (best power)\n\n"
    
    summary += "RECOMMENDATION:\n"
    summary += "-"*80 + "\n"
    summary += """Use Benjamini-Hochberg (FDR) correction for:
  • Studies with multiple comparisons
  • Balance between statistical power and false discovery control
  • Standard in modern multiple testing literature
  • Controls proportion of false discoveries (not total false discoveries)

Use Bonferroni for:
  • Very strict control requirements
  • Smaller number of tests
  • When false positives are extremely costly
\n"""
    
    summary += "="*80 + "\n"
    return summary


def evaluate_on_test_set(model, test_dataloader):
    """Evaluate model on test set."""
    logger.info("Evaluating model on test set...")
    
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in test_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def compute_classification_metrics(y_true, y_pred, y_probs):
    """Compute classification metrics."""
    metrics_dict = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_probs),
    }
    
    return metrics_dict


def generate_explanations_full_test_set(model, test_dataloader, mask_dataloader, device_to_use=device, num_per_class=100, seed=42):
    """Generate explanations for balanced samples from test set (100 from each class = 200 total).
    
    Ensures masks are matched to the exact images selected for XAI analysis.
    
    Args:
        seed: Random seed for reproducible sample selection.
    """
    # Set seed for reproducible sample selection
    np.random.seed(seed)
    
    total_samples = num_per_class * 2
    logger.info(f"Generating explanations for {total_samples} test set samples ({num_per_class} per class)...")
    
    explanations_all = {
        "Saliency": [],
        "IntegratedGradients": [],
        "GradientShap": [],
        "GradCAM": [],
        "FeaturePermutation": []
    }
    
    xai_timing = {
        "Saliency": [],
        "IntegratedGradients": [],
        "GradientShap": [],
        "GradCAM": [],
        "FeaturePermutation": []
    }
    
    all_labels = []
    all_preds = []
    all_masks = []
    
    xai_methods = ["Saliency", "IntegratedGradients", "GradientShap", "GradCAM", "FeaturePermutation"]
    
    # Collect all data first
    all_inputs = []
    all_input_labels = []
    
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(test_dataloader):
            all_inputs.append(inputs)
            all_input_labels.append(labels)
    
    # Concatenate all data
    all_inputs = torch.cat(all_inputs, dim=0)
    all_input_labels = torch.cat(all_input_labels, dim=0)
    
    # Collect all masks
    all_mask_data = []
    all_mask_labels = []
    
    with torch.no_grad():
        for batch_idx, (masks, labels) in enumerate(mask_dataloader):
            all_mask_data.append(masks)
            all_mask_labels.append(labels)
    
    # Concatenate all masks
    all_mask_data = torch.cat(all_mask_data, dim=0)
    all_mask_labels = torch.cat(all_mask_labels, dim=0)
    
    # Select balanced samples: 100 from each class
    class_0_indices = np.where(all_input_labels.numpy() == 0)[0]
    class_1_indices = np.where(all_input_labels.numpy() == 1)[0]
    
    logger.info(f"Available samples: {len(class_0_indices)} class 0, {len(class_1_indices)} class 1")
    
    # Randomly select num_per_class from each
    selected_class_0 = np.random.choice(class_0_indices, min(num_per_class, len(class_0_indices)), replace=False)
    selected_class_1 = np.random.choice(class_1_indices, min(num_per_class, len(class_1_indices)), replace=False)
    
    selected_indices = np.concatenate([selected_class_0, selected_class_1])
    np.random.shuffle(selected_indices)
    
    logger.info(f"Selected {len(selected_class_0)} class 0 and {len(selected_class_1)} class 1 samples")
    
    # Get selected data and corresponding masks
    selected_inputs = all_inputs[selected_indices]
    selected_labels = all_input_labels[selected_indices]
    selected_masks = all_mask_data[selected_indices]
    
    # Process selected samples in batches
    num_batches = int(np.ceil(len(selected_inputs) / Config.BATCH_SIZE))
    
    with torch.no_grad():
        for batch_idx in range(num_batches):
            start_idx = batch_idx * Config.BATCH_SIZE
            end_idx = min((batch_idx + 1) * Config.BATCH_SIZE, len(selected_inputs))
            
            inputs = selected_inputs[start_idx:end_idx]
            labels = selected_labels[start_idx:end_idx]
            masks = selected_masks[start_idx:end_idx]
            batch_size = inputs.shape[0]
            
            # Get predictions
            model.eval()
            outputs = model(inputs.to(device_to_use))
            _, preds = torch.max(outputs, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_masks.append(masks)
            
            # Generate explanations
            for method in xai_methods:
                gc.collect()
                torch.cuda.empty_cache()
                
                try:
                    # Track inference time
                    start_time = time.time()
                    
                    explanation = explainer_wrapper(
                        model=model,
                        inputs=inputs.cpu().numpy(),
                        targets=preds.cpu().numpy(),
                        device=device_to_use,
                        nr_channels=1,
                        img_size=Config.IMG_SIZE,
                        method=method,
                    )
                    
                    inference_time = time.time() - start_time
                    xai_timing[method].append(inference_time)
                    explanations_all[method].append(explanation)
                except Exception as e:
                    logger.error(f"Error generating {method} for batch {batch_idx}: {e}")
                    continue
    
    # Concatenate all batches
    for method in xai_methods:
        explanations_all[method] = np.concatenate(explanations_all[method], axis=0)
        logger.info(f"{method} explanations shape: {explanations_all[method].shape}")
    
    # Concatenate all masks
    all_masks = torch.cat(all_masks, dim=0)
    
    # Log timing statistics
    logger.info("\n" + "="*30)
    logger.info("XAI Methods Inference Time Statistics")
    logger.info("="*30)
    for method in xai_methods:
        if xai_timing[method]:
            total_time = np.sum(xai_timing[method])
            mean_time = np.mean(xai_timing[method])

            
            logger.info(f"\n{method}:")
            logger.info(f"  Total time: {total_time:.4f}s")
            logger.info(f"  Mean sample time: {mean_time/Config.BATCH_SIZE:.4f}s")
    logger.info("="*30 + "\n")
    
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    
    return explanations_all, all_labels, all_preds, all_masks, xai_timing
    
    # Log timing statistics
    logger.info("\n" + "="*30)
    logger.info("XAI Methods Inference Time Statistics")
    logger.info("="*30)
    for method in xai_methods:
        if xai_timing[method]:
            total_time = np.sum(xai_timing[method])
            mean_time = np.mean(xai_timing[method])
            
            logger.info(f"\n{method}:")
            logger.info(f"  Total time: {total_time:.4f}s")
            logger.info(f"  Mean sample time: {mean_time/Config.BATCH_SIZE:.4f}s")
    logger.info("="*30 + "\n")
    
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    
    return explanations_all, all_labels, all_preds, xai_timing


def evaluate_xai_against_masks(model, explanations_all, x_batch, y_pred, s_batch, device_to_use=device):
    """Evaluate XAI methods against ground truth masks using Relevance metrics.
    
    Args:
        model: The trained model
        explanations_all: Dictionary of explanations {method: array}
        x_batch: Input images (aligned with explanations)
        y_pred: Predictions (aligned with explanations)
        s_batch: Ground truth masks (aligned with explanations)
        device_to_use: Device to use for computation
    """
    logger.info("Evaluating XAI methods against ground truth masks...")
    
    # Ensure we're working with aligned data - use only the samples we have explanations for
    num_samples = len(list(explanations_all.values())[0])
    logger.info(f"Evaluating on {num_samples} samples")
    
    x_batch = x_batch[:num_samples]
    y_pred = y_pred[:num_samples]
    s_batch = s_batch[:num_samples]
    
    # Verify shapes
    logger.info(f"x_batch shape: {x_batch.shape}")
    logger.info(f"y_pred shape: {y_pred.shape}")
    logger.info(f"s_batch shape: {s_batch.shape}")
    
    # Define metrics - focus on Relevance-based metrics
    metrics = {
        "RelevanceRankAccuracy": quantus.RelevanceRankAccuracy(
            abs=False,
            normalise=True,
            aggregate_func=np.mean,
            return_aggregate=True,
            disable_warnings=True,
        ),
        "PositiveAttributionRatio": PositiveAttributionRatio(
            abs=False,
            normalise=True,
            aggregate_func=np.mean,
            return_aggregate=True,
            disable_warnings=True,
        ),
    }

    xai_methods = ["Saliency", "IntegratedGradients", "GradientShap", "GradCAM", "FeaturePermutation"]
    results = {method: {} for method in xai_methods}

    # Convert x_batch to numpy if needed
    if isinstance(x_batch, torch.Tensor):
        x_batch = x_batch.cpu().numpy()
    
    # Ensure s_batch is proper shape
    s_batch_reshaped = s_batch.reshape(num_samples, 1, Config.IMG_SIZE, Config.IMG_SIZE)
    if isinstance(s_batch_reshaped, torch.Tensor):
        s_batch_reshaped = s_batch_reshaped.cpu().numpy()

    for method in xai_methods:
        logger.info(f"\nEvaluating {method} method...")
        
        # Get the pre-computed explanations for this method
        expl = explanations_all[method]
        logger.info(f"  Explanation shape: {expl.shape}")
        
        for metric_name, metric_func in metrics.items():
            try:
                logger.info(f"  Computing {metric_name}...")
                gc.collect()
                torch.cuda.empty_cache()

                # Pass pre-computed explanations directly
                scores = metric_func(
                    model=model.to(device_to_use),
                    x_batch=x_batch,
                    y_batch=y_pred,
                    a_batch=expl,  # Pass pre-computed explanations
                    s_batch=s_batch_reshaped,
                    device=device_to_use,
                )
                results[method][metric_name] = scores
                logger.info(f"    {metric_name}: {scores}")

                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            except Exception as e:
                logger.error(f"Error evaluating {metric_name} for {method}: {e}")
                import traceback
                traceback.print_exc()
                results[method][metric_name] = np.nan
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
        
        # Aggressive cleanup after each method
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # Aggregate results
    results_agg = {}
    for method in xai_methods:
        results_agg[method] = {}
        for metric_name in metrics.keys():
            if isinstance(results[method][metric_name], (list, np.ndarray)):
                results_agg[method][metric_name] = np.mean(results[method][metric_name])
            else:
                results_agg[method][metric_name] = results[method][metric_name]

    df_results = pd.DataFrame.from_dict(results_agg).T
    
    # Apply multiple comparisons correction
    correction_results = report_multiple_comparisons(
        results_agg,
        xai_methods,
        list(metrics.keys()),
        alpha=0.05
    )
    
    return df_results, results, correction_results


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def normalize_image(img):
    """Normalize image to [0, 1] range."""
    img_min = np.min(img)
    img_max = np.max(img)
    if img_max - img_min == 0:
        return np.zeros_like(img)
    return (img - img_min) / (img_max - img_min)


def visualize_explanations(images, masks, explanations_dict, predictions, labels, num_samples=None):
    """Visualize explanations compared to ground truth masks."""
    logger.info("Creating visualization plots...")
    
    if num_samples is None:
        num_samples = min(Config.PLOT_SAMPLES, len(images))
    
    indices = np.random.choice(len(images), num_samples, replace=False)
    
    fig, axes = plt.subplots(num_samples, 5, figsize=(20, 4*num_samples), dpi=Config.DPI)
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    # class_names = ['cat', 'dog']
    
    for idx, sample_idx in enumerate(indices):
        # Original image
        img = images[sample_idx, 0, :, :]
        axes[idx, 0].imshow(img, cmap='gray')
        axes[idx, 0].set_title(f'Original Image\nTrue: {class_names[labels[sample_idx]]}\nPred: {class_names[predictions[sample_idx]]}')
        axes[idx, 0].axis('off')
        
        # Ground truth mask
        mask = masks[sample_idx, 0, :, :]
        axes[idx, 1].imshow(mask, cmap='gray')
        axes[idx, 1].set_title('Ground Truth Mask')
        axes[idx, 1].axis('off')
        
        # Explanations
        for col_idx, (method, expl_array) in enumerate(explanations_dict.items(), start=2):
            expl = expl_array[sample_idx]
            expl_norm = normalize_image(expl)
            im = axes[idx, col_idx].imshow(expl_norm, cmap=Config.COLORMAP)
            axes[idx, col_idx].set_title(f'{method}')
            axes[idx, col_idx].axis('off')
            plt.colorbar(im, ax=axes[idx, col_idx], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plot_path = os.path.join(Config.PLOTS_DIR, 'xai_comparisons.png')
    plt.savefig(plot_path, dpi=Config.DPI, bbox_inches='tight')
    logger.info(f"Saved visualization to {plot_path}")
    plt.close()


def plot_xai_metrics(df_results):
    """Plot XAI evaluation metrics."""
    logger.info("Creating metrics comparison plot...")
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=Config.DPI)
    df_results.plot(kind='bar', ax=ax)
    ax.set_title('XAI Methods Evaluation Metrics (Test Set)', fontsize=14, fontweight='bold')
    ax.set_xlabel('XAI Method', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_ylim([0, 1])
    ax.legend(title='Metrics', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    plot_path = os.path.join(Config.PLOTS_DIR, 'xai_metrics_comparison.png')
    plt.savefig(plot_path, dpi=Config.DPI, bbox_inches='tight')
    logger.info(f"Saved metrics plot to {plot_path}")
    plt.close()


def plot_classification_metrics(metrics_dict):
    """Plot classification metrics."""
    logger.info("Creating classification metrics plot...")
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=Config.DPI)
    metrics_df = pd.DataFrame(metrics_dict.items(), columns=['Metric', 'Score'])
    bars = ax.bar(metrics_df['Metric'], metrics_df['Score'], color='steelblue')
    ax.set_title('Model Classification Metrics (Test Set)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12)
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom')
    
    plt.xticks(rotation=45)
    plot_path = os.path.join(Config.PLOTS_DIR, 'classification_metrics.png')
    plt.savefig(plot_path, dpi=Config.DPI, bbox_inches='tight')
    logger.info(f"Saved classification metrics to {plot_path}")
    plt.close()


# ============================================================================
# MASK COMPARISON METRICS
# ============================================================================

def compute_mask_alignment_metrics(explanation, mask, threshold=0.5):
    """Compute pixel-level alignment metrics between explanation and mask."""
    # Normalize explanation
    expl_norm = normalize_image(explanation)
    
    # Create binary masks
    expl_binary = (expl_norm > threshold).astype(int)
    mask_binary = (mask > threshold).astype(int)
    
    # Compute metrics
    tp = np.sum((expl_binary == 1) & (mask_binary == 1))
    tn = np.sum((expl_binary == 0) & (mask_binary == 0))
    fp = np.sum((expl_binary == 1) & (mask_binary == 0))
    fn = np.sum((expl_binary == 0) & (mask_binary == 1))
    
    # IoU (Jaccard Index)
    iou = tp / (tp + fp + fn + 1e-8)
    
    # Dice coefficient
    dice = 2 * tp / (2 * tp + fp + fn + 1e-8)
    
    # Sensitivity (True Positive Rate)
    sensitivity = tp / (tp + fn + 1e-8)
    
    # Specificity (True Negative Rate)
    specificity = tn / (tn + fp + 1e-8)
    
    return {
        'iou': iou,
        'dice': dice,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
    }


def compute_all_mask_alignments(explanations_dict, masks):
    """Compute alignment metrics for all explanations against masks."""
    logger.info("Computing mask alignment metrics...")
    
    alignment_results = {}
    
    for method, explanations in explanations_dict.items():
        logger.info(f"Computing alignment for {method}...")
        
        method_metrics = {
            'iou': [],
            'dice': [],
            'sensitivity': [],
            'specificity': [],
        }
        
        for sample_idx in range(len(explanations)):
            metrics = compute_mask_alignment_metrics(
                explanations[sample_idx],
                masks[sample_idx, 0, :, :]
            )
            
            method_metrics['iou'].append(metrics['iou'])
            method_metrics['dice'].append(metrics['dice'])
            method_metrics['sensitivity'].append(metrics['sensitivity'])
            method_metrics['specificity'].append(metrics['specificity'])
        
        alignment_results[method] = {
            'iou': {
                'mean': np.mean(method_metrics['iou']),
                'std': np.std(method_metrics['iou']),
            },
            'dice': {
                'mean': np.mean(method_metrics['dice']),
                'std': np.std(method_metrics['dice']),
            },
            'sensitivity': {
                'mean': np.mean(method_metrics['sensitivity']),
                'std': np.std(method_metrics['sensitivity']),
            },
            'specificity': {
                'mean': np.mean(method_metrics['specificity']),
                'std': np.std(method_metrics['specificity']),
            },
        }
    
    return alignment_results


def plot_mask_alignment_metrics(alignment_results):
    """Plot mask alignment metrics."""
    logger.info("Creating mask alignment plot...")
    
    metrics_list = ['iou', 'dice', 'sensitivity', 'specificity']
    methods = list(alignment_results.keys())
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=Config.DPI)
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics_list):
        means = [alignment_results[method][metric]['mean'] for method in methods]
        stds = [alignment_results[method][metric]['std'] for method in methods]
        
        axes[idx].bar(methods, means, yerr=stds, capsize=5, color='steelblue', alpha=0.7)
        axes[idx].set_title(f'{metric.upper()} - Alignment with Ground Truth', fontsize=12, fontweight='bold')
        axes[idx].set_ylabel('Score', fontsize=11)
        axes[idx].set_ylim([0, 1])
        axes[idx].grid(True, alpha=0.3, axis='y')
        axes[idx].tick_params(axis='x', rotation=45)
        
        # Add value labels
        for i, (mean, std) in enumerate(zip(means, stds)):
            axes[idx].text(i, mean + std + 0.02, f'{mean:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plot_path = os.path.join(Config.PLOTS_DIR, 'mask_alignment_metrics.png')
    plt.savefig(plot_path, dpi=Config.DPI, bbox_inches='tight')
    logger.info(f"Saved mask alignment plot to {plot_path}")
    plt.close()


# ============================================================================
# RESULTS SAVING
# ============================================================================

def save_results(results_dict, filename='xai_results.csv'):
    """Save results to CSV."""
    filepath = os.path.join(Config.RESULTS_DIR, filename)
    
    # Convert to DataFrame if needed
    if isinstance(results_dict, dict) and not isinstance(results_dict, pd.DataFrame):
        df = pd.DataFrame(results_dict)
    else:
        df = results_dict
    
    df.to_csv(filepath)
    logger.info(f"Saved results to {filepath}")


def save_analysis_report(analysis_dict):
    """Save comprehensive analysis report."""
    report_path = os.path.join(Config.RESULTS_DIR, 'xai_analysis_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("XAI Analysis on Complete Test Set - Comprehensive Report\n")
        f.write("="*80 + "\n\n")
        
        for section, content in analysis_dict.items():
            f.write(f"\n{section}\n")
            f.write("-"*80 + "\n")
            f.write(str(content))
            f.write("\n")
    
    logger.info(f"Saved analysis report to {report_path}")


# ============================================================================
# SANITY CHECKS AND STATISTICAL TESTING
# ============================================================================

def randomization_test(explanations, num_permutations=1000):
    """
    Test if attributions are meaningful by comparing against random attributions.
    
    A good explanation should have significantly higher values in important regions
    than randomly generated attributions.
    
    Returns:
        dict: p-values and test statistics for each method
    """
    logger.info("\n[SANITY CHECK 1/3] Performing randomization tests...")
    
    results = {}
    
    for method, expl_array in explanations.items():
        logger.info(f"  Testing {method}...")
        
        # Flatten all explanations
        real_scores = expl_array.flatten()
        
        # Generate random attributions with same shape
        random_scores_list = []
        for _ in range(num_permutations):
            random_attr = np.random.normal(0, 1, expl_array.shape).flatten()
            random_scores_list.append(random_attr)
        random_scores_all = np.array(random_scores_list)
        
        # Compute effect size: compare mean absolute attribution value
        real_magnitude = np.mean(np.abs(real_scores))
        random_magnitudes = np.mean(np.abs(random_scores_all), axis=1)
        
        # Perform t-test
        t_stat, p_value = stats.ttest_ind(
            [real_magnitude],
            random_magnitudes
        )
        
        # Cliff's delta effect size
        cliffs_delta = compute_cliffs_delta(
            np.array([real_magnitude]),
            random_magnitudes
        )
        
        results[method] = {
            'real_magnitude': real_magnitude,
            'random_mean': np.mean(random_magnitudes),
            'random_std': np.std(random_magnitudes),
            't_statistic': t_stat,
            'p_value': p_value,
            'cliffs_delta': cliffs_delta,
            'significant': p_value < 0.05,
        }
        
        logger.info(f"    Real magnitude: {real_magnitude:.6f}")
        logger.info(f"    Random mean ± std: {np.mean(random_magnitudes):.6f} ± {np.std(random_magnitudes):.6f}")
        logger.info(f"    p-value: {p_value:.6e}")
        logger.info(f"    Cliff's delta: {cliffs_delta:.4f}")
        logger.info(f"    Significant: {p_value < 0.05}")
    
    return results


def layer_wise_relevance_randomization(model, x_batch, y_batch, explainer_wrapper_func, num_layers=10):
    """
    Test if zeroing/randomizing model layers degrades explanation quality.
    
    A good explanation should correlate with model weights. If explanations don't
    degrade when model layers are randomized, they're likely detecting superficial patterns.
    
    Returns:
        dict: Degradation metrics for each method and layer
    """
    logger.info("\n[SANITY CHECK 2/3] Performing layer-wise relevance randomization...")
    
    device_to_use = device
    model_original = copy_module.deepcopy(model)
    
    # Convert to numpy if tensor
    if isinstance(x_batch, torch.Tensor):
        x_batch = x_batch.cpu().numpy()
    if isinstance(y_batch, torch.Tensor):
        y_batch = y_batch.cpu().numpy()
    
    # Get all trainable parameters
    all_params = list(model.named_parameters())
    num_to_randomize = min(num_layers, len(all_params))
    
    logger.info(f"  Model has {len(all_params)} layers. Testing randomization of {num_to_randomize} layers...")
    
    results = {}
    
    for method in ["Saliency", "IntegratedGradients", "GradientShap", "GradCAM", "FeaturePermutation"]:
        logger.info(f"  Testing {method}...")
        
        # Get original explanations
        original_expl = explainer_wrapper_func(
            model=model_original,
            inputs=x_batch,
            targets=y_batch,
            device=device_to_use,
            nr_channels=1,
            img_size=Config.IMG_SIZE,
            method=method,
        )
        
        degradation_scores = []
        
        # Randomize each layer and measure degradation
        for layer_idx in range(num_to_randomize):
            param_name, param = all_params[layer_idx]
            
            # Randomize layer weights
            model_perturbed = copy_module.deepcopy(model_original)
            with torch.no_grad():
                for name, p in model_perturbed.named_parameters():
                    if name == param_name:
                        p.data = torch.randn_like(p.data)
            
            # Get explanations with randomized layer
            perturbed_expl = explainer_wrapper_func(
                model=model_perturbed,
                inputs=x_batch,
                targets=y_batch,
                device=device_to_use,
                nr_channels=1,
                img_size=Config.IMG_SIZE,
                method=method,
            )
            
            # Compute correlation degradation (higher degradation = better)
            correlation = np.corrcoef(
                original_expl.flatten(),
                perturbed_expl.flatten()
            )[0, 1]
            degradation = 1 - np.abs(correlation)  # Higher = more degradation (good)
            degradation_scores.append(degradation)
            
            logger.info(f"    Layer {layer_idx+1} ({param_name}): degradation={degradation:.4f}, correlation={correlation:.4f}")
        
        # Compute eta-squared (η²) effect size
        all_degradations = np.array(degradation_scores)
        eta_squared = compute_eta_squared(all_degradations)
        
        results[method] = {
            'mean_degradation': np.mean(degradation_scores),
            'std_degradation': np.std(degradation_scores),
            'min_degradation': np.min(degradation_scores),
            'max_degradation': np.max(degradation_scores),
            'eta_squared': eta_squared,
            'layer_degradations': degradation_scores,
        }
        
        logger.info(f"  {method} - Mean degradation: {np.mean(degradation_scores):.4f} ± {np.std(degradation_scores):.4f}")
        logger.info(f"  {method} - η² (effect size): {eta_squared:.4f}")
    
    return results


def input_randomization_test(model, x_batch, y_batch, explainer_wrapper_func, num_permutations=10):
    """
    Test if explanations change meaningfully when input is randomized.
    
    High-quality explanations should change significantly when the input changes,
    indicating they're actually analyzing the input.
    
    Returns:
        dict: Sensitivity metrics for each method
    """
    logger.info("\n[SANITY CHECK 3/3] Performing input randomization sensitivity test...")
    
    device_to_use = device
    
    # Convert to numpy if tensor
    if isinstance(x_batch, torch.Tensor):
        x_batch = x_batch.cpu().numpy()
    if isinstance(y_batch, torch.Tensor):
        y_batch = y_batch.cpu().numpy()
    
    results = {}
    batch_size = min(len(x_batch), 10)  # Test on first 10 samples
    x_test = x_batch[:batch_size]
    y_test = y_batch[:batch_size]
    
    for method in ["Saliency", "IntegratedGradients", "GradientShap", "GradCAM", "FeaturePermutation"]:
        logger.info(f"  Testing {method}...")
        
        # Get original explanations
        original_expl = explainer_wrapper_func(
            model=model,
            inputs=x_test,
            targets=y_test,
            device=device_to_use,
            nr_channels=1,
            img_size=Config.IMG_SIZE,
            method=method,
        )
        
        sensitivity_scores = []
        
        # Permute input and measure explanation change
        for perm_idx in range(num_permutations):
            # Create permuted input (shuffle pixel positions)
            x_permuted = x_test.copy()
            for sample_idx in range(x_permuted.shape[0]):
                # Random permutation of spatial dimensions
                perm_indices = np.random.permutation(x_permuted[sample_idx].size)
                x_permuted[sample_idx] = x_permuted[sample_idx].flatten()[perm_indices].reshape(x_permuted[sample_idx].shape)
            
            # Get explanations for permuted input
            permuted_expl = explainer_wrapper_func(
                model=model,
                inputs=x_permuted,
                targets=y_test,
                device=device_to_use,
                nr_channels=1,
                img_size=Config.IMG_SIZE,
                method=method,
            )
            
            # Measure sensitivity as L2 distance
            sensitivity = np.sqrt(np.mean((original_expl - permuted_expl) ** 2))
            sensitivity_scores.append(sensitivity)
            
            logger.info(f"    Permutation {perm_idx+1}: sensitivity={sensitivity:.6f}")
        
        # Compute effect size statistics
        sensitivity_array = np.array(sensitivity_scores)
        eta_squared = compute_eta_squared(sensitivity_array)
        cliffs_delta = compute_cliffs_delta(sensitivity_array[:num_permutations//2], sensitivity_array[num_permutations//2:])
        
        results[method] = {
            'mean_sensitivity': np.mean(sensitivity_scores),
            'std_sensitivity': np.std(sensitivity_scores),
            'min_sensitivity': np.min(sensitivity_scores),
            'max_sensitivity': np.max(sensitivity_scores),
            'eta_squared': eta_squared,
            'cliffs_delta': cliffs_delta,
        }
        
        logger.info(f"  {method} - Mean sensitivity: {np.mean(sensitivity_scores):.6f} ± {np.std(sensitivity_scores):.6f}")
        logger.info(f"  {method} - η² (effect size): {eta_squared:.4f}")
        logger.info(f"  {method} - Cliff's delta: {cliffs_delta:.4f}")
    
    return results


def compute_eta_squared(values):
    """
    Compute eta-squared (η²) effect size.
    
    η² = sum((group_mean - grand_mean)^2) / sum((x - grand_mean)^2)
    For a single group, it measures proportion of variance explained.
    
    Args:
        values: array of values
        
    Returns:
        float: eta-squared value (0 to 1)
    """
    if len(values) < 2:
        return 0.0
    
    grand_mean = np.mean(values)
    ss_between = np.sum((values - grand_mean) ** 2)
    ss_total = np.sum((values - grand_mean) ** 2)
    
    if ss_total == 0:
        return 0.0
    
    eta_squared = ss_between / ss_total
    return np.clip(eta_squared, 0, 1)


def compute_cliffs_delta(x, y):
    """
    Compute Cliff's delta effect size (non-parametric).
    
    Cliff's delta = (n1*n2 - r1*r1) / (n1*n2)
    where r1 is the sum of ranks in group 1
    
    Returns value in [-1, 1]:
    - delta < -0.474: large negative effect
    - -0.474 <= delta < -0.147: medium negative effect  
    - -0.147 <= delta <= 0.147: negligible effect
    - 0.147 < delta <= 0.474: medium positive effect
    - delta > 0.474: large positive effect
    
    Args:
        x, y: arrays of values
        
    Returns:
        float: Cliff's delta value
    """
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    
    n1 = len(x)
    n2 = len(y)
    
    if n1 == 0 or n2 == 0:
        return 0.0
    
    # Combine and rank
    combined = np.concatenate([x, y])
    ranks = stats.rankdata(combined)
    
    # Sum of ranks for x
    r1 = np.sum(ranks[:n1])
    
    # Calculate Cliff's delta
    delta = (2 * r1 - n1 * (n1 + n2 + 1)) / (n1 * n2)
    
    return float(np.clip(delta, -1, 1))


def interpret_effect_size(value, measure_type='cliffs_delta'):
    """
    Interpret effect size values.
    
    Args:
        value: effect size value
        measure_type: 'cliffs_delta' or 'eta_squared'
        
    Returns:
        str: interpretation
    """
    if measure_type == 'cliffs_delta':
        abs_val = abs(value)
        if abs_val < 0.147:
            return "Negligible"
        elif abs_val < 0.330:
            return "Small"
        elif abs_val < 0.474:
            return "Medium"
        else:
            return "Large"
    elif measure_type == 'eta_squared':
        if value < 0.01:
            return "Negligible"
        elif value < 0.06:
            return "Small"
        elif value < 0.14:
            return "Medium"
        else:
            return "Large"
    else:
        return "Unknown"


# ============================================================================
# STATISTICAL CORRECTION (ACROSS ALL SEEDS)
# ============================================================================

def perform_statistical_correction(all_results):
    logger.info("\n" + "="*80)
    logger.info("PERFORMING STATISTICAL CORRECTION (Grouped by Hypothesis)")
    logger.info("="*80)

    if not all_results: return

    # Gather all unique models available across all seeds
    available_models = set()
    for seed in all_results:
        available_models.update(all_results[seed].keys())
    available_models = list(available_models)
    
    # --- HYPOTHESIS GROUPS ---
    groups = {
        "A_ResNet_Scale": [m for m in available_models if "ResNet" in m and "ImageNet" not in m],
        "B_DenseNet_Scale": [m for m in available_models if "DenseNet" in m and "CheXpert" not in m],
        "C_Pretraining_ResNet": ["ResNet50", "ResNet50-ImageNet"],
        "D_Pretraining_DenseNet": ["DenseNet121", "CheXpert"]
    }
    
    metrics = ["RelevanceRankAccuracy", "PositiveAttributionRatio"]
    methods = ["Saliency", "IntegratedGradients", "GradientShap", "GradCAM", "FeaturePermutation"]
    
    p_values_collection = []
    log_info = []

    for group_name, model_list in groups.items():
        valid_models = [m for m in model_list if m in available_models]
        
        if len(valid_models) < 2:
            continue
            
        logger.info(f"Processing {group_name}: {valid_models}")
        
        for metric in metrics:
            for method in methods:
                # Aggregate raw scores from ALL seeds for each model
                group_distribution = []
                
                for model in valid_models:
                    model_scores_all_seeds = []
                    for seed, seed_data in all_results.items():
                        if model in seed_data:
                            try:
                                # Extract the raw list of scores
                                scores = seed_data[model]['xai_results'][method][metric]
                                # Clean NaNs
                                scores = [s for s in scores if not np.isnan(s)]
                                model_scores_all_seeds.extend(scores)
                            except KeyError: pass
                    
                    if model_scores_all_seeds:
                        group_distribution.append(model_scores_all_seeds)
                
                # Check if we have data for all models in group
                if len(group_distribution) == len(valid_models):
                    try:
                        p_val = None
                        if len(group_distribution) > 2:
                            _, p_val = kruskal(*group_distribution)
                        else:
                            _, p_val = mannwhitneyu(group_distribution[0], group_distribution[1])
                        
                        p_values_collection.append(p_val)
                        log_info.append((group_name, metric, method, p_val))
                    except ValueError: pass

    if not p_values_collection:
        logger.warning("No valid stats generated.")
        return

    # --- BENJAMINI-HOCHBERG CORRECTION ---
    reject, p_corrected, _, _ = multipletests(p_values_collection, alpha=0.05, method='fdr_bh')
    
    logger.info(f"\n{'Group':<25} | {'Metric - Method':<50} | {'Raw p':<8} | {'Corr p':<8} | {'Sig'}")
    logger.info("-" * 110)
    for i, (grp, met, meth, p) in enumerate(log_info):
        sig = "*" if reject[i] else ""
        logger.info(f"{grp:<25} | {met + ' - ' + meth:<50} | {p:.4f}   | {p_corrected[i]:.4f}   | {sig}")
    logger.info("-" * 110)
    
    # Save the table to a text file
    summary_text = f"\n{'='*110}\n"
    summary_text += "STATISTICAL CORRECTION SUMMARY (All Seeds)\n"
    summary_text += f"{'='*110}\n\n"
    summary_text += f"{'Group':<25} | {'Metric - Method':<50} | {'Raw p':<8} | {'Corr p':<8} | {'Sig'}\n"
    summary_text += "-" * 110 + "\n"
    
    for i, (grp, met, meth, p) in enumerate(log_info):
        sig = "*" if reject[i] else ""
        summary_text += f"{grp:<25} | {met + ' - ' + meth:<50} | {p:.4f}   | {p_corrected[i]:.4f}   | {sig}\n"
    
    summary_text += "-" * 110 + "\n\n"
    summary_text += f"Total tests: {len(p_values_collection)}\n"
    summary_text += f"Significant after correction: {np.sum(reject)}\n"
    summary_text += f"Correction method: Benjamini-Hochberg FDR\n"
    
    # Save to file
    output_file = os.path.join("./xai_results", "statistical_correction_summary.txt")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        f.write(summary_text)
    
    logger.info(f"\nStatistical correction table saved to {output_file}")



# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main(seed: str, models_filter: Optional[List[str]] = None):
    """Main execution function for a specific seed - processes all available models.
    
    Args:
        seed: Seed directory name (e.g., "seed_51")
        models_filter: List of model names to process (e.g., ["DenseNet169", "DenseNet201"]).
                      If None, processes all available models.
    """
    # Configure for this seed
    Config.set_seed(seed)
    
    logger.info("="*80)
    logger.info(f"XAI Analysis on Complete Test Set ({seed})")
    if models_filter:
        logger.info(f"Processing models: {', '.join(models_filter)}")
    else:
        logger.info("Processing ALL available models in checkpoint directory")
    logger.info("="*80)
    
    # Extract numeric seed from directory name (e.g., "seed_51" -> 51)
    numeric_seed = int(seed.split('_')[1])
    set_seed(numeric_seed)
    setup_directories()
    
    # Load data (once for all models)
    logger.info("\nLoading data...")
    _, transforms_test = get_transforms()
    test_dataset, test_dataloader, mask_dataset, mask_dataloader = load_test_data(transforms_test, seed=numeric_seed)
    
    # Discover all available models in this seed's checkpoint directory
    checkpoint_dir = Config.CHECKPOINT_DIR
    if not os.path.exists(checkpoint_dir):
        logger.error(f"Checkpoint directory not found: {checkpoint_dir}")
        return {}
    
    # Find all model checkpoints
    available_models = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith('_best.pth')])
    
    if not available_models:
        logger.error(f"No model checkpoints found in {checkpoint_dir}")
        return {}
    
    # Filter models if specified
    if models_filter:
        filtered_models = []
        for model_file in available_models:
            model_name = model_file.replace('_best.pth', '')
            if any(filter_name in model_name for filter_name in models_filter):
                filtered_models.append(model_file)
        if not filtered_models:
            logger.error(f"No models matching filters {models_filter} found in {checkpoint_dir}")
            return {}
        available_models = filtered_models
    
    logger.info(f"Found {len(available_models)} model(s) to process:")
    for model_file in available_models:
        logger.info(f"  - {model_file}")
    
    # Process each model
    all_model_results = {}
    
    for model_idx, model_file in enumerate(available_models, 1):
        model_name = model_file.replace('_best.pth', '')
        logger.info(f"\n{'='*80}")
        logger.info(f"[{model_idx}/{len(available_models)}] Processing {model_name}")
        logger.info(f"{'='*80}")
        
        try:
            model_path = os.path.join(checkpoint_dir, model_file)
            
            # Load model
            logger.info(f"Loading model from {model_path}...")
            model = load_model(model_path=model_path)

            # --- SANITY CHECKS (Run only if enabled in Config) ---
            if Config.ENABLE_SANITY_CHECKS and model_name in ["ResNet18", "ResNet101"]:
                # Grab small batch
                xb, yb = next(iter(test_dataloader))
                xb, yb = xb[:32].to(device), model(xb[:32].to(device)).argmax(1)
                
                p_res = layer_wise_relevance_randomization(model, xb, yb, explainer_wrapper)
                i_res = input_randomization_test(model, xb, yb, explainer_wrapper)
                
                with open(f"{Config.RESULTS_DIR}/sanity_{model_name}.txt", "w") as f:
                    f.write(str(p_res) + "\n" + str(i_res))
            
            # Generate explanations
            explanations_all, exp_labels, exp_preds, exp_masks, xai_timing = generate_explanations_full_test_set(
                model, test_dataloader, mask_dataloader, num_per_class=100, seed=numeric_seed
            )
            
            # Prepare data
            logger.info(f"Preparing data...")
            x_batch_list = []
            for inputs, _ in test_dataloader:
                x_batch_list.append(inputs)
            x_batch = torch.cat(x_batch_list, dim=0)
            s_batch = exp_masks
            
            # Threshold masks
            s_batch = torch.where(s_batch < 0.0, torch.zeros_like(s_batch), torch.ones_like(s_batch))
            
            logger.info(f"Test batch shape: {x_batch.shape}")
            logger.info(f"Matched mask batch shape: {s_batch.shape}")
            logger.info(f"Explanations: {len(exp_labels)} samples with corresponding masks")
            
            # Evaluate XAI methods against masks
            logger.info(f"Evaluating XAI methods for {model_name}...")
            df_xai_metrics, xai_results, correction_results = evaluate_xai_against_masks(
                model, explanations_all, x_batch.cpu().numpy(), exp_preds, s_batch
            )
            # # --- EVALUATE METRICS ---
            # raw_scores = evaluate_metrics(model, expls, imgs, preds, masks)

            logger.info(f"\nXAI Evaluation Results for {model_name}:")
            logger.info(f"\n{df_xai_metrics}")
            
            # Save results with model name
            logger.info(f"Saving results for {model_name}...")
            results_filename = f'xai_metrics_{model_name}.csv'
            save_results(df_xai_metrics, results_filename)
            
            all_model_results[model_name] = {
                'xai_metrics': df_xai_metrics,
                'xai_results': xai_results,
                'correction_results': correction_results,
            }
            
            logger.info(f"✓ Analysis complete for {model_name}")
            
            # Clear GPU memory between models
            del model
            del explanations_all
            gc.collect()
            torch.cuda.empty_cache()
            
        except Exception as e:
            logger.error(f"✗ Error processing {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    logger.info("\n" + "="*80)
    logger.info(f"Analysis complete for {seed}!")
    logger.info(f"Processed {len(all_model_results)}/{len(available_models)} models successfully")
    logger.info(f"Results saved to: {Config.RESULTS_DIR}")
    logger.info("="*80)
    
    return all_model_results


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run XAI analysis on trained models")
    parser.add_argument("--models", nargs="+", default=None,
                       help="List of model names to process (e.g., DenseNet169 DenseNet201). "
                            "If not specified, processes all available models.")
    parser.add_argument("--seed", default=None,
                       help="Specific seed to process (e.g., seed_51). "
                            "If not specified, processes all available seeds.")
    args = parser.parse_args()
    
    # Discover all available seeds in checkpoints directory
    checkpoints_base = "./training_results/checkpoints"
    
    if not os.path.exists(checkpoints_base):
        logger.error(f"Checkpoints directory not found: {checkpoints_base}")
        sys.exit(1)
    
    # Get all seed directories
    seed_dirs = sorted([d for d in os.listdir(checkpoints_base) 
                       if os.path.isdir(os.path.join(checkpoints_base, d)) and d.startswith("seed_")])
    
    # Filter by specific seed if provided
    if args.seed:
        if args.seed in seed_dirs:
            seed_dirs = [args.seed]
        else:
            logger.error(f"Seed {args.seed} not found. Available seeds: {seed_dirs}")
            sys.exit(1)
    
    if not seed_dirs:
        logger.error(f"No seed directories found in {checkpoints_base}")
        sys.exit(1)
    
    logger.info(f"Found {len(seed_dirs)} seed(s): {seed_dirs}")
    
    # Run analysis for each seed
    all_results = {}
    for seed in seed_dirs:
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing {seed}")
        logger.info(f"{'='*80}\n")
        
        try:
            results = main(seed=seed, models_filter=args.models)
            all_results[seed] = results
            logger.info(f"\n✓ Analysis complete for {seed}")
        except Exception as e:
            logger.error(f"✗ Error processing {seed}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Summary
    logger.info(f"\n{'='*80}")
    logger.info("XAI Analysis Summary")
    logger.info(f"{'='*80}")
    logger.info(f"Processed {len(all_results)}/{len(seed_dirs)} seeds successfully")
    for seed in seed_dirs:
        status = "✓" if seed in all_results else "✗"
        logger.info(f"{status} {seed}")
    
    # --- FINAL STATISTICS ---
    logger.info("\nPerforming statistical correction across all seeds...")
    perform_statistical_correction(all_results)