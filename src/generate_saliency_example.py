#!/usr/bin/env python3
"""
Generate Saliency Maps Example

This script takes one image from the COVID dataset and generates 5 saliency maps
using different XAI methods: Saliency, Integrated Gradients, GradientShap, GradCAM,
and Feature Permutation.

Author: Generated for Beyond the Black Box project
"""

import os
import sys
import gc
import warnings

import numpy as np
import matplotlib.pyplot as plt
import cv2

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

from captum.attr import Saliency, IntegratedGradients, GradientShap, LayerGradCam

# Suppress warnings
warnings.filterwarnings("ignore")

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths
DATA_DIR = "./data/covid_qu_ex/COVID-19/images"
MASK_DIR = "./data/covid_qu_ex/COVID-19/infection_masks"
CHECKPOINT_DIR = "./training_results/checkpoints/seed_14"
OUTPUT_DIR = "./xai_results"

# Image settings
IMG_SIZE = 224
NUM_CLASSES = 2


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_transforms():
    """Create data transformations."""
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])


def load_model(model_path: str) -> nn.Module:
    """Load trained model from checkpoint."""
    print(f"Loading model from {model_path}...")
    
    # Load ResNet50 architecture
    model = models.resnet50(weights=None)
    
    # Modify for grayscale input
    model.conv1 = nn.Conv2d(1, model.conv1.out_channels, 
                            kernel_size=model.conv1.kernel_size,
                            stride=model.conv1.stride, 
                            padding=model.conv1.padding, 
                            bias=model.conv1.bias is not None)
    
    # Modify for binary classification
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    return model


def get_last_conv_layer(model):
    """Get the last convolutional layer in the model (for GradCAM)."""
    if hasattr(model, 'layer4'):
        return model.layer4
    raise RuntimeError("Could not find layer4 in the model")


# ============================================================================
# XAI METHODS
# ============================================================================

def compute_saliency(model, inputs, targets):
    """Compute Saliency map."""
    saliency = Saliency(model)
    attribution = saliency.attribute(inputs, target=targets)
    return attribution.sum(dim=1).squeeze().cpu().detach().numpy()


def compute_integrated_gradients(model, inputs, targets, n_steps=50):
    """Compute Integrated Gradients attribution."""
    ig = IntegratedGradients(model)
    baselines = torch.zeros_like(inputs)
    attribution = ig.attribute(inputs, baselines=baselines, target=targets, n_steps=n_steps)
    return attribution.sum(dim=1).squeeze().cpu().detach().numpy()


def compute_gradient_shap(model, inputs, targets):
    """Compute GradientSHAP attribution."""
    gs = GradientShap(model)
    baselines = torch.zeros_like(inputs)
    attribution = gs.attribute(inputs, baselines=baselines, target=targets)
    return attribution.sum(dim=1).squeeze().cpu().detach().numpy()


def compute_gradcam(model, inputs, targets):
    """Compute GradCAM attribution."""
    last_conv_layer = get_last_conv_layer(model)
    gradcam = LayerGradCam(model, last_conv_layer)
    attribution = gradcam.attribute(inputs, target=targets, relu_attributions=True)
    
    # Resize to input size
    attribution_np = attribution.squeeze().cpu().detach().numpy()
    if attribution_np.shape != (IMG_SIZE, IMG_SIZE):
        attribution_np = cv2.resize(attribution_np, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    
    return attribution_np


def compute_feature_permutation(model, inputs, targets, patch_size=16):
    """Compute Feature Permutation attribution."""
    model.eval()
    explanation = np.zeros((IMG_SIZE, IMG_SIZE))
    
    with torch.no_grad():
        # Get baseline prediction
        baseline_output = model(inputs)
        baseline_prob = torch.softmax(baseline_output, dim=1)
        baseline_score = baseline_prob[0, targets].item()
        
        # Permute patches and measure importance
        for patch_h in range(0, IMG_SIZE, patch_size):
            for patch_w in range(0, IMG_SIZE, patch_size):
                # Create permuted input
                inputs_permuted = inputs.clone()
                
                # Add random noise to the patch
                noise = torch.randn_like(
                    inputs_permuted[:, :, patch_h:patch_h+patch_size, patch_w:patch_w+patch_size]
                ) * 0.1
                inputs_permuted[:, :, patch_h:patch_h+patch_size, patch_w:patch_w+patch_size] += noise
                
                # Get prediction on permuted input
                permuted_output = model(inputs_permuted)
                permuted_prob = torch.softmax(permuted_output, dim=1)
                permuted_score = permuted_prob[0, targets].item()
                
                # Calculate importance as change in prediction score
                importance = baseline_score - permuted_score
                
                # Fill the explanation map
                explanation[patch_h:patch_h+patch_size, patch_w:patch_w+patch_size] = importance
    
    return explanation


def normalize_attribution(attr):
    """Normalize attribution map for visualization."""
    attr = np.abs(attr)
    if attr.max() > 0:
        attr = attr / attr.max()
    return attr


# ============================================================================
# METRICS
# ============================================================================

def compute_par(attribution, mask):
    """
    Compute Positive Attribution Ratio (PAR).
    
    PAR measures the ratio of positive attributions inside the mask
    to the total positive attributions.
    
    Args:
        attribution: Attribution map (H, W)
        mask: Ground truth mask (H, W)
        
    Returns:
        float: PAR score (0 to 1, higher is better)
    """
    a = attribution.copy().flatten()
    s = mask.copy().flatten()
    
    # Binarize mask
    if s.max() > 1:
        s = (s > 127).astype(bool)
    else:
        s = (s > 0.5).astype(bool)
    
    # Compute PAR
    r_total = np.sum(a[a > 0])
    if r_total == 0:
        return 0.0
    
    r_within = np.sum(a[(a > 0) & s])
    return float(r_within / r_total)


def compute_rra(attribution, mask):
    """
    Compute Relevance Rank Accuracy (RRA).
    
    RRA measures how well the top-k attributed pixels overlap with the mask,
    where k is the number of pixels in the mask.
    
    Args:
        attribution: Attribution map (H, W)
        mask: Ground truth mask (H, W)
        
    Returns:
        float: RRA score (0 to 1, higher is better)
    """
    a = attribution.copy().flatten()
    s = mask.copy().flatten()
    
    # Binarize mask
    if s.max() > 1:
        s = (s > 127).astype(bool)
    else:
        s = (s > 0.5).astype(bool)
    
    # Number of pixels in the mask
    num_mask_pixels = np.sum(s)
    if num_mask_pixels == 0:
        return np.nan
    
    # Normalize attributions
    a_max = np.max(np.abs(a))
    if a_max > 0:
        a = a / a_max
    
    # Get indices of top-k attributed pixels
    top_k_indices = np.argsort(a)[-int(num_mask_pixels):]
    
    # Count how many of top-k are within the mask
    hits = np.sum(s[top_k_indices])
    
    return float(hits / num_mask_pixels)


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Get list of COVID images
    images = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.png')])
    
    if not images:
        print("No images found in the COVID dataset directory!")
        return
    
    selected_image = images[250] 
    image_path = os.path.join(DATA_DIR, selected_image)
    mask_path = os.path.join(MASK_DIR, selected_image)
    print(f"Selected image: {selected_image}")
    
    # Load and preprocess the image
    transform = get_transforms()
    original_image = Image.open(image_path).convert('L')
    input_tensor = transform(original_image).unsqueeze(0).to(device)
    input_tensor.requires_grad = True
    
    # Load the model
    model_path = os.path.join(CHECKPOINT_DIR, "CheXpert_best.pth") # ResNet50-ImageNet_best.pth
    model = load_model(model_path)
    
    # Get model prediction
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0, predicted_class].item()
    
    class_names = ["COVID-19 Infected", "Healthy"]
    print(f"Prediction: {class_names[predicted_class]} (confidence: {confidence:.2%})")
    
    # Target class for attributions (use predicted class)
    target = predicted_class
    
    # Compute all 5 XAI methods
    print("\nComputing saliency maps...")
    
    print("  1. Computing Saliency...")
    saliency_attr = compute_saliency(model, input_tensor, target)
    gc.collect()
    torch.cuda.empty_cache()
    
    print("  2. Computing Integrated Gradients...")
    intgrad_attr = compute_integrated_gradients(model, input_tensor, target)
    gc.collect()
    torch.cuda.empty_cache()
    
    print("  3. Computing GradientSHAP...")
    gradshap_attr = compute_gradient_shap(model, input_tensor, target)
    gc.collect()
    torch.cuda.empty_cache()
    
    print("  4. Computing GradCAM...")
    gradcam_attr = compute_gradcam(model, input_tensor, target)
    gc.collect()
    torch.cuda.empty_cache()
    
    print("  5. Computing Feature Permutation...")
    featperm_attr = compute_feature_permutation(model, input_tensor, target)
    gc.collect()
    torch.cuda.empty_cache()
    
    # Normalize attributions for visualization
    saliency_norm = normalize_attribution(saliency_attr)
    intgrad_norm = normalize_attribution(intgrad_attr)
    gradshap_norm = normalize_attribution(gradshap_attr)
    gradcam_norm = normalize_attribution(gradcam_attr)
    featperm_norm = normalize_attribution(featperm_attr)
    
    # Get original image for display
    original_np = np.array(original_image.resize((IMG_SIZE, IMG_SIZE)))
    
    # Load ground truth infection mask
    gt_mask = Image.open(mask_path).convert('L')
    gt_mask_np = np.array(gt_mask.resize((IMG_SIZE, IMG_SIZE)))
    
    # Compute PAR and RRA scores for each method
    print("\nComputing PAR and RRA scores...")
    
    attributions = {
        'Saliency': saliency_norm,
        'Integrated\nGradients': intgrad_norm,
        'GradientSHAP': gradshap_norm,
        'GradCAM': gradcam_norm,
        'Feature\nPermutation': featperm_norm
    }
    
    scores = {}
    for name, attr in attributions.items():
        par = compute_par(attr, gt_mask_np)
        rra = compute_rra(attr, gt_mask_np)
        scores[name] = {'PAR': par, 'RRA': rra}
        print(f"  {name.replace(chr(10), ' ')}: PAR={par:.3f}, RRA={rra:.3f}")
    
    # Create visualization
    print("\nCreating visualization...")
    
    # Use GridSpec for flexible layout
    from matplotlib.gridspec import GridSpec
    
    fig = plt.figure(figsize=(20, 9))
    gs = GridSpec(2, 5, figure=fig, height_ratios=[1, 1.1], hspace=0.25)
    
    fig.suptitle(f'XAI Methods for Chest Pneumothorax Classification\n'
                 f'Prediction: {class_names[predicted_class]} ({confidence:.2%})', 
                 fontsize=16, fontweight='bold')
    
    # Row 1: Original image and Ground Truth Mask (centered using colspan trick)
    # Create axes spanning columns 1-2 and 2-3 to center them
    ax_orig = fig.add_subplot(gs[0, 1:3])
    ax_mask = fig.add_subplot(gs[0, 2:4])
    
    # Original image
    ax_orig.imshow(original_np, cmap='gray')
    ax_orig.set_title('Original Image', fontsize=12, fontweight='bold')
    ax_orig.axis('off')
    
    # Ground truth infection mask
    ax_mask.imshow(gt_mask_np, cmap='gray')
    ax_mask.set_title('Ground Truth\nInfection Mask', fontsize=12, fontweight='bold')
    ax_mask.axis('off')
    
    # Row 2: 5 XAI methods with scores
    attr_list = [saliency_norm, intgrad_norm, gradshap_norm, gradcam_norm, featperm_norm]
    names = ['Saliency', 'Integrated\nGradients', 'GradientSHAP', 'GradCAM', 'Feature\nPermutation']
    
    for i, (attr, name) in enumerate(zip(attr_list, names)):
        ax = fig.add_subplot(gs[1, i])
        ax.imshow(original_np, cmap='gray', alpha=0.5)
        im = ax.imshow(attr, cmap='jet', alpha=0.7)
        par = scores[name]['PAR']
        rra = scores[name]['RRA']
        ax.set_title(f'{name}\nPAR: {par:.3f} | RRA: {rra:.3f}', fontsize=11, fontweight='bold')
        ax.axis('off')
    
    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.12, 0.015, 0.35])
    fig.colorbar(im, cax=cbar_ax, label='Attribution Intensity')
    
    plt.tight_layout(rect=[0, 0, 0.9, 0.93])
    
    # Save the figure
    output_path = os.path.join(OUTPUT_DIR, 'xai_saliency_example.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\nVisualization saved to: {output_path}")
    
    # Also show the plot
    plt.show()
    
    print("\nDone!")


if __name__ == "__main__":
    main()
