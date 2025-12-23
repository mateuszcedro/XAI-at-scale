#!/usr/bin/env python3
"""
Organize attention masks into respective cat and dog folders.
Matches masks with images and places them in cat/masks and dog/masks directories.
"""

import os
import shutil
from pathlib import Path

def organize_masks():
    """
    Organize attention masks into cat/masks and dog/masks directories.
    """
    
    base_path = "/teamspace/studios/this_studio/Beyond-the-Black-Box/data/pet"
    masks_source = os.path.join(base_path, "attention_masks_png")
    cat_images = os.path.join(base_path, "cat", "images")
    cat_masks = os.path.join(base_path, "cat", "masks")
    dog_images = os.path.join(base_path, "dog", "images")
    dog_masks = os.path.join(base_path, "dog", "masks")
    
    # Create masks directories if they don't exist
    Path(cat_masks).mkdir(parents=True, exist_ok=True)
    Path(dog_masks).mkdir(parents=True, exist_ok=True)
    
    print(f"Source masks directory: {masks_source}")
    print(f"Cat images directory: {cat_images}")
    print(f"Dog images directory: {dog_images}\n")
    
    # Get list of cat images
    cat_image_files = set(os.path.splitext(f)[0] for f in os.listdir(cat_images))
    dog_image_files = set(os.path.splitext(f)[0] for f in os.listdir(dog_images))
    
    print(f"Found {len(cat_image_files)} cat images")
    print(f"Found {len(dog_image_files)} dog images")
    
    # Process all masks
    mask_files = os.listdir(masks_source)
    cat_masks_moved = 0
    dog_masks_moved = 0
    unmatched = 0
    
    for mask_file in mask_files:
        if not mask_file.endswith('.png'):
            continue
        
        mask_name_without_ext = os.path.splitext(mask_file)[0]
        mask_source_path = os.path.join(masks_source, mask_file)
        
        # Check if this mask corresponds to a cat
        if mask_name_without_ext in cat_image_files:
            cat_dest_path = os.path.join(cat_masks, mask_file)
            shutil.move(mask_source_path, cat_dest_path)
            cat_masks_moved += 1
            if cat_masks_moved % 100 == 0:
                print(f"Moved {cat_masks_moved} cat masks...", end='\r')
        
        # Check if this mask corresponds to a dog
        elif mask_name_without_ext in dog_image_files:
            dog_dest_path = os.path.join(dog_masks, mask_file)
            shutil.move(mask_source_path, dog_dest_path)
            dog_masks_moved += 1
            if dog_masks_moved % 100 == 0:
                print(f"Moved {dog_masks_moved} dog masks...", end='\r')
        
        else:
            unmatched += 1
    
    print(f"\n\n✓ Organization complete!")
    print(f"  Cat masks moved: {cat_masks_moved}")
    print(f"  Dog masks moved: {dog_masks_moved}")
    print(f"  Unmatched masks: {unmatched}")
    print(f"\nMasks are now organized in:")
    print(f"  - {cat_masks}")
    print(f"  - {dog_masks}")

if __name__ == "__main__":
    organize_masks()
