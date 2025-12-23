#!/usr/bin/env python3
"""
Convert attention masks from CSV to PNG images.
Reads cat_dog_attention.csv and saves each attention mask as a PNG file.
"""

import pandas as pd
import numpy as np
from PIL import Image
import ast
import os
from pathlib import Path
import sys

def parse_matrix_string(matrix_str):
    """
    Parse matrix string representation and convert to numpy array.
    """
    try:
        # Handle the string representation of a list/matrix
        matrix = ast.literal_eval(matrix_str)
        return np.array(matrix, dtype=np.uint8)
    except (ValueError, SyntaxError) as e:
        print(f"Error parsing matrix: {e}")
        return None

def convert_attention_masks_to_png(csv_path, output_dir="attention_masks_png"):
    """
    Read CSV file with attention masks and convert to PNG images.
    
    Parameters:
    -----------
    csv_path : str
        Path to the CSV file containing attention masks
    output_dir : str
        Output directory for PNG files (default: attention_masks_png)
    """
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"Reading CSV file: {csv_path}")
    print("This may take a moment for large files...\n")
    
    # Read CSV in chunks to handle large file
    total_converted = 0
    failed_conversions = 0
    
    # First, let's count total rows
    total_rows = sum(1 for _ in open(csv_path)) - 1  # Exclude header
    print(f"Total attention masks to process: {total_rows}\n")
    
    # Read and process CSV
    for chunk in pd.read_csv(csv_path, chunksize=10):
        for idx, row in chunk.iterrows():
            try:
                img_name = row['img_idx']
                attention_str = row['attention']
                
                # Parse the matrix string to numpy array
                matrix = parse_matrix_string(attention_str)
                
                if matrix is None:
                    failed_conversions += 1
                    continue
                
                # Normalize if needed (convert binary values to 0-255 range)
                if matrix.max() == 1:
                    matrix = matrix * 255
                else:
                    # Ensure values are in 0-255 range
                    matrix = np.clip(matrix, 0, 255).astype(np.uint8)
                
                # Create PIL Image from array
                if len(matrix.shape) == 2:
                    # Grayscale image
                    img = Image.fromarray(matrix, mode='L')
                else:
                    # Color image (shouldn't happen with attention masks, but handle it)
                    if matrix.shape[2] == 3:
                        img = Image.fromarray(matrix.astype(np.uint8), mode='RGB')
                    else:
                        img = Image.fromarray(matrix.astype(np.uint8))
                
                # Create output filename (replace .jpg/.png with .png)
                output_name = os.path.splitext(img_name)[0] + '.png'
                output_path = os.path.join(output_dir, output_name)
                
                # Create subdirectories if needed (e.g., cat/file.png)
                os.makedirs(os.path.dirname(output_path) or output_dir, exist_ok=True)
                
                # Save image
                img.save(output_path)
                total_converted += 1
                
                if total_converted % 100 == 0:
                    print(f"Converted {total_converted} masks...", end='\r')
                
            except Exception as e:
                print(f"Error processing row {idx}: {e}")
                failed_conversions += 1
    
    print(f"\n✓ Conversion complete!")
    print(f"  Successfully converted: {total_converted}")
    print(f"  Failed conversions: {failed_conversions}")
    print(f"  Output directory: {output_dir}")

if __name__ == "__main__":
    csv_path = "/teamspace/studios/this_studio/Beyond-the-Black-Box/data/pet/cat_dog_attention.csv"
    output_dir = "/teamspace/studios/this_studio/Beyond-the-Black-Box/data/pet/attention_masks_png"
    
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    
    convert_attention_masks_to_png(csv_path, output_dir)
