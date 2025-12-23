#!/usr/bin/env python
"""Display aggregated results summary"""

import json
import logging
from pathlib import Path
from io import StringIO

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Load aggregated results
results_path = Path("training_results/aggregated_results.json")

if not results_path.exists():
    logger.error(f"Aggregated results file not found at {results_path}")
    exit(1)

with open(results_path, 'r') as f:
    aggregated_results = json.load(f)

# Create output buffer for saving to file
output_buffer = StringIO()

# Display summary
header = "\n" + "="*80 + "\nAGGREGATED RESULTS SUMMARY\n" + "="*80
logger.info(header)
output_buffer.write(header + "\n")

# Create output buffer for saving to file
output_buffer = StringIO()

# Display summary
header = "\n" + "="*80 + "\nAGGREGATED RESULTS SUMMARY\n" + "="*80
logger.info(header)
output_buffer.write(header + "\n")

for model_name, metrics in aggregated_results.items():
    model_header = f"\n{model_name}:"
    logger.info(model_header)
    output_buffer.write(model_header + "\n")
    
    for metric_key, stats in metrics.items():
        if metric_key == 'training_time':
            lines = [f"  {metric_key}:",
                    f"    - Mean: {stats['mean']/60:.2f} min ({stats['mean']:.2f}s)",
                    f"    - Std: {stats['std']:.2f}s",
                    f"    - Total: {stats['total']/60:.2f} min (all runs)"]
            for line in lines:
                logger.info(line)
                output_buffer.write(line + "\n")
        elif metric_key == 'avg_epoch_time':
            line = f"  {metric_key}: {stats['mean']:.2f} ± {stats['std']:.2f} seconds"
            logger.info(line)
            output_buffer.write(line + "\n")
        elif metric_key == 'inference_time':
            line = f"  {metric_key}: {stats['mean']:.4f} ± {stats['std']:.4f} seconds"
            logger.info(line)
            output_buffer.write(line + "\n")
        elif metric_key == 'time_per_sample_ms':
            line = f"  {metric_key}: {stats['mean']:.4f} ± {stats['std']:.4f} ms"
            logger.info(line)
            output_buffer.write(line + "\n")
        elif metric_key == 'num_test_samples':
            line = f"  {metric_key}: {stats}"
            logger.info(line)
            output_buffer.write(line + "\n")
        elif isinstance(stats, dict) and 'mean' in stats:
            line = f"  {metric_key}: {stats['mean']:.4f} ± {stats['std']:.4f}"
            logger.info(line)
            output_buffer.write(line + "\n")

# Save to file
output_file = Path("training_results/aggregated_results_summary.txt")
with open(output_file, 'w') as f:
    f.write(output_buffer.getvalue())

logger.info(f"\nResults saved to {output_file}")
