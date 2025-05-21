#!/usr/bin/env python
"""
Script to visualize posterior distributions from model output files.

This script is a convenient wrapper around the visualize_posterior_distributions
function in the sleep_analysis.models module.

Usage:
    python -m sleep_analysis.visualize_posteriors path/to/models_dir [path/to/output_dir]
"""

import sys
import os
from pathlib import Path
from sleep_analysis.models import visualize_posterior_distributions


def main():
    """Process command line arguments and run visualization."""
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} path/to/models_dir [path/to/output_dir]")
        sys.exit(1)

    models_dir = Path(sys.argv[1])
    if not models_dir.exists():
        print(f"Error: Directory {models_dir} does not exist.")
        sys.exit(1)

    # Optional output directory
    output_dir = models_dir
    if len(sys.argv) > 2:
        output_dir = Path(sys.argv[2])
        output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scanning for model output files in {models_dir}")
    print(f"Visualizations will be saved to {output_dir}/posterior_visualizations")

    # Run the visualization function
    visualize_posterior_distributions(output_dir, model_files=None)

    print("Visualization process complete!")


if __name__ == "__main__":
    main()
