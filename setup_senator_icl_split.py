#!/usr/bin/env python3
"""
Set up ICL (in-context learning) split for senator dataset.
Moves first 5 senators alphabetically to icl/ subdirectory for few-shot examples.
"""

import os
import shutil
from pathlib import Path

def setup_icl_split(data_dir, icl_count=5):
    """
    Create ICL subset by moving senators to icl/ subdirectory.
    
    Args:
        data_dir: Path to ./data/senator/
        icl_count: Number of senators to use as ICL examples
    """
    
    icl_dir = os.path.join(data_dir, 'icl')
    os.makedirs(icl_dir, exist_ok=True)
    
    # Get all HTML files sorted alphabetically
    html_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.html')])
    
    if len(html_files) < icl_count:
        print(f"WARNING: Only {len(html_files)} HTML files found, but requested {icl_count} for ICL")
        icl_count = len(html_files)
    
    # Select first icl_count for ICL examples
    icl_files = html_files[:icl_count]
    eval_files = html_files[icl_count:]
    
    print(f"Setting up ICL split: {icl_count} for examples, {len(eval_files)} for evaluation")
    print(f"\nICL Example Files:")
    for fname in icl_files:
        src = os.path.join(data_dir, fname)
        dst = os.path.join(icl_dir, fname)
        shutil.copy(src, dst)  # Copy, don't move - keep originals in main dir
        print(f"  ✓ {fname}")
    
    print(f"\nEvaluation Files: {len(eval_files)} profiles")
    print(f"First few: {', '.join(eval_files[:3])}")
    print(f"\nICL split complete!")
    print(f"- ICL examples: {icl_dir}/")
    print(f"- Evaluation: {data_dir}/ (all profiles)")

if __name__ == '__main__':
    data_dir = './data/senator'
    
    if not os.path.exists(data_dir):
        print(f"ERROR: Data directory not found at {data_dir}")
    else:
        setup_icl_split(data_dir, icl_count=5)
