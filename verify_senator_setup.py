#!/usr/bin/env python3
"""
Verification script for senator dataset setup.
Checks that config loads, task manager initializes, and data is valid.
"""

import sys
import os
import json

# Add repo to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from LLMPersonalInfoExtraction.utils import open_config
from LLMPersonalInfoExtraction.tasks import TaskManager

def verify_setup():
    """Verify the senator dataset setup"""
    
    print("=" * 70)
    print("SENATOR DATASET VERIFICATION")
    print("=" * 70)
    
    # 1. Verify config file exists and loads
    print("\n[1] Loading task config...")
    config_path = './configs/task_configs/senator.json'
    try:
        config = open_config(config_path)
        print(f"  ✓ Config loaded from {config_path}")
        print(f"    Dataset: {config['dataset_info']['dataset']}")
        print(f"    Path: {config['dataset_info']['path']}")
    except Exception as e:
        print(f"  ✗ Failed to load config: {e}")
        return False
    
    # 2. Verify data directory structure
    print("\n[2] Checking data directory structure...")
    data_dir = './data/senator'
    required_files = {
        'labels.json': 'Ground truth labels',
        'info_categories.txt': 'Information categories',
    }
    
    for fname, desc in required_files.items():
        fpath = os.path.join(data_dir, fname)
        if os.path.exists(fpath):
            print(f"  ✓ {fname} ({desc})")
        else:
            print(f"  ✗ {fname} NOT FOUND")
            return False
    
    # 3. Check HTML files
    html_files = [f for f in os.listdir(data_dir) if f.endswith('.html')]
    print(f"\n[3] HTML files: {len(html_files)} profiles")
    print(f"    Examples: {', '.join(html_files[:3])}")
    
    # 4. Check ICL directory
    icl_dir = os.path.join(data_dir, 'icl')
    if os.path.exists(icl_dir):
        icl_files = [f for f in os.listdir(icl_dir) if f.endswith('.html')]
        print(f"\n[4] ICL examples: {len(icl_files)} profiles in {icl_dir}/")
        print(f"    ICL files: {', '.join(sorted(icl_files))}")
    
    # 5. Verify labels.json structure
    print("\n[5] Verifying labels.json...")
    labels_path = os.path.join(data_dir, 'labels.json')
    try:
        with open(labels_path, 'r') as f:
            labels = json.load(f)
        print(f"  ✓ Loaded {len(labels)} senator records")
        
        # Check sample record
        first_key = list(labels.keys())[0]
        sample = labels[first_key]
        print(f"\n  Sample record ({first_key}):")
        for key, val in sample.items():
            if isinstance(val, (str, type(None))):
                val_str = val[:50] + "..." if isinstance(val, str) and len(val) > 50 else str(val)
            elif isinstance(val, list):
                val_str = f"[{len(val)} items]"
            else:
                val_str = str(val)[:50]
            print(f"    - {key}: {val_str}")
        
        # Check fields present
        required_fields = ['birthdate', 'gender', 'race_ethnicity', 'committee_roles', 'religion', 'education']
        all_fields_present = all(field in sample for field in required_fields)
        if all_fields_present:
            print(f"  ✓ All {len(required_fields)} required fields present")
        else:
            print(f"  ✗ Missing fields!")
            return False
            
    except Exception as e:
        print(f"  ✗ Failed to load labels.json: {e}")
        return False
    
    # 6. Verify info categories
    print("\n[6] Verifying info_categories.txt...")
    try:
        with open(os.path.join(data_dir, 'info_categories.txt'), 'r') as f:
            categories = [line.strip() for line in f if line.strip()]
        print(f"  ✓ {len(categories)} categories: {', '.join(categories)}")
    except Exception as e:
        print(f"  ✗ Failed to load categories: {e}")
        return False
    
    # 7. Test TaskManager initialization
    print("\n[7] Testing TaskManager initialization...")
    try:
        task_manager = TaskManager(config)
        print(f"  ✓ TaskManager created successfully")
        print(f"    Total profiles: {len(task_manager)}")
        
        # Try loading a sample
        sample_data, sample_label = task_manager[0]
        print(f"  ✓ Sample profile loaded")
        print(f"    HTML lines: {len(sample_data)}")
        print(f"    Ground truth fields: {list(sample_label.keys())}")
    except Exception as e:
        print(f"  ✗ TaskManager failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 70)
    print("✓ ALL VERIFICATION CHECKS PASSED")
    print("=" * 70)
    print("\nReady to run experiments!")
    print("Example command:")
    print("  python main.py --task_config_path ./configs/task_configs/senator.json \\")
    print("                 --model_config_path ./configs/model_configs/gpt_config.json \\")
    print("                 --defense no")
    
    return True

if __name__ == '__main__':
    success = verify_setup()
    sys.exit(0 if success else 1)
