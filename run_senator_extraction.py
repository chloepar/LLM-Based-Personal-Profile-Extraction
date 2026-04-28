#!/usr/bin/env python3
"""
Senator Dataset Extraction Configuration Matrix & Orchestration

Defines the full extraction experiment matrix:
- 5 defense mechanisms (no, mask, replace_at_dot, pi_ci, hyperlink)
- 4 prompt types (direct, pseudocode, contextual, persona)
- 2 ICL settings (0 zero-shot, 5 few-shot)
- 2 adaptive attack settings (no, yes)
- Total: 5 × 4 × 2 × 2 = 80 configurations

Usage:
  python run_senator_extraction.py --mode pilot --scale 10-20
  python run_senator_extraction.py --mode full --scale 95
"""

import os
import subprocess
import sys
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path


# ============================================================================
# EXTRACTION CONFIGURATION MATRIX
# ============================================================================

DEFENSES = [
    'no',              # Baseline (no protection)
    'mask',            # Mask person name
    'replace_at_dot',  # Replace @ and .
    'pi_ci',           # Prompt injection - contextual instruction
    'hyperlink'        # Convert to hyperlinks
]

PROMPT_TYPES = [
    'direct',          # Standard Q&A
    'pseudocode',      # Code format
    'contextual',      # With framing context
    'persona'          # Role-based
]

ICL_NUMBERS = [0, 5]        # Zero-shot vs few-shot
ADAPTIVE_ATTACKS = ['no', 'yes']

# Configuration priorities (order of execution)
# Baseline first, then defenses, then adaptive attacks
PRIORITY_ORDER = [
    ('no', 'direct', 0, 'no'),           # 0: Baseline zero-shot
    ('no', 'direct', 5, 'no'),           # 1: Baseline few-shot
    ('mask', 'direct', 5, 'no'),         # 2: Mask + direct + few-shot
    ('replace_at_dot', 'direct', 5, 'no'),  # 3: Symbol replacement
    ('pi_ci', 'direct', 5, 'no'),        # 4: Prompt injection
    ('hyperlink', 'direct', 5, 'no'),    # 5: Hyperlink
    ('pi_ci', 'pseudocode', 5, 'yes'),   # 6: Adaptive attack test
]


# ============================================================================
# EXTRACTION MATRIX MANAGEMENT
# ============================================================================

def generate_all_configs():
    """Generate all 80 configurations"""
    configs = []
    for defense in DEFENSES:
        for prompt_type in PROMPT_TYPES:
            for icl_num in ICL_NUMBERS:
                for adaptive in ADAPTIVE_ATTACKS:
                    configs.append({
                        'defense': defense,
                        'prompt_type': prompt_type,
                        'icl_num': icl_num,
                        'adaptive_attack': adaptive
                    })
    return configs


def get_priority_configs():
    """Get prioritized subset for efficient testing"""
    priority_configs = []
    for defense, prompt_type, icl_num, adaptive in PRIORITY_ORDER:
        priority_configs.append({
            'defense': defense,
            'prompt_type': prompt_type,
            'icl_num': icl_num,
            'adaptive_attack': adaptive
        })
    return priority_configs


def get_pilot_configs():
    """Get minimal configs for pilot testing (10 total)"""
    return [
        # Baseline variants (3)
        {'defense': 'no', 'prompt_type': 'direct', 'icl_num': 0, 'adaptive_attack': 'no'},
        {'defense': 'no', 'prompt_type': 'direct', 'icl_num': 5, 'adaptive_attack': 'no'},
        {'defense': 'no', 'prompt_type': 'pseudocode', 'icl_num': 5, 'adaptive_attack': 'no'},
        
        # Defense variants (4)
        {'defense': 'mask', 'prompt_type': 'direct', 'icl_num': 5, 'adaptive_attack': 'no'},
        {'defense': 'replace_at_dot', 'prompt_type': 'direct', 'icl_num': 5, 'adaptive_attack': 'no'},
        {'defense': 'pi_ci', 'prompt_type': 'direct', 'icl_num': 5, 'adaptive_attack': 'no'},
        {'defense': 'hyperlink', 'prompt_type': 'direct', 'icl_num': 5, 'adaptive_attack': 'no'},
        
        # Adaptive attack (2)
        {'defense': 'pi_ci', 'prompt_type': 'pseudocode', 'icl_num': 5, 'adaptive_attack': 'yes'},
        {'defense': 'mask', 'prompt_type': 'contextual', 'icl_num': 5, 'adaptive_attack': 'yes'},
        
        # Alternative prompt type (1)
        {'defense': 'no', 'prompt_type': 'persona', 'icl_num': 5, 'adaptive_attack': 'no'},
    ]


# ============================================================================
# EXTRACTION EXECUTION
# ============================================================================

def build_command(config, task_config_path, model_config_path, verbose=1, max_profiles=0, inter_profile_delay=0, api_key_pos=0):
    """Build main.py command for a config"""
    cmd = [
        'python3', 'main.py',
        '--task_config_path', task_config_path,
        '--model_config_path', model_config_path,
        '--model_name', 'llama-3.1-8b-instant',
        '--defense', config['defense'],
        '--prompt_type', config['prompt_type'],
        '--icl_num', str(config['icl_num']),
        '--adaptive_attack', config['adaptive_attack'],
        '--verbose', str(verbose),
        '--redundant_info_filtering', 'True',
        '--max_profiles', str(max_profiles),
        '--inter_profile_delay', str(inter_profile_delay),
        '--api_key_pos', str(api_key_pos),
    ]
    return cmd


def run_extraction(config, task_config_path, model_config_path, verbose=1, max_profiles=0, inter_profile_delay=0, api_key_pos=0):
    """Execute single extraction config"""
    cmd = build_command(config, task_config_path, model_config_path, verbose, max_profiles, inter_profile_delay, api_key_pos)
    
    print(f"\n{'='*70}")
    print(f"Config: defense={config['defense']}, prompt={config['prompt_type']}, "
          f"icl={config['icl_num']}, adaptive={config['adaptive_attack']}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*70}")
    
    try:
        result = subprocess.run(cmd, check=True)
        print(f"✓ Config completed successfully")
        return True
    except subprocess.TimeoutExpired:
        print(f"✗ Config timed out (1 hour limit)")
        return False
    except subprocess.CalledProcessError as e:
        print(f"✗ Config failed with error code {e.returncode}")
        return False
    except Exception as e:
        print(f"✗ Config failed with error: {e}")
        return False


def run_extraction_batch(configs, task_config_path, model_config_path, verbose=1, resume=False, max_profiles=0, inter_profile_delay=0, parallel=1):
    """Run batch with checkpoint support. parallel=N runs N configs concurrently, each on a different API key."""
    checkpoint_path = './extraction_checkpoint.json'

    if resume:
        checkpoint = load_checkpoint(checkpoint_path)
        completed_keys = {tuple(k) for k in checkpoint['completed']}
        results = {'successful': 0, 'failed': 0, 'timed_out': 0, 'configs': checkpoint['results']}
    else:
        checkpoint = {'completed': set(), 'results': []}
        completed_keys = set()
        results = {'successful': 0, 'failed': 0, 'timed_out': 0, 'configs': []}

    pending = [(i, c) for i, c in enumerate(configs, 1) if config_to_key(c) not in completed_keys]
    skipped = len(configs) - len(pending)
    total = len(configs)
    print(f"\nSkipping {skipped} already-completed configs, {len(pending)} remaining.")

    def _run(i, config, api_key_pos):
        print(f"\n[{i}/{total}] Starting (key={api_key_pos}): defense={config['defense']}, prompt={config['prompt_type']}, icl={config['icl_num']}, adaptive={config['adaptive_attack']}")
        return config, run_extraction(config, task_config_path, model_config_path, verbose, max_profiles, inter_profile_delay, api_key_pos)

    try:
        with ThreadPoolExecutor(max_workers=parallel) as executor:
            futures = {
                executor.submit(_run, i, config, idx % parallel): config
                for idx, (i, config) in enumerate(pending)
            }
            for future in as_completed(futures):
                config, success = future.result()
                completed_keys.add(config_to_key(config))
                status = 'success' if success else 'failed'
                results['successful' if success else 'failed'] += 1
                results['configs'].append({**config, 'status': status})
                checkpoint['completed'] = completed_keys
                checkpoint['results'] = results['configs']
                save_checkpoint(checkpoint, checkpoint_path)
                print(f"{'✓' if success else '✗'} Finished: defense={config['defense']}, prompt={config['prompt_type']}, icl={config['icl_num']}, adaptive={config['adaptive_attack']} → {status}")
    except KeyboardInterrupt:
        print("\n✗ Batch interrupted - checkpoint saved")

    print(f"\n✓ {len(completed_keys)} completed, {skipped} skipped")
    return results


def save_results(results, output_path='./extraction_results.json'):
    """Save batch results to JSON"""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Results saved to {output_path}")

def config_to_key(config):
    """Convert config dict to hashable key"""
    return (config['defense'], config['prompt_type'], config['icl_num'], config['adaptive_attack'])

def load_checkpoint(checkpoint_path='./extraction_checkpoint.json'):
    """Load existing checkpoint or return empty"""
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r') as f:
            return json.load(f)
    return {'completed': set(), 'results': []}

def save_checkpoint(checkpoint, checkpoint_path='./extraction_checkpoint.json'):
    """Save checkpoint with completed configs"""
    # Convert set to list for JSON serialization
    checkpoint_data = {
        'completed': [list(k) for k in checkpoint['completed']],
        'results': checkpoint['results'],
        'timestamp': datetime.now().isoformat()
    }
    with open(checkpoint_path, 'w') as f:
        json.dump(checkpoint_data, f, indent=2)

# ============================================================================
# MAIN ORCHESTRATION
# ============================================================================

def print_matrix_info():
    """Print extraction matrix info"""
    print("\n" + "="*70)
    print("EXTRACTION CONFIGURATION MATRIX")
    print("="*70)
    print(f"Defenses: {len(DEFENSES)} - {', '.join(DEFENSES)}")
    print(f"Prompt Types: {len(PROMPT_TYPES)} - {', '.join(PROMPT_TYPES)}")
    print(f"ICL Settings: {len(ICL_NUMBERS)} - {ICL_NUMBERS}")
    print(f"Adaptive Attack: {len(ADAPTIVE_ATTACKS)} - {ADAPTIVE_ATTACKS}")
    print(f"\nTotal Configurations: {len(DEFENSES) * len(PROMPT_TYPES) * len(ICL_NUMBERS) * len(ADAPTIVE_ATTACKS)} = 80")
    print(f"Pilot Configs: {len(get_pilot_configs())}")
    print(f"Priority Configs: {len(get_priority_configs())}")
    print("="*70 + "\n")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Run senator extraction experiments')
    parser.add_argument('--mode', choices=['matrix', 'priority', 'pilot', 'demo'], 
                       default='pilot',
                       help='Execution mode')
    parser.add_argument('--task_config', default='./configs/task_configs/senator.json',
                       help='Task config path')
    parser.add_argument('--model_config', default='./configs/model_configs/groq_config.json',
                       help='Model config path')
    parser.add_argument('--verbose', type=int, default=1,
                       help='Verbosity level')
    parser.add_argument('--print_only', action='store_true',
                    help='Print commands without executing')
    parser.add_argument('--resume', action='store_true',
                    help='Resume from checkpoint')
    parser.add_argument('--max_profiles', type=int, default=0,
                       help='Max senators to process per config (0 = all 100)')
    parser.add_argument('--inter_profile_delay', type=float, default=0,
                       help='Extra seconds to sleep between profiles (helps with rate limits)')
    parser.add_argument('--prompt_filter', default='',
                       help='Only run configs with this prompt type (e.g. direct, pseudocode, contextual, persona)')
    parser.add_argument('--adaptive_filter', default='',
                       help='Only run configs with this adaptive_attack value (no or yes)')
    parser.add_argument('--defense_filter', default='',
                       help='Only run configs with this defense (e.g. no, mask, pi_ci, replace_at_dot, hyperlink)')
    parser.add_argument('--parallel', type=int, default=1,
                       help='Number of configs to run concurrently (use one per API key, e.g. 4)')

    args = parser.parse_args()

    # Print matrix info
    print_matrix_info()

    # Select configs based on mode
    if args.mode == 'matrix':
        configs = generate_all_configs()
        print(f"Mode: FULL MATRIX - {len(configs)} configurations")
    elif args.mode == 'priority':
        configs = get_priority_configs()
        print(f"Mode: PRIORITY - {len(configs)} configurations")
    elif args.mode == 'pilot':
        configs = get_pilot_configs()
        print(f"Mode: PILOT - {len(configs)} configurations")
    elif args.mode == 'demo':
        configs = [get_pilot_configs()[0]]
        print(f"Mode: DEMO - {len(configs)} configuration (first only)")

    if args.prompt_filter:
        configs = [c for c in configs if c['prompt_type'] == args.prompt_filter]
        print(f"Filtered to prompt_type='{args.prompt_filter}': {len(configs)} configurations")

    if args.adaptive_filter:
        configs = [c for c in configs if c['adaptive_attack'] == args.adaptive_filter]
        print(f"Filtered to adaptive_attack='{args.adaptive_filter}': {len(configs)} configurations")

    if args.defense_filter:
        configs = [c for c in configs if c['defense'] == args.defense_filter]
        print(f"Filtered to defense='{args.defense_filter}': {len(configs)} configurations")
    
    print(f"\nConfigs to run:")
    for i, cfg in enumerate(configs, 1):
        print(f"  {i}. defense={cfg['defense']}, prompt={cfg['prompt_type']}, "
              f"icl={cfg['icl_num']}, adaptive={cfg['adaptive_attack']}")
    
    # Print-only mode
    if args.print_only:
        print(f"\n{'='*70}")
        print("PRINT-ONLY MODE - No execution")
        print("="*70)
        for i, config in enumerate(configs, 1):
            cmd = build_command(config, args.task_config, args.model_config, args.verbose, args.max_profiles, args.inter_profile_delay)
            print(f"\n[{i}] {' '.join(cmd)}")
        return
    
    # Confirm before running
    print(f"\n{'='*70}")
    response = input(f"Ready to run {len(configs)} configurations. Continue? [y/N]: ").strip().lower()
    if response != 'y':
        print("Cancelled.")
        return
    
    # Run batch
    print(f"\nStarting extraction batch at {datetime.now().isoformat()}")
    results = run_extraction_batch(configs, args.task_config, args.model_config, args.verbose, resume=args.resume, max_profiles=args.max_profiles, inter_profile_delay=args.inter_profile_delay, parallel=args.parallel)
    
    # Save results
    results['timestamp'] = datetime.now().isoformat()
    results['mode'] = args.mode
    results['total'] = len(configs)
    save_results(results)
    
    # Summary
    print(f"\n{'='*70}")
    print("BATCH SUMMARY")
    print("="*70)
    print(f"Total: {results['total']}")
    print(f"Successful: {results['successful']}")
    print(f"Failed: {results['failed']}")
    print(f"Success Rate: {results['successful']/results['total']*100:.1f}%")
    print("="*70)


if __name__ == '__main__':
    main()
