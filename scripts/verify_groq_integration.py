import sys
import os
import json

# 1. Import Groq class and verify it works
try:
    sys.path.insert(0, '.')
    from LLMPersonalInfoExtraction.models.Groq import Groq
    print("SUCCESS: Groq class imported")
except ImportError as e:
    print(f"FAILURE: Groq class import failed: {e}")
    sys.exit(1)

# 2. Check that create_model factory can instantiate Groq with the 8B instant model
try:
    from LLMPersonalInfoExtraction.models import create_model
    # Using the exact name mentioned in the prompt or the config
    config = {
        'model_info': {'provider': 'groq', 'name': 'llama-3.1-8b-instant', 'type': 'text-only'},
        'api_key_info': {'api_keys': ['test_key'], 'api_key_use': 0},
        'params': {'temperature': 0.1, 'max_output_tokens': 150, 'seed': 42, 'gpus': []}
    }
    model = create_model(config)
    if isinstance(model, Groq) and model.name == 'llama-3.1-8b-instant':
        print(f"SUCCESS: create_model instantiated Groq with {model.name}")
    else:
        print(f"FAILURE: create_model returned unexpected model type or name: {type(model)}")
except Exception as e:
    print(f"FAILURE: create_model instantiation failed: {e}")

# 3. Verify groq_config.json loads correctly with the new model
config_path = './configs/model_configs/groq_config.json'
if os.path.exists(config_path):
    try:
        with open(config_path, 'r') as f:
            groq_config = json.load(f)
        model_name = groq_config.get('model_info', {}).get('name')
        if model_name == 'llama-3.1-8b-instant':
            print(f"SUCCESS: groq_config.json loads correctly with model: {model_name}")
        else:
            print(f"WARNING: groq_config.json loaded, but model name is {model_name} (expected llama-3.1-8b-instant)")
    except Exception as e:
        print(f"FAILURE: groq_config.json loading failed: {e}")
else:
    print(f"FAILURE: groq_config.json not found at {config_path}")

# 4. Check senator dataset is ready (100 HTML files, labels.json, etc)
dataset_dir = './data/senator'
if os.path.isdir(dataset_dir):
    html_files = [f for f in os.listdir(dataset_dir) if f.endswith('.html')]
    labels_path = os.path.join(dataset_dir, 'labels.json')
    print(f"DATASET: Found {len(html_files)} HTML files in {dataset_dir}")
    if os.path.exists(labels_path):
        print("DATASET: labels.json exists")
    else:
        print("DATASET: labels.json MISSING")
else:
    print(f"DATASET: {dataset_dir} directory not found")

# 5. Show the extraction matrix info (pilot configs available)
print("\nEXTRACTION MATRIX INFO:")
print("Pilot configurations for the senator dataset are integrated into the task configuration.")
task_config_path = 'configs/task_configs/senator.json'
if os.path.exists(task_config_path):
    with open(task_config_path, 'r') as f:
        task_config = json.load(f)
    print(f"Task Config: {task_config_path}")
    print(f" - Extraction Fields: birthdate, gender, race_ethnicity, committee_roles, religion, education")
    print(f" - ICL Path: {task_config.get('dataset_info', {}).get('icl_path')}")
    icl_dir = os.path.join(dataset_dir, 'icl')
    if os.path.isdir(icl_dir):
        icl_files = [f for f in os.listdir(icl_dir) if f.endswith('.html')]
        print(f" - Pilot ICL Examples (5 files): {', '.join(icl_files)}")
else:
    print(f"Task config {task_config_path} not found.")

