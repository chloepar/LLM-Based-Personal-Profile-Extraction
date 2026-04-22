# Senator Dataset Extension - Progress Tracker

## Project Overview
Extending the PIE (LLM-Based Personal Information Extraction) framework to evaluate information extraction and defense mechanisms on U.S. Senator profiles in HTML format.

**Status**: 🔄 **PHASE 2 IN PROGRESS** - Groq integration complete, extraction orchestration running

---

## Phase 1: Data Preparation ✅

### ✅ Step 1: Convert CSV to JSON Labels
- **File Created**: [convert_senate_csv_to_json.py](convert_senate_csv_to_json.py)
- **Input**: `external_data/ground_truth/senate_ground_truth_updated_manual.csv`
- **Output**: `./data/senator/labels.json`
- **Records Converted**: 100 senator profiles
- **Fields Extracted**: 6 fields per record
  - `birthdate`
  - `gender`
  - `race_ethnicity`
  - `committee_roles`
  - `religion`
  - `education`

### ✅ Step 2: Copy Senator HTML Files
- **Source**: `external_data/senate_html/` (100 HTML files)
- **Destination**: `./data/senator/` (100 HTML files)
- **Status**: All files successfully copied

### ✅ Step 3: Create Info Categories File
- **File**: `./data/senator/info_categories.txt`
- **Contents**: 6 extraction target fields (one per line)
  ```
  birthdate
  gender
  race_ethnicity
  committee_roles
  religion
  education
  ```

---

## Phase 2: Configuration Setup ✅

### ✅ Step 4: Create Task Configuration
- **File**: [configs/task_configs/senator.json](configs/task_configs/senator.json)
- **Structure**: Mirrors existing task configs (celebrity.json, physician.json)
- **Config**:
  ```json
  {
    "task_info": {
      "task": "info_extraction",
      "type": "text_generation"
    },
    "dataset_info": {
      "dataset": "senator",
      "path": "./data/senator",
      "label_path": "./data/senator/labels.json",
      "icl_path": "./data/senator",
      "icl_label_path": "./data/senator/labels.json"
    }
  }
  ```

### ✅ Step 5: Set Up ICL (In-Context Learning) Split
- **File**: [setup_senator_icl_split.py](setup_senator_icl_split.py)
- **ICL Examples**: 5 senator profiles in `./data/senator/icl/`
  - Adam_Schiff_CA.html
  - Alan_Armstrong_OK.html
  - Alex_Padilla_CA.html
  - Amy_Klobuchar_MN.html
  - Andy_Kim_NJ.html
- **Evaluation Set**: 95 remaining senator profiles
- **Purpose**: Few-shot learning with domain-specific examples

---

## Phase 3: Verification ✅

### ✅ Step 6: Complete Verification
- **File**: [verify_senator_setup.py](verify_senator_setup.py)
- **Verification Checks**:
  - ✅ Config file loads correctly
  - ✅ Data directory structure valid
  - ✅ 100 HTML profiles present
  - ✅ 5 ICL examples in subdirectory
  - ✅ labels.json contains 100 valid records
  - ✅ All 6 required fields present in each record
  - ✅ info_categories.txt valid
  - ✅ TaskManager initializes successfully
  - ✅ Sample profile loads without errors

**Verification Result**: ✅ ALL CHECKS PASSED

---

## Phase 2: Groq Integration & Extraction Setup 🔄

### ✅ Step 7: Implement Groq LLaMA Model Provider
- **File Created**: [LLMPersonalInfoExtraction/models/Groq.py](LLMPersonalInfoExtraction/models/Groq.py)
- **Model**: llama-3.1-8b-instant (Groq)
- **Features**:
  - Implements Model interface
  - Temperature: 0.1 (low randomness for deterministic extraction)
  - Max tokens: 150
  - Seed: 100 (reproducible results)
  - Integrated with GroqClient API
- **Config File**: [configs/model_configs/groq_config.json](configs/model_configs/groq_config.json)
- **Factory Registration**: Updated [LLMPersonalInfoExtraction/models/__init__.py](LLMPersonalInfoExtraction/models/__init__.py)
- **Status**: ✅ Verified working

### ✅ Step 8: Fix Schema Mismatches
- **Issue**: Framework expected `name` field in labels
- **Fix**: Augmented all 100 senator records with `name` field extracted from senator_id
  - Example: `"Adam_Schiff_CA"` → `{"name": "Adam Schiff", ...}`
- **File Updated**: `./data/senator/labels.json`
- **System Prompts**: Updated `./data/system_prompts/info_category.txt` with senator-specific fields (lowercase):
  ```
  name
  birthdate
  gender
  race_ethnicity
  committee_roles
  religion
  education
  ```
- **Status**: ✅ Complete

### 🔄 Step 9: Create Extraction Orchestration Script
- **File Created**: [run_senator_extraction.py](run_senator_extraction.py)
- **Features**:
  - **Extraction Matrix**: 80 total configurations
    - Defenses: 5 (no, mask, replace_at_dot, pi_ci, hyperlink)
    - Prompt Types: 4 (direct, pseudocode, contextual, persona)
    - ICL Settings: 2 (0 zero-shot, 5 few-shot)
    - Adaptive Attack: 2 (no, yes)
  - **Execution Modes**:
    - `--mode demo`: 1 config (quick test)
    - `--mode pilot`: 10 configs (comprehensive baseline)
    - `--mode priority`: 7 prioritized configs
    - `--mode matrix`: All 80 configs
  - **Checkpoint/Resume Support**:
    - `extraction_checkpoint.json`: Tracks completed configs (batch level)
    - `.checkpoint.npz`: Per-config profile progress (within each result dir)
    - `--resume` flag resumes from last checkpoint
  - **Model**: Groq llama-3.1-8b-instant (with `--model_name` auto-configured)
- **Usage**:
  ```bash
  python3 run_senator_extraction.py --mode pilot
  python3 run_senator_extraction.py --mode pilot --resume  # Resume after interruption
  python3 run_senator_extraction.py --mode pilot --print_only  # Dry run
  ```
- **Status**: 🔄 Running (10 pilot configs in progress)

---

## Directory Structure

```
LLM-Based-Personal-Profile-Extraction/
├── ./data/senator/
│   ├── labels.json                 (100 ground truth records, 6 fields each)
│   ├── info_categories.txt         (extraction fields: birthdate, gender, etc.)
│   ├── *.html                      (100 senator profile HTML files)
│   └── icl/
│       ├── Adam_Schiff_CA.html
│       ├── Alan_Armstrong_OK.html
│       ├── Alex_Padilla_CA.html
│       ├── Amy_Klobuchar_MN.html
│       └── Andy_Kim_NJ.html
├── configs/
│   ├── model_configs/
│   │   └── groq_config.json        (Groq llama-3.1-8b-instant config)
│   └── task_configs/
│       └── senator.json            (task configuration)
├── LLMPersonalInfoExtraction/models/
│   ├── Groq.py                     (Groq provider implementation)
│   └── __init__.py                 (factory registration)
├── data/system_prompts/
│   ├── info_category.txt           (senator extraction fields)
│   ├── direct.txt                  (direct prompt template)
│   └── [other prompt types]        (pseudocode, contextual, persona)
├── convert_senate_csv_to_json.py   (CSV → JSON converter)
├── setup_senator_icl_split.py      (ICL split setup script)
├── verify_senator_setup.py         (verification script)
├── verify_groq_integration.py      (Groq integration test)
└── run_senator_extraction.py       (batch orchestration & checkpoint)
```

---

## Extracted Information Fields

| Field | Type | Example |
|-------|------|---------|
| `birthdate` | String (MM/DD/YY) | 6/22/60 |
| `gender` | String | Male, Female |
| `race_ethnicity` | String | White, Hispanic, Black, Asian American |
| `committee_roles` | String (pipe-separated) | Senate Committee on...\|Subcommittee... |
| `religion` | String | Jewish, Catholic, Baptist, etc. |
| `education` | JSON Array | [{"degree": "BA", "institution": "...", "year": null}] |

---

## Ready-to-Run Commands

### Baseline: Information Extraction (No Defense)
```bash
python main.py --task_config_path ./configs/task_configs/senator.json \
               --model_config_path ./configs/model_configs/gpt_config.json \
               --defense no \
               --icl_num 5 \
               --verbose 1
```

### With Mask Defense
```bash
python main.py --task_config_path ./configs/task_configs/senator.json \
               --model_config_path ./configs/model_configs/gpt_config.json \
               --defense mask \
               --icl_num 5 \
               --verbose 1
```

### With Prompt Injection Defense
```bash
python main.py --task_config_path ./configs/task_configs/senator.json \
               --model_config_path ./configs/model_configs/gpt_config.json \
               --defense pi_ci \
               --icl_num 5 \
               --verbose 1
```

### Adaptive Attack Testing
```bash
python main.py --task_config_path ./configs/task_configs/senator.json \
               --model_config_path ./configs/model_configs/gpt_config.json \
               --defense pi_ci \
               --adaptive_attack yes \
               --icl_num 5 \
               --verbose 1
```

### Evaluation with Different Models
```bash
# With PaLM2
python main.py --task_config_path ./configs/task_configs/senator.json \
               --model_config_path ./configs/model_configs/palm2_config.json \
               --defense no --icl_num 5

# With LLaMA (specify GPU)
python main.py --task_config_path ./configs/task_configs/senator.json \
               --model_config_path ./configs/model_configs/llama_config.json \
               --defense no --icl_num 5 --gpus "0"
```

---

## Output Locations

- **Results**: `./result/` — Extraction metrics and scores
- **Logs**: `./log/` — Detailed execution logs with Accuracy and ROUGE-1 scores
- **Additional Evaluation**: Use `./evaluate.py` for BERT score computation

---

## Phase 3: Testing & Evaluation (NEXT)

### Planned Experiments

| # | Experiment | Defense | Adaptive Attack | LLM Models | Status |
|---|------------|---------|-----------------|-----------|--------|
| 1 | Baseline Extraction (zero-shot) | None | No | Groq LLaMA | 🔄 Running |
| 2 | Baseline Extraction (few-shot) | None | No | Groq LLaMA | ⏳ Pilot |
| 3 | Mask Defense | Mask | No | Groq LLaMA | ⏳ Pilot |
| 4 | Symbol Replacement | replace_at_dot | No | Groq LLaMA | ⏳ Pilot |
| 5 | Prompt Injection Defense | pi_ci | No | Groq LLaMA | ⏳ Pilot |
| 6 | Hyperlink Defense | hyperlink | No | Groq LLaMA | ⏳ Pilot |
| 7 | Adaptive Attack vs pi_ci | pi_ci | Yes | Groq LLaMA | ⏳ Pilot |
| 8 | Full Matrix | All (80 configs) | Mixed | Groq LLaMA | ⏳ Planned |

### Expected Metrics
- **Accuracy**: Percentage of correctly extracted fields
- **ROUGE-1**: Token overlap score between extracted and ground truth
- **BERT Score**: Semantic similarity (optional, computed via evaluate.py)

---

## Key Implementation Details

### Data Quality
- ✅ 100 senator profiles from real U.S. Senate websites
- ✅ Ground truth manually curated from external_data/ground_truth/senate_ground_truth_updated_manual.csv
- ✅ 6 information categories (excluding jewish_heritage from original CSV)

### Task Configuration
- ✅ Task type: Text generation (info extraction)
- ✅ Dataset: senator (distinct from existing celebrity/physician)
- ✅ HTML parsing: Default parser (extracts from `<p>`, `<h1>`, `<h2>`, `<li>` tags)
- ✅ ICL strategy: 5 senator examples for few-shot learning

### Framework Compatibility
- ✅ Uses existing TaskManager and Evaluator classes
- ✅ Compatible with all defense mechanisms (MaskDefense, PromptInjectionDefense, SymbolReplacementDefense, HyperLinkDefense, NoDefense)
- ✅ Supports all LLM providers (GPT, Gemini, PaLM2, LLaMA, Vicuna, Flan, InternLM)

---

## Notes & Considerations

### Data Handling
- CSV fields like `committee_roles` contain pipe-separated values (pipe escaped as `|`)
- `education` field stored as JSON array with degree, institution, year
- Empty/missing fields preserved as `null` in JSON for accurate ground truth

### Information Privacy
- Senator profiles are public information from official U.S. Senate websites
- Ground truth includes publicly available biographical and professional data

### Performance Tuning
- **ICL Strategy**: Currently using senator examples for senator dataset. If baseline accuracy is poor, consider fallback to celebrity/physician examples.
- **GPU Memory**: LLaMA/Vicuna may require GPU specification; adjust `--gpus` flag as needed
- **API Rate Limits**: GPT-4/Gemini/PaLM2 queries may hit rate limits; use `--api_key_pos` to rotate keys

### Reproducibility
- All random seeds initialized in Model.py for deterministic results
- Seed value configurable via model config JSON
- Results saved with experiment metadata for tracking

---

## Files Created/Modified

### New Files Created
- `convert_senate_csv_to_json.py` — CSV to JSON conversion utility
- `setup_senator_icl_split.py` — ICL split setup script
- `verify_senator_setup.py` — Verification and validation script
- `configs/task_configs/senator.json` — Task configuration
- `./data/senator/labels.json` — Ground truth labels (100 records)
- `./data/senator/info_categories.txt` — Extraction field list
- `./data/senator/*.html` — 100 senator profile files
- `./data/senator/icl/*.html` — 5 ICL example files

### Files NOT Modified
- Existing PIE framework code remains unchanged
- Backward compatible with existing experiments (celebrity, physician)

---

## Troubleshooting

### Issue: TaskManager fails to load
**Solution**: Run `verify_senator_setup.py` to diagnose. Common causes:
- labels.json missing or malformed JSON
- HTML files not copied to `./data/senator/`
- Config path incorrect

### Issue: Low extraction accuracy
**Solution**: 
- Try with `--icl_num 5` (few-shot with ICL examples)
- Verify HTML parsing is extracting text properly by inspecting logs
- Consider different model (GPT-4 vs Gemini vs PaLM2)

### Issue: Memory errors with LLaMA
**Solution**: Specify available GPU with `--gpus "0"` or adjust batch size in model config

---

## Next Steps

1. **Run baseline extraction** with GPT-4 to establish performance baseline
2. **Test defense mechanisms** one by one to evaluate protection effectiveness
3. **Run adaptive attacks** against defenses to measure resilience
4. **Collect multi-model comparison** across different LLM providers
5. **Generate evaluation report** with BERT scores and comparative analysis

---

## Checkpoint & Resume Mechanism

**Two-Level Checkpointing**:
1. **Batch Level** (`extraction_checkpoint.json`):
   - Tracks which of the 10/7/80 configs have completed
   - Saved after each config finishes
   - Allows skipping completed configs on resume
   
2. **Config Level** (`.checkpoint.npz` in result dir):
   - Tracks profile progress within a single config (0-94 of 95 profiles)
   - Saved after each profile processes
   - Allows resuming mid-config without reprocessing earlier profiles

**Delete to Reset**:
- Delete `extraction_checkpoint.json` → restart entire batch from config 1
- Delete `.checkpoint.npz` in result dir → restart that config from profile 0

---

**Last Updated**: April 22, 2026
**Status**: 🔄 Phase 2 In Progress - Groq integration complete, pilot extraction running
