# Code Cleanup Analysis

## Unnecessary Code Identified

### 1. Unused Imports in Pipelines

**`run_non_reference_eval.py`:**
- ❌ `import numpy as np` - Never used
- ❌ `from collections import defaultdict` - Never used

**`run_reference_based_eval_pipeline.py`:**
- ❌ `RISK_FILE` variable - Defined but never used (line 43)

### 2. Unused Modules (Not Imported by Any Pipeline)

**`src/eval_metrics.py`:**
- ❌ Entire module unused - Only commented out in playground.ipynb
- Contains: `calculate_rouge_scores`, `calculate_length_ratio`, `evaluate_note`
- **Decision**: Can be removed or kept for future use

**`src/semantic_eval.py`:**
- ❌ Entire module unused - Only commented out in playground.ipynb
- Contains: `SemanticEvaluator` class
- **Decision**: Can be removed or kept for future use
- **Note**: We already have semantic similarity in `section_eval.py` via embeddings

### 3. Utility Scripts (Not Required for Pipeline Execution)

**`src/inspect_results.py`:**
- ⚠️ Utility script for inspecting results
- Not imported by any pipeline
- **Decision**: Keep (useful for debugging/analysis)

**`src/viz_utils.py`:**
- ⚠️ Visualization utilities
- Not imported by any pipeline
- **Decision**: Keep (useful for reporting, but not required for pipeline execution)

## Recommendations

### High Priority (Safe to Remove)
1. Remove unused imports from `run_non_reference_eval.py`
2. Remove unused `RISK_FILE` from `run_reference_based_eval_pipeline.py`

### Medium Priority (Consider Removing)
1. Remove `eval_metrics.py` if ROUGE scores aren't needed
2. Remove `semantic_eval.py` if not using (we have better semantic eval in `section_eval.py`)

### Low Priority (Keep)
1. Keep `inspect_results.py` - useful utility
2. Keep `viz_utils.py` - useful for reporting

## Files Actually Used by Pipelines

### Reference-Based Pipeline Uses:
- `section_eval.py` ✅
- `clinical_risk.py` ✅
- `generate_model_note.py` (via section_eval) ✅
- `clinical_facts.py` (via section_eval) ✅
- `embeddings.py` (via section_eval) ✅
- `llm_client.py` (via generate_model_note) ✅
- `config_loader.py` (via various modules) ✅

### Non-Reference Pipeline Uses:
- `section_eval.py` ✅
- `clinical_facts.py` ✅
- `llm_client.py` ✅
- `embeddings.py` (via section_eval) ✅
- `config_loader.py` (via various modules) ✅

### Self-Validation Pipeline Uses:
- `section_eval.py` ✅
- `clinical_facts.py` ✅
- `embeddings.py` (via section_eval) ✅
- `config_loader.py` (via various modules) ✅

