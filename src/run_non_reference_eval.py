#!/usr/bin/env python3
"""
Non-Reference Based Evaluation Pipeline for DeepScribe.
Evaluates Model SOAP vs Transcript (Ground Truth).
"""

import os
import json
import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

# Local imports
try:
    from section_eval import (
        generate_and_parse_soap,
        flatten_facts_to_list,
        semantic_match_facts
    )
    from clinical_facts import extract_facts_with_llm
except ImportError as e:
    print(f"Import Error: {e}")
    print("Ensure you are running this script from the project root or src directory.")
    exit(1)

# Setup Paths
try:
    SCRIPT_DIR = Path(__file__).parent
except NameError:
    SCRIPT_DIR = Path(os.getcwd())

if SCRIPT_DIR.name == 'src':
    PROJECT_ROOT = SCRIPT_DIR.parent
else:
    PROJECT_ROOT = SCRIPT_DIR

RESULTS_DIR = PROJECT_ROOT / "results"
OUTPUT_FILE = RESULTS_DIR / "non_reference_evals.json"
REF_EVAL_FILE = RESULTS_DIR / "reference_based_evals.json"
DATASET_PATH = "hf://datasets/adesouza1/soap_notes/my_dataset/train.json"

def calculate_non_ref_metrics(transcript_facts_list, model_facts_list):
    """
    Calculate Hallucination Rate and Coverage Rate using semantic matching.
    """
    if not model_facts_list:
        return 0.0, 0.0 # No facts in model -> 0 hallucination (good?), 0 coverage (bad)
        
    if not transcript_facts_list:
        # Transcript has no facts? weird. 
        # If model has facts, they are all hallucinations.
        return 1.0, 0.0

    # Use existing semantic matcher
    # matched_model: facts in model that match transcript
    # matched_ref: facts in transcript that match model
    # missing: facts in transcript NOT in model
    # hallucinated: facts in model NOT in transcript
    match_result = semantic_match_facts(model_facts_list, transcript_facts_list)
    
    hallucinated_count = len(match_result['hallucinated'])
    matched_transcript_count = len(match_result['matched_ref']) # Facts in transcript found in model
    
    total_model_facts = len(model_facts_list)
    total_transcript_facts = len(transcript_facts_list)
    
    # Hallucinated Rate = |Hallucinated| / |Total Model Facts|
    hallucinated_rate = hallucinated_count / max(1, total_model_facts)
    
    # Coverage Rate (Recall) = |Matched Transcript Facts| / |Total Transcript Facts|
    coverage_rate = matched_transcript_count / max(1, total_transcript_facts)
    
    return round(hallucinated_rate, 4), round(coverage_rate, 4)

def triage_for_llm_judge(non_ref_result):
    # Higher weight on hallucination because it's an active safety risk
    risk_score = (non_ref_result['hallucination_rate'] * 0.7 + 
                  (1 - non_ref_result['coverage_rate']) * 0.3)
    
    # Threshold 0.2 means:
    # - >28% hallucination triggers review (even with perfect coverage)
    # - <33% coverage triggers review (even with zero hallucination)
    return {
        'needs_judge': risk_score > 0.20,
        'risk_score': round(risk_score, 3),
        'reason': 'High Hallucination' if non_ref_result['hallucination_rate'] > 0.3 else 'Low Coverage',
        'priority': 'critical' if risk_score > 0.5 else 'high' if risk_score > 0.3 else 'medium'
    }

def llm_judge_high_risk(transcript_facts, model_facts, hallucinated_facts, missing_facts):
    """
    DeepScribe Clinical Safety Judge - Production eval for ambient AI notes.
    """
    from llm_client import query_llm
    
    prompt = f"""
You are evaluating DeepScribe ambient AI clinical notes for PRODUCTION QUALITY.

CRITICAL FAILURES (DeepScribe priorities):
1. HALLUCINATED MEDICATIONS/TESTS → Patient harm
2. MISSING CRITICAL FINDINGS → Incomplete documentation  
3. UNSUPPORTED DIAGNOSIS → Wrong treatment path
4. DANGEROUS OMISSIONS → Safety gaps

Input:
TRANSCRIPT FACTS: {transcript_facts}
MODEL FACTS: {model_facts}
HALLUCINATED BY MODEL: {hallucinated_facts}
MISSING FROM MODEL: {missing_facts}

Output STRICT JSON:

{{
  "deepscribe_issues": [
    {{
      "category": "hallucinated_medication|missing_critical_finding|unsupported_diagnosis|dangerous_omission|other",
      "statement": "Model invented 'order chest CT'",
      "risk_impact": "patient_safety|documentation_quality|treatment_delay",
      "severity": "CRITICAL|HIGH|MEDIUM",
      "fix": "Remove unsupported test OR verify in EHR"
    }}
  ],
  "production_score": 0.0-1.0,
  "recommendation": "PRODUCTION_OK|QA_REVIEW|RETRAIN_PROMPT|MODEL_ISSUE",
  "needs_clinician_review": true/false
}}

Prioritize PATIENT SAFETY > Documentation Quality.
"""
    
    try:
        return query_llm(
            messages=[
                {"role": "system", "content": "DeepScribe Production Eval Bot. Output ONLY valid JSON matching schema exactly."},
                {"role": "user", "content": prompt}
            ],
            json_mode=True,
            temperature=0.0
        )
    except Exception as e:
        return {"error": str(e), "fallback_score": 0.0}

def non_reference_eval_pipeline(limit=50, id_list=None):
    print("="*60)
    print("🚀 STARTING NON-REFERENCE EVAL PIPELINE")
    print("="*60)
    
    # 1. Load Data
    print(f"⏳ Loading Dataset...")
    try:
        df = pd.read_json(DATASET_PATH)
        print(f"✅ Loaded {len(df)} records.")
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return

    # Filter IDs
    if id_list:
        ids_to_process = [int(i) for i in id_list if int(i) in df.index]
        print(f"🎯 Targeting IDs: {ids_to_process}")
    else:
        ids_to_process = df.index[:limit].tolist()
        print(f"📋 Processing first {len(ids_to_process)} notes.")

    results = []
    
    pbar = tqdm(ids_to_process, desc="Evaluating")
    
    for note_id in pbar:
        try:
            row = df.loc[note_id]
            transcript = row['patient_convo']
            health_problem = row.get('health_problem', 'Unknown')
            
            # A. Generate Model SOAP
            # This generates the note using the LLM
            model_soap = generate_and_parse_soap(transcript, health_problem)
            
            # Combine all sections to get full model text
            model_text = f"{model_soap['subjective']} {model_soap['objective']} {model_soap['assessment']} {model_soap['plan']}"
            
            # B. Extract Facts
            # 1. Transcript Facts (Ground Truth)
            t_facts_obj = extract_facts_with_llm(transcript)
            t_facts_list = flatten_facts_to_list(t_facts_obj)
            
            # 2. Model Facts
            m_facts_obj = extract_facts_with_llm(model_text)
            m_facts_list = flatten_facts_to_list(m_facts_obj)
            
            # C. Calculate Metrics
            hallucination_rate, coverage_rate = calculate_non_ref_metrics(t_facts_list, m_facts_list)
            
            match_result = semantic_match_facts(m_facts_list, t_facts_list)
            
            # --- Triage Logic ---
            triage_result = triage_for_llm_judge({
                'hallucination_rate': hallucination_rate, 
                'coverage_rate': coverage_rate
            })
            
            judge_output = None
            if triage_result['needs_judge']:
                # Run LLM-as-a-judge
                try:
                   judge_output = llm_judge_high_risk(
                        t_facts_list, 
                        m_facts_list, 
                        list(match_result['hallucinated']), 
                        list(match_result['missing'])
                    )
                except Exception as e:
                    print(f"Error in LLM judge: {e}")
            
            results.append({
                'id': int(note_id),
                'health_problem': health_problem,
                'hallucination_rate': hallucination_rate,
                'coverage_rate': coverage_rate,
                'triage': triage_result,
                'judge_evaluation': judge_output,  # New field
                'fact_counts': {
                    'transcript': len(t_facts_list),
                    'model': len(m_facts_list)
                },
                'details': {
                    'transcript_facts': t_facts_list,
                    'model_facts': m_facts_list,
                    'hallucinated_facts': list(match_result['hallucinated']),
                    'missing_facts': list(match_result['missing']),
                    'matched_transcript_facts': list(match_result['matched_ref'])
                }
            })
            
            pbar.set_postfix({
                'Halluc': f"{hallucination_rate:.2f}",
                'Cov': f"{coverage_rate:.2f}",
                'Judge': '✅' if triage_result['needs_judge'] else '⏭️'
            })
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error processing {note_id}: {e}")
            continue

    # Save Results
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Results saved to {OUTPUT_FILE}")
    
    # Group by Health Problem
    df_res = pd.DataFrame(results)
    if not df_res.empty:
        print("\n" + "="*50)
        print("📊 METRICS BY HEALTH PROBLEM")
        print("="*50)
        
        grouped = df_res.groupby('health_problem')[['hallucination_rate', 'coverage_rate']].mean()
        print(grouped)
        
        # Comparison with Reference-Based (if available)
        if REF_EVAL_FILE.exists():
            print("\n" + "="*50)
            print("🆚 COMPARISON VS REFERENCE-BASED")
            print("="*50)
            try:
                with open(REF_EVAL_FILE, 'r') as f:
                    ref_data = json.load(f)
                    ref_notes = ref_data.get('notes', [])
                    if ref_notes:
                        df_ref = pd.DataFrame(ref_notes)
                        
                        df_ref['ref_halluc_rate'] = df_ref['metrics'].apply(lambda x: x['overall']['hallucinated_rate'])
                        df_ref['ref_f1'] = df_ref['metrics'].apply(lambda x: x['overall']['f1'])
                        
                        avg_ref_halluc = df_ref['ref_halluc_rate'].mean()
                        avg_nonref_halluc = df_res['hallucination_rate'].mean()
                        
                        print(f"{'Metric':<25} | {'Non-Reference':<15} | {'Reference-Based':<15}")
                        print("-" * 60)
                        print(f"{'Avg Hallucination Rate':<25} | {avg_nonref_halluc:<15.4f} | {avg_ref_halluc:<15.4f}")
                        # Fixed the formatting error from previous attempt
                        avg_coverage = df_res['coverage_rate'].mean()
                        avg_ref_f1 = df_ref['ref_f1'].mean()
                        print(f"{'Avg Coverage/Recall':<25} | {avg_coverage:<15.4f} | {avg_ref_f1:<15.4f} (F1)")
            except Exception as e:
                print(f"Could not load reference comparison: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=50, help="Number of notes to process")
    parser.add_argument("--id", type=int, nargs='+', help="Specific IDs")
    args = parser.parse_args()
    
    non_reference_eval_pipeline(limit=args.limit, id_list=args.id)
