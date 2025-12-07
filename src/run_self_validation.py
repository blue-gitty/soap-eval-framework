#!/usr/bin/env python3
"""
Self-Validation Evaluation Pipeline for DeepScribe.
Validates extraction reliability by comparing Transcript vs Gold (Clinician SOAP).
Compares against Model-Gold results to confirm pipeline reliability.
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
        parse_clinician_soap,
        flatten_facts_to_list,
        semantic_match_facts,
        compute_semantic_similarity
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
OUTPUT_FILE = RESULTS_DIR / "self_validation_evals.json"
REF_EVAL_FILE = RESULTS_DIR / "reference_based_evals.json"

DATASET_PATH = "hf://datasets/adesouza1/soap_notes/my_dataset/train.json"

def compute_transcript_gold_metrics(transcript_facts_list, gold_facts_list, transcript_text, gold_text):
    """
    Compute all reference metrics for Transcript vs Gold comparison.
    
    Returns:
        Dict with missing_rate, hallucinated_rate, precision, recall, f1, semantic_similarity
    """
    if not transcript_facts_list and not gold_facts_list:
        return {
            'missing_rate': 0.0,
            'hallucinated_rate': 0.0,
            'precision': 1.0,
            'recall': 1.0,
            'f1': 1.0,
            'semantic_similarity': 0.0
        }
    
    if not gold_facts_list:
        # Gold has no facts - all transcript facts are "hallucinated" (not in gold)
        return {
            'missing_rate': 0.0,
            'hallucinated_rate': 1.0 if transcript_facts_list else 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'semantic_similarity': compute_semantic_similarity(transcript_text, gold_text)
        }
    
    if not transcript_facts_list:
        # Transcript has no facts - all gold facts are missing
        return {
            'missing_rate': 1.0,
            'hallucinated_rate': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'semantic_similarity': compute_semantic_similarity(transcript_text, gold_text)
        }
    
    # Use semantic matching
    match_result = semantic_match_facts(transcript_facts_list, gold_facts_list)
    
    matched_count = len(match_result['matched_ref'])
    missing_count = len(match_result['missing'])
    hallucinated_count = len(match_result['hallucinated'])
    
    total_gold = len(gold_facts_list)
    total_transcript = len(transcript_facts_list)
    
    # Calculate metrics
    missing_rate = missing_count / max(1, total_gold)
    hallucinated_rate = hallucinated_count / max(1, total_transcript)
    precision = matched_count / max(1, total_transcript)
    recall = matched_count / max(1, total_gold)
    f1 = 2 * precision * recall / max(0.001, precision + recall)
    
    semantic_sim = compute_semantic_similarity(transcript_text, gold_text)
    
    return {
        'missing_rate': round(missing_rate, 4),
        'hallucinated_rate': round(hallucinated_rate, 4),
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1': round(f1, 4),
        'semantic_similarity': round(semantic_sim, 4)
    }

def evaluate_transcript_gold_overall(transcript, clinician_soap):
    """
    Evaluate Transcript vs Gold SOAP - Overall metrics only.
    
    Compares full transcript facts vs full gold SOAP facts.
    Returns overall metrics: missing_rate, hallucinated_rate, f1, semantic_similarity
    """
    # Extract transcript facts (full transcript)
    transcript_facts = extract_facts_with_llm(transcript)
    transcript_facts_list = flatten_facts_to_list(transcript_facts)
    
    # Extract gold facts from full clinician SOAP
    gold_text = f"{clinician_soap['subjective']} {clinician_soap['objective']} {clinician_soap['assessment']} {clinician_soap['plan']}"
    gold_facts = extract_facts_with_llm(gold_text)
    gold_facts_list = flatten_facts_to_list(gold_facts)
    
    # Compute Overall Metrics: Full Transcript vs Full Gold SOAP
    overall_metrics = compute_transcript_gold_metrics(
        transcript_facts_list,
        gold_facts_list,
        transcript,
        gold_text
    )
    
    return overall_metrics

def self_validation_pipeline(limit=20, id_list=None):
    """
    Main self-validation pipeline.
    Compares Transcript vs Gold extraction to validate pipeline reliability.
    """
    print("="*60)
    print("🔍 STARTING SELF-VALIDATION PIPELINE")
    print("="*60)
    print("Validating extraction reliability: Transcript vs Gold (Clinician SOAP)")
    print("="*60)
    
    # 1. Load Data
    print(f"⏳ Loading Dataset...")
    try:
        df = pd.read_json(DATASET_PATH)
        print(f"✅ Loaded {len(df)} records.")
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return
    
    # 2. Filter IDs
    if id_list:
        ids_to_process = [int(i) for i in id_list if int(i) in df.index]
        print(f"🎯 Targeting IDs: {ids_to_process}")
    else:
        ids_to_process = df.index[:limit].tolist()
        print(f"📋 Processing first {len(ids_to_process)} notes for validation.")
    
    if not ids_to_process:
        print("❌ No valid IDs to process.")
        return
    
    # 3. Load Reference-Based Results for Comparison
    model_gold_results = {}
    if REF_EVAL_FILE.exists():
        try:
            with open(REF_EVAL_FILE, 'r') as f:
                ref_data = json.load(f)
                ref_notes = ref_data.get('notes', []) if isinstance(ref_data, dict) else ref_data
                for note in ref_notes:
                    model_gold_results[note['id']] = note
            print(f"✅ Loaded {len(model_gold_results)} Model-Gold results for comparison.")
        except Exception as e:
            print(f"⚠️  Could not load reference results: {e}")
    else:
        print("⚠️  Reference-based results not found. Will only show Transcript-Gold metrics.")
    
    # 4. Processing Loop
    results = []
    
    pbar = tqdm(ids_to_process, desc="Validating")
    
    for note_id in pbar:
        try:
            row = df.loc[note_id]
            transcript = row['patient_convo']
            health_problem = row.get('health_problem', 'Unknown')
            clinician_note_text = row['soap_notes']
            
            # A. Parse Clinician SOAP
            clinician_soap = parse_clinician_soap(clinician_note_text)
            
            # B. Evaluate Overall Metrics: Transcript vs Gold SOAP
            overall_metrics = evaluate_transcript_gold_overall(transcript, clinician_soap)
            
            # C. Construct Result Object
            result_entry = {
                'id': int(note_id),
                'health_problem': health_problem,
                'timestamp': datetime.now().isoformat(),
                'missing_rate': overall_metrics['missing_rate'],
                'hallucinated_rate': overall_metrics['hallucinated_rate'],
                'f1': overall_metrics['f1'],
                'semantic_similarity': overall_metrics['semantic_similarity'],
                'precision': overall_metrics['precision'],
                'recall': overall_metrics['recall'],
                'status': 'success'
            }
            
            results.append(result_entry)
            
            # Update progress
            pbar.set_postfix({
                'Missing': f"{overall_metrics['missing_rate']:.2f}",
                'Halluc': f"{overall_metrics['hallucinated_rate']:.2f}",
                'F1': f"{overall_metrics['f1']:.2f}"
            })
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\n❌ Error processing Note ID {note_id}: {e}")
            continue
    
    # 5. Save Results (JSON format like reference-based)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    # Calculate summary
    if results:
        df_res = pd.DataFrame(results)
        summary = {
            'total_notes': len(results),
            'avg_missing_rate': float(df_res['missing_rate'].mean()),
            'avg_hallucinated_rate': float(df_res['hallucinated_rate'].mean()),
            'avg_f1': float(df_res['f1'].mean()),
            'avg_semantic_similarity': float(df_res['semantic_similarity'].mean()),
            'avg_precision': float(df_res['precision'].mean()),
            'avg_recall': float(df_res['recall'].mean()),
            'last_updated': datetime.now().isoformat()
        }
    else:
        summary = {}
    
    with open(OUTPUT_FILE, 'w') as f:
        json.dump({'summary': summary, 'notes': results}, f, indent=2)
    print(f"\n✅ Results saved to {OUTPUT_FILE}")
    
    # 6. Generate Comparison Table (console output only)
    if results:
        print_comparison_table(results, model_gold_results)
    
    print("\n" + "="*60)
    print("✅ SELF-VALIDATION PIPELINE COMPLETE")
    print("="*60)

def print_comparison_table(transcript_gold_results, model_gold_results):
    """
    Print comparison table to console (no file outputs).
    Shows overall metrics: missing_rate, hallucinated_rate, f1, semantic_similarity
    """
    print("\n" + "="*60)
    print("📊 GENERATING COMPARISON REPORT")
    print("="*60)
    
    # Calculate averages for Transcript-Gold
    df_tg = pd.DataFrame(transcript_gold_results)
    
    tg_avg_missing = df_tg['missing_rate'].mean()
    tg_avg_halluc = df_tg['hallucinated_rate'].mean()
    tg_avg_f1 = df_tg['f1'].mean()
    tg_avg_sim = df_tg['semantic_similarity'].mean()
    
    # Calculate averages for Model-Gold (if available)
    mg_avg_missing = None
    mg_avg_halluc = None
    mg_avg_f1 = None
    mg_avg_sim = None
    
    if model_gold_results:
        mg_notes = [v for k, v in model_gold_results.items() if k in [r['id'] for r in transcript_gold_results]]
        if mg_notes:
            df_mg = pd.json_normalize(mg_notes, sep='_')
            mg_avg_missing = df_mg['metrics_overall_missing_rate'].mean()
            mg_avg_halluc = df_mg['metrics_overall_hallucinated_rate'].mean()
            mg_avg_f1 = df_mg['metrics_overall_f1'].mean()
            mg_avg_sim = df_mg['metrics_overall_semantic_similarity'].mean()
    
    # Create comparison table - Only key metrics
    metrics_data = []
    
    metrics = [
        ('missing_rate', 'Missing Rate'),
        ('hallucinated_rate', 'Hallucinated Rate'),
        ('f1', 'F1 Score'),
        ('semantic_similarity', 'Semantic Similarity')
    ]
    
    tg_values = [tg_avg_missing, tg_avg_halluc, tg_avg_f1, tg_avg_sim]
    mg_values = [mg_avg_missing, mg_avg_halluc, mg_avg_f1, mg_avg_sim] if mg_avg_missing is not None else [None] * 4
    
    # Threshold check: ALL < 3% for extraction reliability
    threshold = 0.03
    all_below_threshold = all(v < threshold for v in [tg_avg_missing, tg_avg_halluc])
    
    for (key, label), tg_val, mg_val in zip(metrics, tg_values, mg_values):
        # For missing_rate and hallucinated_rate, check < 3% and format as percentage
        # For f1 and semantic_similarity, show as decimal
        if key in ['missing_rate', 'hallucinated_rate']:
            below_threshold = tg_val < threshold
            tg_display = f"{tg_val*100:.1f}%" if tg_val is not None else "N/A"
            mg_display = f"{mg_val*100:.1f}%" if mg_val is not None else "N/A"
        else:
            below_threshold = None  # Not applicable for other metrics
            tg_display = f"{tg_val:.2f}" if tg_val is not None else "N/A"
            mg_display = f"{mg_val:.2f}" if mg_val is not None else "N/A"
        
        threshold_display = 'TRUE' if below_threshold else ('FALSE' if below_threshold is False else 'N/A')
        
        metrics_data.append({
            'Metric': label,
            'Transcript-Gold': tg_display,
            'Model-Gold': mg_display,
            '✅ <3%': threshold_display
        })
    
    df_comparison = pd.DataFrame(metrics_data)
    
    # Print table
    print("\n" + "="*80)
    print("COMPARISON TABLE: Transcript-Gold vs Model-Gold")
    print("="*80)
    print(df_comparison.to_string(index=False))
    print("="*80)
    
    # Interpretation
    print("\n" + "="*80)
    print("📋 INTERPRETATION")
    print("="*80)
    
    if all_below_threshold:
        max_rate = max(tg_avg_missing, tg_avg_halluc) * 100
        print(f"✅ EXTRACTION RELIABLE: Transcript-Gold {max_rate:.1f}% confirms reliable pipeline")
        print(f"   Missing Rate: {tg_avg_missing*100:.1f}%, Hallucinated Rate: {tg_avg_halluc*100:.1f}% (both < 3%)")
        print(f"   F1: {tg_avg_f1:.2f}, Semantic Similarity: {tg_avg_sim:.2f}")
    else:
        print(f"⚠️  EXTRACTION NEEDS REVIEW: Some metrics exceed 3% threshold")
        if tg_avg_missing >= threshold:
            print(f"   - Missing Rate: {tg_avg_missing*100:.1f}% (threshold: 3%)")
        if tg_avg_halluc >= threshold:
            print(f"   - Hallucinated Rate: {tg_avg_halluc*100:.1f}% (threshold: 3%)")
    
    if mg_avg_missing is not None:
        print(f"\n📊 Model-Gold Comparison:")
        print(f"   - Transcript-Gold Missing Rate: {tg_avg_missing*100:.1f}%")
        print(f"   - Model-Gold Missing Rate: {mg_avg_missing*100:.1f}%")
        print(f"   - Transcript-Gold Hallucinated Rate: {tg_avg_halluc*100:.1f}%")
        print(f"   - Model-Gold Hallucinated Rate: {mg_avg_halluc*100:.1f}%")
    
    print("="*80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Self-Validation Eval Pipeline")
    parser.add_argument("--limit", type=int, default=20, help="Number of notes to process (default: 20)")
    parser.add_argument("--id", type=int, nargs='+', help="Specific Note IDs to process")
    args = parser.parse_args()
    
    self_validation_pipeline(limit=args.limit, id_list=args.id)

