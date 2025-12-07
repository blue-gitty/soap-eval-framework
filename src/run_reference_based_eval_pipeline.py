#!/usr/bin/env python3
"""
Reference-Based Evaluation Pipeline for DeepScribe.
Wraps logic from playground.ipynb into a robust script.
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
        parse_clinician_soap, 
        evaluate_soap_sections, 
        to_dict
    )
    from clinical_risk import calculate_clinical_risk_score, categorize_risk
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
RESULTS_FILE = RESULTS_DIR / "reference_based_evals.json"

# Configuration
DATASET_PATH = "hf://datasets/adesouza1/soap_notes/my_dataset/train.json"

def load_existing_results():
    """Load existing results to support idempotency."""
    if RESULTS_FILE.exists():
        try:
            with open(RESULTS_FILE, 'r') as f:
                data = json.load(f)
                return data.get('notes', []), data.get('summary', {})
        except json.JSONDecodeError:
            print(f"⚠️ Warning: {RESULTS_FILE} is corrupted. Starting fresh.")
    return [], {}

def save_results(notes, summary):
    """Save results atomically to JSON."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Calculate simple running averages for summary
    if notes:
        df = pd.json_normalize(notes, sep='_')
        summary = {
            'total_notes': len(notes),
            'avg_overall_f1': float(df['metrics_overall_f1'].mean()) if 'metrics_overall_f1' in df else 0,
            'avg_semantic_sim': float(df['metrics_overall_semantic_similarity'].mean()) if 'metrics_overall_semantic_similarity' in df else 0,
            'last_updated': datetime.now().isoformat()
        }

    temp_file = RESULTS_FILE.with_suffix('.tmp')
    with open(temp_file, 'w') as f:
        json.dump({'summary': summary, 'notes': notes}, f, indent=2)
    
    # Atomic replace
    if temp_file.exists():
        if RESULTS_FILE.exists():
            os.replace(temp_file, RESULTS_FILE)
        else:
            os.rename(temp_file, RESULTS_FILE)

def run_pipeline(limit=None, id_list=None, force=False):
    """Main execution loop."""
    print("="*60)
    print("🚀 STARTING REFERENCE-BASED EVAL PIPELINE")
    print("="*60)
    
    # 1. Load Data - Optimized: check if we even need to load data first
    # (But we need the index to know what's available vs processed, so we must load)
    print(f"⏳ Loading Dataset: {DATASET_PATH}...")
    try:
        # Optimization: Only read necessary columns if dataset allows (JSON usually reads all)
        df = pd.read_json(DATASET_PATH)
        print(f"✅ Loaded {len(df)} records.")
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return

    # 2. Load Progress (Idempotency)
    existing_notes, _ = load_existing_results()
    processed_ids = {n['id'] for n in existing_notes}
    print(f"ℹ️  Found {len(processed_ids)} already processed notes.")

    # 3. Filter IDs to process
    available_ids = set(df.index)
    
    if id_list:
        # User requested specific IDs
        ids_to_process = []
        for i in id_list:
            if int(i) not in available_ids:
                print(f"⚠️  ID {i} not found in dataset. Skipping.")
                continue
            
            if int(i) in processed_ids and not force:
                print(f"⏭️  ID {i} already processed. Skipping (use --force to re-run).")
                continue
            
            ids_to_process.append(int(i))
            
        if not ids_to_process:
            print("✅ No valid new IDs to process.")
            return
        print(f"🎯 Targeting specific IDs: {ids_to_process}")
    else:
        # Default: Process all new IDs
        ids_to_process = [i for i in df.index if i not in processed_ids]
    
    if limit:
        ids_to_process = ids_to_process[:limit]
        print(f"⚠️  Limit set: processing max {limit} notes.")

    if not ids_to_process:
        print("✅ No new notes to process. Pipeline complete.")
        return

    print(f"📋 Queued {len(ids_to_process)} notes for processing.")
    
    # 4. Processing Loop
    new_results = []
    
    # If overwriting (force), remove old entries for these IDs from existing_notes
    if force and id_list:
        existing_notes = [n for n in existing_notes if n['id'] not in ids_to_process]
    
    pbar = tqdm(ids_to_process, desc="Evaluating")
    
    for note_id in pbar:
        try:
            row = df.loc[note_id]
            transcript = row['patient_convo']
            health_problem = row.get('health_problem', 'Unknown')
            clinician_note_text = row['soap_notes']

            # A. Generate Model SOAP
            model_soap = generate_and_parse_soap(transcript, health_problem)
            
            # B. Parse Clinician SOAP
            clinician_soap = parse_clinician_soap(clinician_note_text)
            
            # C. Run Evaluation (Includes Transcript Fact Extraction)
            metrics = evaluate_soap_sections(model_soap, clinician_soap, transcript=transcript)
            metrics_dict = to_dict(metrics)
            
            # D. Risk Scoring
            risk_score = calculate_clinical_risk_score(metrics_dict)
            risk_cat = categorize_risk(risk_score)
            
            # E. Construct Result Object
            result_entry = {
                'id': int(note_id),
                'health_problem': health_problem,
                'timestamp': datetime.now().isoformat(),
                'metrics': metrics_dict,
                'clinical_risk_score': round(risk_score, 4),
                'risk_category': risk_cat,
                'model_soap': model_soap,
                'clinician_soap': clinician_soap,
                'status': 'success'
            }
            
            existing_notes.append(result_entry)
            new_results.append(result_entry)
            
            # Update progress description
            pbar.set_postfix({
                'Risk': f"{risk_score:.2f}",
                'F1': f"{metrics.overall.f1:.2f}"
            })
            
            # SAVE INCREMENTALLY (Batch save every 5 for speed, or if it's the last one)
            # note_id is the index from dataset, loop variable is just the iterator
            # We track progress by length of new_results
            if len(new_results) % 5 == 0 or len(new_results) == len(ids_to_process):
                 save_results(existing_notes, {})

        except KeyboardInterrupt:
            print("\n🛑 Pipeline stopped by user.")
            break
        except Exception as e:
            print(f"\n❌ Error processing Note ID {note_id}: {e}")
            continue

    # 5. Final Report
    print("\n" + "="*60)
    print("✅ PIPELINE EXECUTION COMPLETE")
    print("="*60)
    print(f"Processed: {len(new_results)} new notes")
    print(f"Total:     {len(existing_notes)} notes available")
    print(f"Results saved to: {RESULTS_FILE}")
    
    if existing_notes:
        df_res = pd.DataFrame(existing_notes)
        print("\n--- Summary ---")
        
        if 'metrics' in df_res.columns:
             avg_f1 = df_res['metrics'].apply(lambda m: m.get('overall', {}).get('f1', 0)).mean()
             avg_sim = df_res['metrics'].apply(lambda m: m.get('overall', {}).get('semantic_similarity', 0)).mean()
             print(f"Avg Overall F1:    {avg_f1:.3f}")
             print(f"Avg Semantic Sim:  {avg_sim:.3f}")
             
        if 'clinical_risk_score' in df_res.columns:
            print(f"Avg Risk Score:    {df_res['clinical_risk_score'].mean():.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reference-Based Eval Pipeline")
    parser.add_argument("--limit", type=int, help="Limit number of notes to process")
    parser.add_argument("--id", type=int, nargs='+', help="Specific Note IDs to process")
    parser.add_argument("--force", action="store_true", help="Force re-run even if ID exists")
    args = parser.parse_args()
    
    run_pipeline(limit=args.limit, id_list=args.id, force=args.force)
