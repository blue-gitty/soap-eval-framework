import json
import pandas as pd
from pathlib import Path

# Paths
REF_FILE = Path("results/reference_based_evals.json")
NON_REF_FILE = Path("results/non_reference_evals.json")

def peek_json_structure(filepath, name):
    print(f"\n{'='*50}")
    print(f"🧐 Inspecting: {name}")
    print(f"{'='*50}")
    
    if not filepath.exists():
        print(f"❌ File not found: {filepath}")
        return

    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        # Handle list vs dict structure
        notes = data.get('notes', []) if isinstance(data, dict) else data
        
        if not notes:
            print("⚠️ No notes found in file.")
            return

        print(f"✅ Total Records: {len(notes)}")
        
        # Take the first record to show structure
        first_record = notes[0]
        
        print("\n🔑 Available Keys (per ID):")
        for key in first_record.keys():
            print(f" - {key}")
            
        # Deep dive into 'metrics' if it exists (Reference-based)
        if 'metrics' in first_record:
            print("\n📊 Metrics Structure (inside 'metrics'):")
            m = first_record['metrics']
            print(f"   Sections: {list(m.keys())}")
            if 'overall' in m:
                print(f"   Fields in 'overall': {list(m['overall'].keys())}")
                
        # Deep dive into 'triage' or 'judge_evaluation' (Non-Reference)
        if 'triage' in first_record:
             print("\n⚖️  Triage Structure:")
             print(f"   {list(first_record['triage'].keys())}")

    except Exception as e:
        print(f"Error reading file: {e}")

if __name__ == "__main__":
    peek_json_structure(REF_FILE, "Reference-Based Evals")
    peek_json_structure(NON_REF_FILE, "Non-Reference Evals")
