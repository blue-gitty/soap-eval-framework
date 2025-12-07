#!/usr/bin/env python3
"""
Master Evaluation Suite for DeepScribe.

Simple 4-step pipeline:
1. Run reference-based evaluation → reference_based_evals.json
2. Run non-reference evaluation → non_reference_evals.json
3. Run self-validation → self_validation_evals.json
4. Generate metrics, visualizations, and reports
"""

import sys
import argparse
import shutil
from pathlib import Path

# Add src to path
SCRIPT_DIR = Path(__file__).parent
SRC_DIR = SCRIPT_DIR / "src"
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))
else:
    print(f"❌ Error: src directory not found at {SRC_DIR}")
    sys.exit(1)

# Import pipeline functions
try:
    from run_reference_based_eval_pipeline import run_pipeline as run_ref_pipeline
    from run_non_reference_eval import non_reference_eval_pipeline
    from run_self_validation import self_validation_pipeline
    from viz_utils import run_dashboard, run_validation
    from config_loader import load_config
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print(f"   Make sure you're running from the project root directory")
    print(f"   Expected src directory at: {SRC_DIR}")
    sys.exit(1)

def get_model_info():
    """Get current model configuration."""
    config = load_config()
    llm_config = config.get('llm', {})
    provider = llm_config.get('provider', 'unknown')
    model = llm_config.get('model', 'unknown')
    return provider, model

def check_processed_results():
    """Check if processed results exist in results/processed/ folder."""
    processed_dir = SCRIPT_DIR / "results" / "processed"
    files = {
        'ref': processed_dir / "reference_based_evals.json",
        'non_ref': processed_dir / "non_reference_evals.json",
        'self_val': processed_dir / "self_validation_evals.json"
    }
    
    all_exist = all(path.exists() for path in files.values())
    return all_exist, files

def copy_processed_results():
    """Copy processed results from results/processed/ to results/."""
    processed_dir = SCRIPT_DIR / "results" / "processed"
    results_dir = SCRIPT_DIR / "results"
    
    files_to_copy = {
        'ref': ("reference_based_evals.json", "Reference-based"),
        'non_ref': ("non_reference_evals.json", "Non-reference"),
        'self_val': ("self_validation_evals.json", "Self-validation")
    }
    
    results_dir.mkdir(parents=True, exist_ok=True)
    
    copied = []
    for key, (filename, label) in files_to_copy.items():
        src = processed_dir / filename
        dst = results_dir / filename
        
        if src.exists():
            shutil.copy2(src, dst)
            copied.append(label)
            print(f"  ✅ Copied {label}: {filename}")
        else:
            print(f"  ⚠️  Missing {label}: {filename}")
    
    return len(copied) == 3

def verify_results_exist():
    """Verify all three result files exist in results/."""
    results_dir = SCRIPT_DIR / "results"
    files = {
        'ref': results_dir / "reference_based_evals.json",
        'non_ref': results_dir / "non_reference_evals.json",
        'self_val': results_dir / "self_validation_evals.json"
    }
    
    missing = [name for name, path in files.items() if not path.exists()]
    if missing:
        print(f"⚠️  Missing result files: {', '.join(missing)}")
        return False
    return True


def run_full_eval_suite(limit=30, generate_charts=True, id_list=None, use_processed=False, force=False):
    """
    Run complete evaluation suite in 4 simple steps:
    1. Reference-based evaluation → reference_based_evals.json
    2. Non-reference evaluation → non_reference_evals.json
    3. Self-validation → self_validation_evals.json
    4. Generate metrics, visualizations, and reports
    
    If use_processed=True or processed results exist, skip pipeline execution
    and use pre-existing results from results/processed/.
    """
    print("\n" + "="*80)
    print("🚀 DEEPSCRIBE FULL EVALUATION SUITE")
    print("="*80)
    
    # Get model info
    provider, model = get_model_info()
    print(f"\n📋 Model: {provider}/{model}")
    
    # Check for processed results
    processed_exist, processed_files = check_processed_results()
    
    # Determine if we should use processed results
    # Logic: Use processed if:
    #   - --use-processed flag is set, OR
    #   - Processed results exist AND not --force AND no --limit specified (auto-detect)
    user_wants_pipeline = limit != 30 or id_list is not None  # User explicitly set limit or IDs
    should_use_processed = use_processed or (processed_exist and not force and not user_wants_pipeline)
    
    if should_use_processed and processed_exist:
        print("\n" + "="*80)
        print("📦 USING PRE-EXISTING PROCESSED RESULTS")
        print("="*80)
        print("→ Copying from results/processed/ to results/")
        print("→ Skipping pipeline execution (fast mode)")
        
        if copy_processed_results():
            print("\n✅ All processed results copied successfully!")
            print("→ Proceeding to Step 4: Generate Metrics & Visualizations")
        else:
            print("\n⚠️  Some processed results missing. Falling back to full pipeline...")
            should_use_processed = False
    
    if not should_use_processed:
        # Run full pipeline
        print(f"\n📋 Limit: {limit} notes per pipeline")
        if id_list:
            print(f"📋 ID List: {id_list}")
        
        # ========================================================================
        # STEP 1: Reference-Based Evaluation
        # ========================================================================
        print("\n" + "="*80)
        print("📊 STEP 1: Reference-Based Evaluation")
        print("="*80)
        print("→ Generating: results/reference_based_evals.json")
        try:
            run_ref_pipeline(limit=limit, id_list=id_list, force=force)
            print("✅ Step 1 complete: reference_based_evals.json")
        except Exception as e:
            print(f"❌ Error in Step 1: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # ========================================================================
        # STEP 2: Non-Reference Evaluation
        # ========================================================================
        print("\n" + "="*80)
        print("📊 STEP 2: Non-Reference Evaluation")
        print("="*80)
        print("→ Generating: results/non_reference_evals.json")
        try:
            non_reference_eval_pipeline(limit=limit, id_list=id_list)
            print("✅ Step 2 complete: non_reference_evals.json")
        except Exception as e:
            print(f"❌ Error in Step 2: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # ========================================================================
        # STEP 3: Self-Validation
        # ========================================================================
        print("\n" + "="*80)
        print("📊 STEP 3: Self-Validation")
        print("="*80)
        print("→ Generating: results/self_validation_evals.json")
        try:
            # Use same limit for self-validation (user can control via --limit)
            self_validation_pipeline(limit=limit, id_list=id_list)
            print("✅ Step 3 complete: self_validation_evals.json")
        except Exception as e:
            print(f"❌ Error in Step 3: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # ========================================================================
    # STEP 4: Generate Metrics, Visualizations, and Reports
    # ========================================================================
    print("\n" + "="*80)
    print("📊 STEP 4: Generate Metrics & Visualizations")
    print("="*80)
    
    # Verify all results exist
    if not verify_results_exist():
        print("⚠️  Some result files are missing. Continuing anyway...")
    
    # Generate charts
    if generate_charts:
        print("\n📈 Generating executive charts...")
        try:
            run_dashboard()
            print("✅ Charts generated in reports/executive_dashboard/")
        except Exception as e:
            print(f"⚠️  Error generating charts: {e}")
            import traceback
            traceback.print_exc()
    
    # Run meta-analysis
    print("\n🔬 Running meta-analysis...")
    try:
        run_validation()
        print("✅ Meta-analysis complete: framework_validation_metrics.json")
    except Exception as e:
        print(f"⚠️  Error in meta-analysis: {e}")
        import traceback
        traceback.print_exc()
    
    # Final Summary
    print("\n" + "="*80)
    print("✅ EVALUATION SUITE COMPLETE")
    print("="*80)
    print(f"\n📁 Output Locations:")
    print(f"  • Results: results/")
    print(f"    - reference_based_evals.json")
    print(f"    - non_reference_evals.json")
    print(f"    - self_validation_evals.json")
    print(f"  • Reports: reports/")
    if generate_charts:
        print(f"    - executive_dashboard/ (charts)")
    print(f"    - framework_validation_metrics.json (meta-analysis)")
    print(f"\n📋 Model Used: {provider}/{model}")
    print("="*80)
    
    return True

def main():
    parser = argparse.ArgumentParser(
        description="Run complete DeepScribe evaluation suite (4-step pipeline)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Pipeline Steps:
  1. Reference-based evaluation → reference_based_evals.json
  2. Non-reference evaluation → non_reference_evals.json
  3. Self-validation → self_validation_evals.json
  4. Generate metrics, visualizations, and reports

Examples:
  # Fast mode: Use pre-existing processed results
  python run_full_eval_suite.py --use-processed --charts
  
  # Full pipeline: Run all evaluations (slow)
  python run_full_eval_suite.py --limit 30 --charts
  
  # Force regeneration even if processed exists
  python run_full_eval_suite.py --limit 30 --force --charts
  
  # Auto-detect: Uses processed if available, otherwise runs pipeline
  python run_full_eval_suite.py --charts

Note: --limit applies to ALL 3 pipelines (ref, non-ref, self-val)
      Model is determined by config.yaml
      Processed results are in results/processed/ (3 JSON files)
        """
    )
    
    parser.add_argument(
        '--limit',
        type=int,
        default=30,
        help='Number of notes to evaluate per pipeline (default: 30). Applies to all 3 pipelines.'
    )
    
    parser.add_argument(
        '--charts',
        action='store_true',
        help='Generate executive charts and visualizations (default: False)'
    )
    
    parser.add_argument(
        '--id-list',
        type=int,
        nargs='+',
        help='Specific note IDs to evaluate (optional, overrides --limit)'
    )
    
    parser.add_argument(
        '--use-processed',
        action='store_true',
        help='Use pre-existing results from results/processed/ (skip pipeline execution)'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force regeneration even if processed results exist'
    )
    
    args = parser.parse_args()
    
    # Run the suite
    success = run_full_eval_suite(
        limit=args.limit,
        generate_charts=args.charts,
        id_list=args.id_list,
        use_processed=args.use_processed,
        force=args.force
    )
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())

