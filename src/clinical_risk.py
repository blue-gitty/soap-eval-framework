"""
Clinical Risk Scoring for SOAP Note Evaluation.
Computes weighted risk score prioritizing patient safety.
"""

import json
import pandas as pd
from typing import Dict, List, Optional
from pathlib import Path
import os

# Get the directory where this module is located
_MODULE_DIR = Path(__file__).parent
_PROJECT_ROOT = _MODULE_DIR.parent
_DEFAULT_RESULTS_PATH = _PROJECT_ROOT / "results" / "reference_based_evals.json"


def calculate_clinical_risk_score(metrics: Dict) -> float:
    """
    Calculate clinical risk score from section metrics.
    
    Weights prioritize patient safety:
    - Plan: 50% (treatment errors = patient harm)
    - Objective: 25% (wrong vitals/labs = misdiagnosis)
    - Assessment: 15% (wrong diagnosis = wrong treatment)
    - Subjective: 10% (least immediate safety impact)
    
    Hallucinations get 1.5x penalty (fabricated info is worse than missing).
    
    Args:
        metrics: Dict with section metrics (from evaluate_soap_sections)
        
    Returns:
        Risk score between 0-1 (higher = more concerning)
    """
    risk_score = (
        # Plan (highest weight - patient safety)
        0.35 * metrics['plan']['missing_rate'] +
        0.15 * metrics['plan']['hallucinated_rate'] * 1.5 +
        
        # Objective (clinical accuracy)
        0.15 * metrics['objective']['missing_rate'] +
        0.10 * metrics['objective']['hallucinated_rate'] * 1.5 +
        
        # Assessment (diagnostic accuracy)
        0.10 * metrics['assessment']['missing_rate'] +
        0.05 * metrics['assessment']['hallucinated_rate'] +
        
        # Subjective (lowest direct patient impact)
        0.05 * metrics['subjective']['missing_rate'] +
        0.05 * metrics['subjective']['hallucinated_rate']
    )
    
    return min(1.0, max(0.0, risk_score))  # Clamp to 0-1


def categorize_risk(score: float) -> str:
    """Categorize risk score into actionable levels."""
    if score < 0.2:
        return "LOW"
    elif score < 0.4:
        return "MODERATE"
    elif score < 0.6:
        return "HIGH"
    else:
        return "CRITICAL"


def get_risk_emoji(category: str) -> str:
    """Get emoji for risk category."""
    return {
        "LOW": "🟢",
        "MODERATE": "🟡",
        "HIGH": "🟠",
        "CRITICAL": "🔴"
    }.get(category, "⚪")



# Note: File loading and reporting functions removed to keep module focused on logic.
# Use run_reference_based_eval_pipeline.py for execution.
