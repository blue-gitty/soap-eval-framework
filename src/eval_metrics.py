"""
Evaluation metrics for clinical notes.
Deterministic and fast metrics for quality assessment.
"""

from rouge_score import rouge_scorer
from typing import Dict, List
import json


def calculate_rouge_scores(reference: str, generated: str) -> Dict[str, float]:
    """Calculate ROUGE scores between reference and generated text."""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, generated)
    
    return {
        'rouge1': scores['rouge1'].fmeasure,
        'rouge2': scores['rouge2'].fmeasure,
        'rougeL': scores['rougeL'].fmeasure
    }


def calculate_length_ratio(reference: str, generated: str) -> Dict[str, any]:
    """Calculate length-based metrics."""
    ref_len = len(reference)
    gen_len = len(generated)
    
    ratio = gen_len / ref_len if ref_len > 0 else 0
    
    if 0.7 <= ratio <= 1.3:
        score = 1.0
    elif 0.5 <= ratio <= 1.5:
        score = 0.8
    else:
        score = 0.5
    
    return {
        'reference_length': ref_len,
        'generated_length': gen_len,
        'length_ratio': ratio,
        'length_score': score
    }


def evaluate_note(
    transcript: str,
    generated_note: str,
    ground_truth_note: str,
    transcript_entities: Dict = None,
    generated_entities: Dict = None
) -> Dict[str, any]:
    """Run all fast evaluation metrics on a single note."""
    results = {}
    
    results['rouge'] = calculate_rouge_scores(ground_truth_note, generated_note)
    results['length'] = calculate_length_ratio(ground_truth_note, generated_note)
    
    scores = [
        results['rouge']['rougeL'],
        results['length']['length_score']
    ]
    
    results['overall_score'] = sum(scores) / len(scores)
    
    return results
