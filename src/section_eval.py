"""
Section-level SOAP evaluation metrics.
Computes missing facts and hallucinations per SOAP section.
"""

import json
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict


@dataclass
class SectionMetrics:
    """Metrics for a single SOAP section."""
    missing_count: int = 0
    missing_rate: float = 0.0
    hallucinated_count: int = 0
    hallucinated_rate: float = 0.0
    matched_count: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    semantic_similarity: float = 0.0  # Simple text-level cosine similarity


@dataclass
class SOAPMetrics:
    """Complete metrics for all SOAP sections."""
    subjective: SectionMetrics
    objective: SectionMetrics
    assessment: SectionMetrics
    plan: SectionMetrics
    overall: SectionMetrics


def generate_and_parse_soap(transcript: str, health_problem: str = None) -> Dict[str, str]:
    """
    Generate SOAP note and return parsed dict with guaranteed keys.
    
    Returns:
        Dict with keys: subjective, objective, assessment, plan
        All values are strings (empty string if not available)
    """
    from generate_model_note import generate_model_note
    
    try:
        soap_dict = generate_model_note(transcript, health_problem=health_problem)
        
        # Ensure all keys exist with string values
        return {
            'subjective': str(soap_dict.get('subjective', '')),
            'objective': str(soap_dict.get('objective', '')),
            'assessment': str(soap_dict.get('assessment', '')),
            'plan': str(soap_dict.get('plan', ''))
        }
    except Exception as e:
        print(f"Error generating SOAP: {e}")
        return {
            'subjective': '',
            'objective': '',
            'assessment': '',
            'plan': ''
        }


def parse_clinician_soap(soap_text: str) -> Dict[str, str]:
    """
    Parse clinician's SOAP note text into sections.
    
    Handles format like:
    Subjective:
    [text...]
    
    Objective:
    [text...]
    """
    sections = {
        'subjective': '',
        'objective': '',
        'assessment': '',
        'plan': ''
    }
    
    # Normalize text
    text = soap_text.strip()
    
    # Find section boundaries
    section_markers = ['Subjective:', 'Objective:', 'Assessment:', 'Plan:']
    positions = []
    
    for marker in section_markers:
        pos = text.find(marker)
        if pos != -1:
            positions.append((pos, marker))
    
    # Sort by position
    positions.sort(key=lambda x: x[0])
    
    # Extract each section
    for i, (pos, marker) in enumerate(positions):
        start = pos + len(marker)
        end = positions[i + 1][0] if i + 1 < len(positions) else len(text)
        
        section_name = marker.lower().replace(':', '')
        sections[section_name] = text[start:end].strip()
    
    return sections


def extract_section_facts(section_text: str) -> 'ClinicalFacts':
    """Extract clinical facts from a single SOAP section."""
    from clinical_facts import extract_facts_with_llm, ClinicalFacts
    
    if not section_text or not section_text.strip():
        return ClinicalFacts()
    
    return extract_facts_with_llm(section_text)


def flatten_facts_to_list(facts: 'ClinicalFacts') -> list:
    """Convert ClinicalFacts to a list of normalized strings."""
    from clinical_facts import flatten_facts
    
    fact_list = flatten_facts(facts)
    # Normalize: lowercase and strip
    return [f.lower().strip() for f in fact_list if f.strip()]


def compute_semantic_similarity(text1: str, text2: str) -> float:
    """
    Compute simple semantic similarity between two texts.
    Uses embeddings to get overall text similarity score.
    
    Args:
        text1: First text (e.g., model section)
        text2: Second text (e.g., clinician section)
        
    Returns:
        Cosine similarity score between 0-1
    """
    if not text1 or not text2 or not text1.strip() or not text2.strip():
        return 0.0
    
    from embeddings import compute_similarity_matrix
    
    try:
        sim_matrix = compute_similarity_matrix([text1], [text2])
        return float(sim_matrix[0][0])
    except Exception as e:
        print(f"Warning: Could not compute semantic similarity: {e}")
        return 0.0


def semantic_match_facts(
    model_facts: list,
    reference_facts: list,
    threshold: float = None
) -> dict:
    """
    Match facts using semantic similarity instead of exact string matching.
    Uses configured embeddings provider (Gemini, OpenAI, or local).
    
    Args:
        model_facts: List of facts from model output
        reference_facts: List of facts from ground truth
        threshold: Cosine similarity threshold for match (default: from config)
    
    Returns:
        Dict with matched_model, matched_ref, missing, hallucinated sets
    """
    from embeddings import compute_similarity_matrix, get_similarity_threshold
    
    if threshold is None:
        # Optimization: use cached config if available
        if hasattr(semantic_match_facts, "_threshold_cache"):
            threshold = semantic_match_facts._threshold_cache
        else:
            threshold = get_similarity_threshold()
            semantic_match_facts._threshold_cache = threshold
    
    if not model_facts or not reference_facts:
        return {
            'matched_model': set(),
            'matched_ref': set(),
            'missing': set(reference_facts) if reference_facts else set(),
            'hallucinated': set(model_facts) if model_facts else set()
        }
    
    # Compute similarity matrix using configured embeddings provider
    similarity_matrix = compute_similarity_matrix(model_facts, reference_facts)
    
    matched_model = set()
    matched_ref = set()
    
    # Greedy matching: for each model fact, find best matching reference fact
    for i, m_fact in enumerate(model_facts):
        for j, r_fact in enumerate(reference_facts):
            if r_fact in matched_ref:
                continue  # Already matched
            if similarity_matrix[i][j] >= threshold:
                matched_model.add(m_fact)
                matched_ref.add(r_fact)
                break  # Move to next model fact
    
    missing = set(reference_facts) - matched_ref
    hallucinated = set(model_facts) - matched_model
    
    return {
        'matched_model': matched_model,
        'matched_ref': matched_ref,
        'missing': missing,
        'hallucinated': hallucinated
    }


def compute_section_metrics(
    model_facts: 'ClinicalFacts',
    reference_facts: 'ClinicalFacts',
    transcript_facts: 'ClinicalFacts' = None,
    use_semantic: bool = True,
    threshold: float = 0.75
) -> SectionMetrics:
    """
    Compute metrics for a single section.
    
    Args:
        model_facts: Facts extracted from model-generated section
        reference_facts: Facts extracted from clinician's section (ground truth)
        transcript_facts: Facts from original transcript (for hallucination detection)
        use_semantic: If True, use semantic matching; else exact string matching
        threshold: Cosine similarity threshold for semantic matching
    
    Returns:
        SectionMetrics with missing rate, hallucination rate, precision, recall, F1
    """
    model_list = flatten_facts_to_list(model_facts)
    reference_list = flatten_facts_to_list(reference_facts)
    
    if use_semantic:
        # Semantic matching
        match_result = semantic_match_facts(model_list, reference_list, threshold)
        matched = match_result['matched_model']
        missing = match_result['missing']
        
        # For hallucination: check against transcript if provided
        if transcript_facts:
            transcript_list = flatten_facts_to_list(transcript_facts)
            # Hallucinated = in model, not matched to reference, AND not similar to transcript
            potential_hallucinated = match_result['hallucinated']
            
            if potential_hallucinated and transcript_list:
                # Check each potential hallucination against transcript
                transcript_match = semantic_match_facts(
                    list(potential_hallucinated), 
                    transcript_list, 
                    threshold
                )
                # True hallucinations: not found in transcript either
                hallucinated = transcript_match['hallucinated']
            else:
                hallucinated = potential_hallucinated
        else:
            hallucinated = match_result['hallucinated']
    else:
        # Exact string matching (fallback)
        model_set = set(model_list)
        reference_set = set(reference_list)
        
        matched = model_set & reference_set
        missing = reference_set - model_set
        
        if transcript_facts:
            transcript_set = set(flatten_facts_to_list(transcript_facts))
            hallucinated = model_set - reference_set - transcript_set
        else:
            hallucinated = model_set - reference_set
    
    # Calculate rates
    n_reference = max(1, len(reference_list))
    n_model = max(1, len(model_list))
    
    missing_rate = len(missing) / n_reference
    hallucinated_rate = len(hallucinated) / n_model
    
    # Precision, Recall, F1
    precision = len(matched) / n_model
    recall = len(matched) / n_reference
    f1 = 2 * precision * recall / max(0.001, precision + recall)
    
    return SectionMetrics(
        missing_count=len(missing),
        missing_rate=round(missing_rate, 4),
        hallucinated_count=len(hallucinated),
        hallucinated_rate=round(hallucinated_rate, 4),
        matched_count=len(matched),
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4)
    )


def evaluate_soap_sections(
    model_soap: Dict[str, str],
    clinician_soap: Dict[str, str],
    transcript: str = None
) -> SOAPMetrics:
    """
    Evaluate all SOAP sections and compute section-level metrics.
    Uses parallel processing for fact extraction to reduce latency.
    
    Args:
        model_soap: Model-generated SOAP as dict
        clinician_soap: Clinician's SOAP as dict
        transcript: Original transcript (optional, for hallucination detection)
    
    Returns:
        SOAPMetrics with per-section and overall metrics
    """
    import concurrent.futures
    
    sections = ['subjective', 'objective', 'assessment', 'plan']
    section_results = {}
    
    # 1. Prepare Extraction Tasks
    # keys: 'transcript', 'model_subjective', 'clinician_subjective', etc.
    extraction_tasks = {}
    
    if transcript:
        extraction_tasks['transcript'] = transcript
        
    for section in sections:
        extraction_tasks[f'model_{section}'] = model_soap.get(section, '')
        extraction_tasks[f'clinician_{section}'] = clinician_soap.get(section, '')
        
    # 2. Run Extraction in Parallel
    extracted_facts = {}
    
    # Use max_workers=5 to avoid hitting rate limits too hard, but speed up significantly
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_key = {
            executor.submit(extract_section_facts, text): key 
            for key, text in extraction_tasks.items()
        }
        
        for future in concurrent.futures.as_completed(future_to_key):
            key = future_to_key[future]
            try:
                extracted_facts[key] = future.result()
            except Exception as e:
                print(f"Error extracting for {key}: {e}")
                from clinical_facts import ClinicalFacts
                extracted_facts[key] = ClinicalFacts()

    # 3. Retrieve Transcript Facts
    transcript_facts = extracted_facts.get('transcript')
    
    # 4. Evaluate Sections
    all_model_facts = []
    all_clinician_facts = []
    all_model_text = []
    all_clinician_text = []
    
    for section in sections:
        model_text = model_soap.get(section, '')
        clinician_text = clinician_soap.get(section, '')
        
        model_facts = extracted_facts.get(f'model_{section}')
        clinician_facts = extracted_facts.get(f'clinician_{section}')
        
        # Compute fact-level metrics (CPU bound mostly)
        section_metrics = compute_section_metrics(
            model_facts, 
            clinician_facts,
            transcript_facts
        )
        
        # Add semantic similarity (simple text-level overlap)
        section_metrics.semantic_similarity = round(
            compute_semantic_similarity(model_text, clinician_text), 4
        )
        
        section_results[section] = section_metrics
        
        # Collect for overall
        all_model_facts.extend(flatten_facts_to_list(model_facts))
        all_clinician_facts.extend(flatten_facts_to_list(clinician_facts))
        all_model_text.append(model_text)
        all_clinician_text.append(clinician_text)
    
    # 5. Compute Overall Metrics
    from clinical_facts import ClinicalFacts
    
    # Create pseudo-facts objects for overall calculation
    overall_model = ClinicalFacts(other_findings=list(set(all_model_facts)))
    overall_clinician = ClinicalFacts(other_findings=list(set(all_clinician_facts)))
    
    overall_metrics = compute_section_metrics(
        overall_model,
        overall_clinician,
        transcript_facts
    )
    
    # Overall semantic similarity (full note comparison)
    overall_metrics.semantic_similarity = round(
        compute_semantic_similarity(
            ' '.join(all_model_text), 
            ' '.join(all_clinician_text)
        ), 4
    )
    
    return SOAPMetrics(
        subjective=section_results['subjective'],
        objective=section_results['objective'],
        assessment=section_results['assessment'],
        plan=section_results['plan'],
        overall=overall_metrics
    )

def print_soap_metrics(metrics: SOAPMetrics, note_id: int = 0):
    """Pretty print SOAP metrics."""
    print(f"\n{'='*50}")
    print(f"Note {note_id} Section-Level Results")
    print(f"{'='*50}")
    
    sections = ['subjective', 'objective', 'assessment', 'plan']
    
    for section in sections:
        m = getattr(metrics, section)
        print(f"{section.capitalize():12} | Missing: {m.missing_rate:5.1%} ({m.missing_count}) | "
              f"Halluc: {m.hallucinated_rate:5.1%} ({m.hallucinated_count}) | "
              f"F1: {m.f1:.2f} | Sim: {m.semantic_similarity:.2f}")
    
    print("-" * 50)
    print(f"{'Overall':12} | Missing: {metrics.overall.missing_rate:5.1%} | "
          f"Halluc: {metrics.overall.hallucinated_rate:5.1%} | "
          f"F1: {metrics.overall.f1:.2f} | Sim: {metrics.overall.semantic_similarity:.2f}")


def to_dict(metrics: SOAPMetrics) -> Dict:
    """Convert SOAPMetrics to serializable dict."""
    return {
        'subjective': asdict(metrics.subjective),
        'objective': asdict(metrics.objective),
        'assessment': asdict(metrics.assessment),
        'plan': asdict(metrics.plan),
        'overall': asdict(metrics.overall)
    }


# Example usage:
# from section_eval import generate_and_parse_soap, parse_clinician_soap, evaluate_soap_sections
#
# model_soap = generate_and_parse_soap(transcript, health_problem)
# clinician_soap = parse_clinician_soap(ground_truth_text)
# metrics = evaluate_soap_sections(model_soap, clinician_soap, transcript)
# print_soap_metrics(metrics)
