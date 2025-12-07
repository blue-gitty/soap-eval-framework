"""
LLM-based clinical fact extraction.
Supports OpenAI GPT-4o-mini and local Ollama models (Gemma 3B).
Includes zero-shot, one-shot, and few-shot prompting strategies.
"""

import json
import time
import os
from typing import List, Literal
from pydantic import BaseModel, Field

# Optional imports - will check availability
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False




class ClinicalFacts(BaseModel):
    """Structured clinical facts extracted from text."""
    problems: List[str] = Field(default_factory=list, description="Patient problems, complaints, diagnoses")
    medications: List[str] = Field(default_factory=list, description="Drugs with dose or frequency if available")
    allergies: List[str] = Field(default_factory=list, description="Known allergies")
    plans: List[str] = Field(default_factory=list, description="Tests, referrals, lifestyle advice, follow-up")
    other_findings: List[str] = Field(default_factory=list, description="Important facts that don't fit above")


def get_zero_shot_prompt(text: str) -> str:
    """Generate zero-shot prompt (no examples)."""
    return f"""You are a clinical NLP assistant.
Extract structured clinical facts from the following text.

Extract facts into this JSON schema:
{{
  "problems": [string],           # patient problems / complaints / diagnoses
  "medications": [string],        # drugs with dose or frequency if available
  "allergies": [string],
  "plans": [string],              # tests, referrals, lifestyle advice, follow-up
  "other_findings": [string]      # anything important that doesn't fit above
}}

Rules:
- Use short phrases, not full sentences.
- Do not invent facts that are not clearly supported by the text.
- If you are unsure, omit the fact.
- Return ONLY valid JSON.

TEXT:
{text}

OUTPUT:"""


def get_one_shot_prompt(text: str) -> str:
    """Generate one-shot prompt (1 example)."""
    return f"""You are a clinical NLP assistant.
Extract structured clinical facts from transcripts and SOAP notes.

Extract facts into this JSON schema:
{{
  "problems": [string],
  "medications": [string],
  "allergies": [string],
  "plans": [string],
  "other_findings": [string]
}}

Rules:
- Use short phrases, not full sentences.
- Do not invent facts not supported by the text.
- Return ONLY valid JSON.

Example:

TEXT:
Patient: I've had knee pain for 6 months. I take ibuprofen 600mg as needed but it hurts my stomach.
Doctor: We'll order an MRI and refer you to physical therapy. You have arthritis.

OUTPUT:
{{
  "problems": ["knee pain for 6 months", "arthritis"],
  "medications": ["ibuprofen 600mg as needed"],
  "allergies": [],
  "plans": ["order MRI", "refer to physical therapy"],
  "other_findings": ["ibuprofen causes stomach discomfort"]
}}

Now extract facts for this text:

TEXT:
{text}

OUTPUT:"""


def get_few_shot_prompt(text: str) -> str:
    """Generate few-shot prompt (multiple examples)."""
    return f"""You are a clinical NLP assistant.
Extract structured clinical facts from transcripts and SOAP notes.

Extract facts into this JSON schema:
{{
  "problems": [string],
  "medications": [string],
  "allergies": [string],
  "plans": [string],
  "other_findings": [string]
}}

Rules:
- Use short phrases, not full sentences.
- Do not invent facts not supported by the text.
- Return ONLY valid JSON.

Example 1:

TEXT:
Patient: I've had knee pain for 6 months. I take ibuprofen 600mg as needed but it hurts my stomach. 
Doctor: We'll order an MRI and refer you to physical therapy. I guess you have arthritis.

OUTPUT:
{{
  "problems": ["knee pain for 6 months", "arthritis"],
  "medications": ["ibuprofen 600mg as needed"],
  "allergies": [],
  "plans": ["order MRI", "refer to physical therapy"],
  "other_findings": ["ibuprofen causes stomach discomfort"]
}}

Example 2:

TEXT:
Patient has diabetes and hypertension. Currently on metformin 500mg twice daily and lisinopril 10mg daily. Allergic to penicillin. Plan to check A1C and refer to dietitian.

OUTPUT:
{{
  "problems": ["diabetes", "hypertension"],
  "medications": ["metformin 500mg twice daily", "lisinopril 10mg daily"],
  "allergies": ["penicillin"],
  "plans": ["check A1C", "refer to dietitian"],
  "other_findings": []
}}

Now extract facts for this text:

TEXT:
{text}

OUTPUT:"""


def extract_facts_with_llm(
    text: str,
    model: str = None,
    provider: Literal["openai", "ollama", "gemini"] = None,
    prompt_strategy: Literal["zero-shot", "one-shot", "few-shot"] = None,
    max_retries: int = None
) -> ClinicalFacts:
    """
    Extract clinical facts using LLM with few-shot prompting.
    
    Args:
        text: Clinical text (transcript or SOAP note)
        model: Model name (default: from config.yaml)
        provider: "openai", "ollama", or "gemini" (default: from config.yaml)
        prompt_strategy: "zero-shot", "one-shot", or "few-shot" (default: from config.yaml)
        max_retries: Number of retry attempts on failure (default: from config.yaml)
        
    Returns:
        ClinicalFacts object with structured facts
    """
    # Load defaults from config if not provided - CACHED
    if not hasattr(extract_facts_with_llm, "_config_cache"):
        from config_loader import get_llm_config
        extract_facts_with_llm._config_cache = get_llm_config()
    
    llm_config = extract_facts_with_llm._config_cache
    
    if provider is None:
        provider = llm_config.get('provider', 'ollama')
    if model is None:
        model = llm_config.get('model', 'gemma3:4b')
    if prompt_strategy is None:
        prompt_strategy = llm_config.get('prompt_strategy', 'few-shot')
    if max_retries is None:
        max_retries = llm_config.get('max_retries', 3)
    
    # Select prompt based on strategy (Lazy evaluation to save string ops)
    if prompt_strategy == "zero-shot":
        prompt = get_zero_shot_prompt(text)
        system_content = "You are a clinical NLP assistant. Return ONLY valid JSON."
    elif prompt_strategy == "one-shot":
        prompt = get_one_shot_prompt(text)
        system_content = "You are a clinical NLP assistant. Extract facts as JSON."
    else:  # few-shot (default)
        prompt = get_few_shot_prompt(text)
        system_content = "You are a clinical NLP assistant. Extract facts as JSON."

    for attempt in range(max_retries):
        try:
            facts_dict = {}
            
            if provider == "openai":
                if not OPENAI_AVAILABLE:
                    raise ImportError("OpenAI client not installed. Run `pip install openai`.")
                
                api_key = os.getenv('OPENAI_API_KEY')
                if not api_key:
                    raise ValueError(
                        "OPENAI_API_KEY environment variable not set. "
                        "Set it via: export OPENAI_API_KEY='your-key' or create .env file"
                    )
                
                client = openai.OpenAI(api_key=api_key)
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.1
                )
                facts_str = response.choices[0].message.content
                facts_dict = json.loads(facts_str)
                

            elif provider == "ollama":
                if not OLLAMA_AVAILABLE:
                    raise ImportError("Ollama client not installed. Run `pip install ollama`.")
                
                response = ollama.chat(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": prompt}
                    ],
                    format='json',
                    options={'temperature': 0.1}
                )
                facts_str = response['message']['content']
                facts_dict = json.loads(facts_str)
            
            elif provider == "gemini":
                # Use unified llm_client for Gemini support
                from llm_client import query_llm
                facts_dict = query_llm(
                    messages=[
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": prompt}
                    ],
                    provider="gemini",
                    model=model,
                    json_mode=True,
                    temperature=0.1,
                    max_retries=max_retries
                )
            
            else:
                raise ValueError(f"Unknown provider: {provider}")

            # Validate with Pydantic
            facts = ClinicalFacts(**facts_dict)
            return facts
            
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Error extracting facts ({provider}/{model}): {e}")
                return ClinicalFacts()
            time.sleep(1)


def flatten_facts(facts: ClinicalFacts) -> List[str]:
    """Flatten all facts into a single list."""
    all_facts = []
    for field in facts.model_dump().values():
        all_facts.extend(field)
    return all_facts


def compare_facts(
    transcript_facts: ClinicalFacts,
    generated_facts: ClinicalFacts,
    ground_truth_facts: ClinicalFacts = None
) -> dict:
    """
    Compare facts between transcript, generated note, and optionally ground truth.
    
    Returns:
        Dict with missing facts, hallucinated facts, and metrics
    """
    t_set = set(flatten_facts(transcript_facts))
    g_set = set(flatten_facts(generated_facts))
    
    missing = t_set - g_set
    hallucinated = g_set - t_set
    
    missing_rate = len(missing) / max(1, len(t_set))
    hallucination_rate = len(hallucinated) / max(1, len(g_set))
    
    recall = len(t_set & g_set) / max(1, len(t_set))
    precision = len(t_set & g_set) / max(1, len(g_set))
    f1 = 2 * (recall * precision) / max(0.001, recall + precision)
    
    results = {
        'missing_facts': list(missing),
        'hallucinated_facts': list(hallucinated),
        'missing_rate': missing_rate,
        'hallucination_rate': hallucination_rate,
        'recall': recall,
        'precision': precision,
        'f1': f1,
        'transcript_fact_count': len(t_set),
        'generated_fact_count': len(g_set)
    }
    
    if ground_truth_facts:
        gt_set = set(flatten_facts(ground_truth_facts))
        missing_vs_gt = gt_set - g_set
        extra_vs_gt = g_set - gt_set
        
        recall_vs_gt = len(gt_set & g_set) / max(1, len(gt_set))
        precision_vs_gt = len(gt_set & g_set) / max(1, len(g_set))
        f1_vs_gt = 2 * (recall_vs_gt * precision_vs_gt) / max(0.001, recall_vs_gt + precision_vs_gt)
        
        results['vs_ground_truth'] = {
            'missing_facts': list(missing_vs_gt),
            'extra_facts': list(extra_vs_gt),
            'recall': recall_vs_gt,
            'precision': precision_vs_gt,
            'f1': f1_vs_gt
        }
    
    return results
