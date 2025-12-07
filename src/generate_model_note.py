"""
Generate SOAP notes from medical transcripts using LLM.
Supports OpenAI and Ollama with few-shot prompting via unified llm_client.
Returns structured JSON output.
"""

import json
import time
import os
from typing import Literal, Dict

# Common medications for spelling correction
COMMON_MEDICATIONS = [
    "acetaminophen", "ibuprofen", "aspirin", "lisinopril", "metformin",
    "amlodipine", "metoprolol", "omeprazole", "losartan", "gabapentin",
    "hydrochlorothiazide", "atorvastatin", "simvastatin", "levothyroxine",
    "prednisone", "amoxicillin", "azithromycin", "ciprofloxacin", "fluoxetine",
    "sertraline", "alprazolam", "lorazepam", "tramadol", "oxycodone",
    "hydrocodone", "warfarin", "clopidogrel", "albuterol", "montelukast",
    "pantoprazole", "esomeprazole", "duloxetine", "venlafaxine", "bupropion",
    "trazodone", "quetiapine", "risperidone", "aripiprazole", "insulin",
    "glipizide", "sitagliptin", "empagliflozin", "liraglutide", "cyclobenzaprine"
]


# System prompts - different levels of detail
SYSTEM_PROMPTS = {
    "basic": """As a highly skilled medical assistant, your task is to meticulously review the provided TRANSCRIPT and craft a clinical SOAP note in the form of a JSON object. Please adhere strictly to the following guidelines:
- Subjective, Objective, and Assessment sections must be written as PARAGRAPHS (no bullet points or lists).
- For the Plan section ONLY, use NUMBERED lists (1. 2. 3.) to show priority/sequence of actions.
- Incorporate as much detailed information as possible from the transcript into the SOAP note. Thoroughness is key!
- If certain information required for any fields is missing from the transcript, exclude those fields from the JSON object entirely. Do not include fields with empty strings or "unknown" values.
- The transcript may not explicitly mention differential diagnoses. As an expert, you are expected to formulate a differential diagnosis based on the transcript information.
Your expertise and attention to detail will ensure the generation of a comprehensive and accurate SOAP note.""",

    "enhanced": """As a highly skilled medical assistant, your task is to meticulously review the provided TRANSCRIPT and craft a clinical SOAP note in the form of a JSON object. Please adhere strictly to the following guidelines:
- Subjective, Objective, and Assessment sections must be written as PARAGRAPHS (no bullet points or lists).
- For the Plan section ONLY, use NUMBERED lists (1. 2. 3.) to show priority/sequence of actions.
- Incorporate as much detailed information as possible from the transcript into the SOAP note. Thoroughness is key!
- If certain information required for any fields is missing from the transcript, exclude those fields from the JSON object entirely. Do not include fields with empty strings or "unknown" values.
- The transcript may not explicitly mention differential diagnoses. As an expert, you are expected to formulate a differential diagnosis based on the transcript information.
- Be vigilant for formatting and spelling errors in the transcript, particularly regarding prescription medications. Correct these errors accurately. Pay special attention to the spelling and formatting of any prescription medications mentioned.
Your expertise and attention to detail will ensure the generation of a comprehensive and accurate SOAP note.""",

    "advanced": f"""As a highly skilled medical assistant, your task is to meticulously review the provided TRANSCRIPT and craft a clinical SOAP note in the form of a JSON object. Please adhere strictly to the following guidelines:
- Subjective, Objective, and Assessment sections must be written as PARAGRAPHS (no bullet points or lists).
- For the Plan section ONLY, use NUMBERED lists (1. 2. 3.) to show priority/sequence of actions.
- Incorporate as much detailed information as possible from the transcript into the SOAP note. Thoroughness is key!
- If certain information required for any fields is missing from the transcript, exclude those fields from the JSON object entirely. Do not include fields with empty strings or "unknown" values.
- The transcript may not explicitly mention differential diagnoses. As an expert, you are expected to formulate a differential diagnosis based on the transcript information.
- Be vigilant for formatting and spelling errors in the transcript, particularly regarding prescription medications. Here is a list of common medication names: {', '.join(COMMON_MEDICATIONS)}. The transcript may include misspellings of these or other medications. Be sure to provide the correct spelling. Correct medication dosage transcriptions by standardizing the format to use a slash ("/") between different ingredient amounts. Convert verbal expressions of dosage, such as "five slash three twenty-five milligrams" or "five milligrams and three hundred twenty-five milligrams," to the format "5/325 milligrams." Ensure the correct separation of amounts and units according to standard prescription practices.
Your expertise and attention to detail will ensure the generation of a comprehensive and accurate SOAP note."""
}


FEW_SHOT_EXAMPLES = """
### Example 1: Gastrointestinal Issues

**Transcript:**
Physician: Good morning, Mr. Diaz. How are you feeling today?

Patient: Not great, Dr. Burns. I've been feeling lousy lately. That heartburn I mentioned is constant, even with the medication.

Physician: I see. Have you noticed any other symptoms along with the heartburn?

Patient: Yes, I've had some sharp stomach pain, bloating, and just general discomfort in my abdomen. And I've been having diarrhea on and off.

Physician: Any nausea or vomiting?

Patient: Yes, both. And I've lost some weight without trying. My appetite just isn't what it used to be.

Physician: I understand. Have you noticed anything that seems to make these symptoms better or worse?

Patient: Spicy foods definitely make it worse. And stress seems to trigger the pain too.

Physician: Based on what you're describing, I'd like to order an endoscopy and some blood tests to get a better picture of what's going on. I also want you to start keeping a food diary to track any triggers.

**JSON Output:**
{
  "subjective": "Mr. Diaz reports feeling lousy lately with constant heartburn that persists despite medication. He also experiences sharp stomach pain, bloating, abdominal discomfort, intermittent diarrhea, nausea, vomiting, weight loss, loss of appetite, and fatigue. He notes that spicy foods and stress may exacerbate his symptoms.",
  "objective": "Mr. Diaz appears fatigued during the appointment. He denies any recent travel or exposures to illness. No acute distress noted.",
  "assessment": "Mr. Diaz presents with chronic GERD and gastrointestinal issues requiring further evaluation. Symptoms are consistent with possible peptic ulcer disease or inflammatory bowel condition. Differential diagnosis includes peptic ulcer disease, gastritis, inflammatory bowel disease, celiac disease, and functional dyspepsia.",
  "plan": "1. Order endoscopy to evaluate upper GI tract\\n2. Order blood tests including CBC, CMP, H. pylori\\n3. Initiate food diary to track triggers\\n4. Continue current GERD medication\\n5. Follow up with results to determine appropriate treatment plan\\n6. Consider PPI adjustment if needed"
}

---

### Example 2: Diabetes Management

**Transcript:**
Physician: Good morning, Nina. How are you feeling today?

Patient: Good morning, Dr. Fowler. I've been feeling alright, but I have some unusual symptoms lately.

Physician: Can you tell me more about these symptoms?

Patient: I've been feeling increasingly thirsty and urinating more frequently, especially at night. And my vision has been a bit blurry lately.

Physician: Have you noticed any numbness or tingling sensations?

Patient: Yes, my feet have been tingling, especially in the mornings. And I've been feeling more fatigued than usual.

Physician: Have you been managing your diabetes well?

Patient: To be honest, I haven't been checking my levels as regularly as I should. My last A1C was a bit higher than usual.

Physician: I'd like to schedule you for some follow-up tests - blood test for A1C, blood lipids, and kidney function. We'll also schedule a retinal exam.

**JSON Output:**
{
  "subjective": "Nina reports feeling increasingly thirsty and urinating more frequently, especially at night. She also mentions experiencing blurry vision, tingling sensations in her feet, fatigue, and occasional dizziness. Nina admits to not checking her blood sugar levels regularly and notes that her last A1C result was higher than usual.",
  "objective": "Nina appears alert and oriented during the conversation. Vital signs are within normal limits. No signs of distress or acute pain are observed. Laboratory results from the last visit show an A1C of 7.1% and LDL cholesterol of 115 mg/dL.",
  "assessment": "Nina is presenting with symptoms suggestive of uncontrolled diabetes, including polyuria, polydipsia, blurry vision, peripheral neuropathy, and fatigue. Her elevated A1C indicates poor glycemic control. Differential diagnosis includes diabetic neuropathy, diabetic retinopathy, diabetic nephropathy, medication non-adherence, and thyroid dysfunction.",
  "plan": "1. Schedule follow-up tests: A1C, blood lipids, kidney function\\n2. Arrange retinal exam to evaluate eye health\\n3. Review current medication regimen for potential adjustments\\n4. Discuss lifestyle modifications\\n5. Ensure vaccinations and cancer screenings are up to date\\n6. Schedule follow-up appointment for next week"
}

---

### Example 3: Dermatology - Psoriasis

**Transcript:**
Physician: Hi, I'm Dr. White. How are you doing today, Seth?

Patient: Hi, Dr. White. I've been doing alright. I guess you're here to check on my psoriasis?

Physician: Yes, that's right. Can you tell me about your current symptoms?

Patient: My skin has been feeling pretty dry lately, and I've had some flaking on my elbows and knees. It's not too itchy, but it's been bothering me.

Physician: Have you noticed any changes since your last appointment?

Patient: I feel like it's been pretty consistent. I've been using the cream that was prescribed, but I'm not sure if it's really doing much.

Physician: Have you been using any new soaps or lotions?

Patient: Actually, yes. My wife got me this new moisturizer that's supposed to be good for dry skin.

Physician: I can see your psoriasis is well-controlled, but there's some dryness and flaking on your elbows and knees, and redness on your lower back. I think we should switch you to a different moisturizer and prescribe a topical corticosteroid for the redness.

**JSON Output:**
{
  "subjective": "Seth Martin reports experiencing dry skin and flaking on his elbows and knees, with occasional redness and irritation. He mentions using a new moisturizer that may be causing some irritation. Seth states that his psoriasis has been consistent, and he is unsure if the prescribed cream is effective.",
  "objective": "Upon examination, Seth's psoriasis is well-controlled, but there is noticeable dryness, flaking, redness on his lower back, elbows, and knees. The patient's skin shows signs of irritation, possibly due to the new moisturizer used.",
  "assessment": "Seth's psoriasis is currently well-controlled, but he is experiencing dry skin and irritation, likely due to the new moisturizer. A change in moisturizer and the addition of a topical corticosteroid for the redness and irritation are recommended. Differential diagnosis includes contact dermatitis from new moisturizer, psoriasis flare, eczema, and allergic reaction.",
  "plan": "1. Discontinue new moisturizer\\n2. Prescribe fragrance-free moisturizer for dry skin\\n3. Prescribe topical corticosteroid for lower back redness/irritation\\n4. Schedule follow-up to monitor treatment effectiveness\\n5. Consider referral to dermatology if no improvement"
}
"""


def generate_model_note(
    transcript: str,
    health_problem: str = None,
    model: str = None,
    provider: Literal["openai", "ollama", "gemini"] = None,
    prompt_strategy: Literal["zero-shot", "one-shot", "few-shot"] = None,
    system_prompt_level: Literal["basic", "enhanced", "advanced"] = "enhanced",
    max_retries: int = 3
) -> Dict[str, str]:
    """
    Generate structured JSON SOAP note from transcript.
    
    Args:
        transcript: Doctor-patient conversation transcript
        health_problem: Chief complaint/reason for visit (from dataset column)
        model: Model name (default: from config.yaml)
        provider: "openai", "ollama", or "gemini" (default: from config.yaml)
        prompt_strategy: "zero-shot", "one-shot", or "few-shot" (default: from config.yaml)
        system_prompt_level: "basic", "enhanced", or "advanced" for medication correction
        max_retries: Number of retry attempts on failure
        
    Returns:
        Dict with keys: subjective, objective, assessment, plan
    """
    # Load defaults from config if not provided - CACHED
    if not hasattr(generate_model_note, "_config_cache"):
        from config_loader import get_llm_config
        generate_model_note._config_cache = get_llm_config()
    
    llm_config = generate_model_note._config_cache
    
    if provider is None:
        provider = llm_config.get('provider', 'ollama')
    if model is None:
        model = llm_config.get('model', 'gemma3:4b')
    if prompt_strategy is None:
        prompt_strategy = llm_config.get('prompt_strategy', 'few-shot')
    
    system_prompt = SYSTEM_PROMPTS.get(system_prompt_level, SYSTEM_PROMPTS["enhanced"])
    
    # Add health problem context if provided
    health_context = ""
    if health_problem:
        health_context = f"\n\n**Chief Complaint/Health Problem:** {health_problem}"
    
    json_format = """{
  "subjective": "Single paragraph with patient's reported symptoms, history, complaints.",
  "objective": "Single paragraph with examination findings, vital signs, lab results.",
  "assessment": "Single paragraph with clinical impression, diagnosis, and differential diagnosis.",
  "plan": "NUMBERED list (1. 2. 3.) with treatment plan, medications, tests, referrals, follow-up."
}"""
    
    # Build user prompt based on strategy
    if prompt_strategy == "zero-shot":
        user_prompt = f"""Convert this transcript to a JSON SOAP note.

Output EXACTLY this JSON format:
{json_format}

Rules:
- Subjective, Objective, Assessment must be PARAGRAPHS (no bullet points)
- Plan must use NUMBERED list (1. 2. 3.)
- Include differential diagnosis IN the assessment paragraph
- Correct any medication spelling errors
- Exclude fields if no information available
- Focus on the chief complaint: {health_problem if health_problem else 'as described in transcript'}

Transcript:
{transcript}{health_context}

JSON:"""
    
    elif prompt_strategy == "one-shot":
        one_shot = FEW_SHOT_EXAMPLES.split("---")[0]
        user_prompt = f"""Convert transcripts to JSON SOAP notes.

Output format:
{json_format}

Example:
{one_shot}

Now convert this transcript:
{health_context}
Transcript:
{transcript}

JSON:"""
    
    else:  # few-shot (default)
        user_prompt = f"""Convert transcripts to JSON SOAP notes.

Output format:
{json_format}

Examples:
{FEW_SHOT_EXAMPLES}

Now convert this transcript:
{health_context}
Transcript:
{transcript}

JSON:"""

    # Use unified LLM client
    from llm_client import query_llm
    
    try:
        return query_llm(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            provider=provider,
            model=model,
            json_mode=True,
            max_retries=max_retries
        )
    except Exception as e:
        print(f"Error generating note: {e}")
        return {
            "subjective": "",
            "objective": "",
            "assessment": "",
            "plan": ""
        }


def format_soap_note(soap_dict: Dict[str, str]) -> str:
    """Convert JSON SOAP dict to formatted string."""
    sections = []
    
    if soap_dict.get('subjective'):
        sections.append(f"Subjective:\n{soap_dict['subjective']}")
    
    if soap_dict.get('objective'):
        sections.append(f"Objective:\n{soap_dict['objective']}")
    
    if soap_dict.get('assessment'):
        sections.append(f"Assessment:\n{soap_dict['assessment']}")
    
    if soap_dict.get('plan'):
        sections.append(f"Plan:\n{soap_dict['plan']}")
    
    return "\n\n".join(sections)
