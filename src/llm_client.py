"""
Unified LLM Client for DeepScribe Evals.
Handles communication with OpenAI, Gemini (API), and Ollama (Local) seamlessly.
"""

import os
import json
import time
from typing import List, Dict, Any, Optional, Literal

# Check available providers
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

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# Cache for config
_CONFIG_CACHE = None

def get_default_llm_config():
    """Get config from config_loader or default values."""
    global _CONFIG_CACHE
    if _CONFIG_CACHE:
        return _CONFIG_CACHE
        
    try:
        from config_loader import get_llm_config
        _CONFIG_CACHE = get_llm_config()
    except ImportError:
        # Fallback if config_loader not found
        _CONFIG_CACHE = {
            'provider': 'ollama',
            'model': 'gemma3:4b',
            'temperature': 0.1,
            'max_retries': 3
        }
    return _CONFIG_CACHE

def query_llm(
    messages: List[Dict[str, str]],
    provider: str = None,
    model: str = None,
    temperature: float = None,
    json_mode: bool = False,
    max_retries: int = None
) -> Any:
    """
    Send a query to the configured LLM provider.
    
    Args:
        messages: List of message dicts (role, content)
        provider: 'openai', 'gemini', or 'ollama' (overrides config)
        model: Model name (overrides config)
        temperature: 0.0 to 1.0 (overrides config)
        json_mode: If True, enforces JSON object response
        max_retries: Number of retries (overrides config)
        
    Returns:
        The response content (str) or parsed JSON dict (if json_mode=True)
    """
    config = get_default_llm_config()
    
    # Resolve parameters
    provider = provider or config.get('provider', 'ollama')
    model = model or config.get('model', 'gemma3:4b')
    temperature = temperature if temperature is not None else config.get('temperature', 0.1)
    max_retries = max_retries if max_retries is not None else config.get('max_retries', 3)
    
    last_error = None
    
    for attempt in range(max_retries):
        try:
            if provider == 'openai':
                if not OPENAI_AVAILABLE:
                    raise ImportError("OpenAI client not installed. Run `pip install openai`.")
                
                api_key = os.getenv('OPENAI_API_KEY')
                if not api_key:
                    raise ValueError(
                        "OPENAI_API_KEY environment variable not set. "
                        "Set it via: export OPENAI_API_KEY='your-key' or create .env file"
                    )
                
                client = openai.OpenAI(api_key=api_key)
                
                kwargs = {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature
                }
                
                if json_mode:
                    kwargs["response_format"] = {"type": "json_object"}
                
                response = client.chat.completions.create(**kwargs)
                content = response.choices[0].message.content.strip()
                
                if json_mode:
                    return json.loads(content)
                return content
                
            elif provider == 'ollama':
                if not OLLAMA_AVAILABLE:
                    raise ImportError("Ollama client not installed. Run `pip install ollama`.")
                
                options = {'temperature': temperature}
                kwargs = {
                    "model": model,
                    "messages": messages,
                    "options": options
                }
                
                if json_mode:
                    kwargs["format"] = "json"
                
                response = ollama.chat(**kwargs)
                content = response['message']['content'].strip()
                
                if json_mode:
                    return json.loads(content)
                return content
            
            elif provider == 'gemini':
                if not GEMINI_AVAILABLE:
                    raise ImportError("Google Generative AI client not installed. Run `pip install google-generativeai`.")
                
                api_key = os.getenv('GEMINI_API_KEY')
                if not api_key:
                    raise ValueError(
                        "GEMINI_API_KEY environment variable not set. "
                        "Set it via: export GEMINI_API_KEY='your-key' or create .env file"
                    )
                
                # Configure Gemini (only once)
                if not hasattr(query_llm, "_gemini_configured"):
                    genai.configure(api_key=api_key)
                    query_llm._gemini_configured = True
                
                # Convert messages format for Gemini
                # Extract system message and user messages
                system_instruction = None
                prompt_parts = []
                
                for msg in messages:
                    if msg['role'] == 'system':
                        system_instruction = msg['content']
                    elif msg['role'] == 'user':
                        prompt_parts.append(msg['content'])
                    elif msg['role'] == 'assistant':
                        # For chat history, we'd need to use ChatSession
                        # For now, skip assistant messages in single-turn calls
                        pass
                
                # If no user messages, use system as prompt
                if not prompt_parts and system_instruction:
                    prompt_parts = [system_instruction]
                    system_instruction = None
                
                # Get the model with optional system instruction
                if system_instruction:
                    gemini_model = genai.GenerativeModel(
                        model,
                        system_instruction=system_instruction
                    )
                else:
                    gemini_model = genai.GenerativeModel(model)
                
                # Build generation config
                generation_config = genai.types.GenerationConfig(
                    temperature=temperature,
                )
                
                if json_mode:
                    # For JSON mode, use response_mime_type
                    generation_config.response_mime_type = 'application/json'
                    # Also add instruction if no system message
                    if not system_instruction and prompt_parts:
                        prompt_parts[-1] = prompt_parts[-1] + "\n\nIMPORTANT: Respond with ONLY valid JSON. No markdown, no code blocks, just the JSON object."
                
                # Generate content
                # Combine all prompt parts into a single string for Gemini
                full_prompt = "\n\n".join(prompt_parts) if prompt_parts else ""
                response = gemini_model.generate_content(
                    full_prompt,
                    generation_config=generation_config
                )
                
                content = response.text.strip()
                
                if json_mode:
                    # Try to parse JSON (remove markdown code blocks if present)
                    content = content.strip()
                    if content.startswith('```'):
                        # Remove markdown code blocks
                        lines = content.split('\n')
                        content = '\n'.join(lines[1:-1]) if len(lines) > 2 else content
                    elif content.startswith('```json'):
                        lines = content.split('\n')
                        content = '\n'.join(lines[1:-1]) if len(lines) > 2 else content
                    return json.loads(content)
                return content
            
            else:
                raise ValueError(f"Unknown provider: {provider}")
                
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                time.sleep(1 * (attempt + 1))  # Exponential backoff
            
    raise ValueError(f"LLM Query failed after {max_retries} attempts. Last error: {last_error}")
