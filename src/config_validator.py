"""
Configuration validation and health checks for production use.
Validates API keys, provider configurations, and dependencies.
"""

import os
from typing import Dict, List, Tuple, Optional
from pathlib import Path

def validate_api_key(provider: str, api_key: Optional[str] = None) -> Tuple[bool, str]:
    """
    Validate that API key exists for a given provider.
    
    Args:
        provider: 'openai', 'gemini', or 'ollama'
        api_key: Optional API key to check (if None, checks env vars)
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if provider == 'ollama':
        # Ollama doesn't need API key (local)
        return True, ""
    
    if api_key is None:
        if provider == 'openai':
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                return False, "OPENAI_API_KEY environment variable not set. Required for OpenAI provider."
        elif provider == 'gemini':
            api_key = os.getenv('GEMINI_API_KEY')
            if not api_key:
                return False, "GEMINI_API_KEY environment variable not set. Required for Gemini provider."
        else:
            return False, f"Unknown provider: {provider}"
    
    if not api_key or api_key.strip() == "":
        return False, f"API key for {provider} is empty"
    
    # Basic format validation
    if provider == 'openai' and not api_key.startswith('sk-'):
        return False, "OpenAI API key format appears invalid (should start with 'sk-')"
    
    if provider == 'gemini' and len(api_key) < 20:
        return False, "Gemini API key format appears invalid (too short)"
    
    return True, ""

def validate_provider_available(provider: str) -> Tuple[bool, str]:
    """
    Check if provider dependencies are installed.
    
    Args:
        provider: 'openai', 'ollama', or 'gemini'
        
    Returns:
        Tuple of (is_available, error_message)
    """
    if provider == 'openai':
        try:
            import openai
            return True, ""
        except ImportError:
            return False, "OpenAI client not installed. Run: pip install openai"
    
    elif provider == 'ollama':
        try:
            import ollama
            return True, ""
        except ImportError:
            return False, "Ollama client not installed. Run: pip install ollama"
    
    elif provider == 'gemini':
        try:
            import google.generativeai
            return True, ""
        except ImportError:
            return False, "Google Generative AI client not installed. Run: pip install google-generativeai"
    
    else:
        return False, f"Unknown provider: {provider}"

def validate_llm_provider(provider: str) -> Tuple[bool, str]:
    """
    Validate LLM provider specifically (supports gemini for LLM calls).
    
    Args:
        provider: 'openai', 'ollama', or 'gemini'
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    return validate_provider_available(provider)

def validate_config(config: Dict) -> List[str]:
    """
    Validate entire configuration and return list of errors/warnings.
    
    Args:
        config: Configuration dictionary from config_loader
        
    Returns:
        List of error/warning messages (empty if all valid)
    """
    errors = []
    warnings = []
    
    # Validate LLM provider
    llm_config = config.get('llm', {})
    provider = llm_config.get('provider', 'ollama')
    
    # Check provider is available
    is_available, msg = validate_provider_available(provider)
    if not is_available:
        errors.append(f"LLM Provider '{provider}': {msg}")
    else:
        # Check API key if needed
        is_valid, key_msg = validate_api_key(provider)
        if not is_valid:
            errors.append(f"LLM Provider '{provider}': {key_msg}")
    
    # Validate embeddings provider
    emb_config = config.get('embeddings', {})
    emb_provider = emb_config.get('provider', 'gemini')
    
    is_available, msg = validate_provider_available(emb_provider)
    if not is_available:
        errors.append(f"Embeddings Provider '{emb_provider}': {msg}")
    else:
        # Check API key if needed
        is_valid, key_msg = validate_api_key(emb_provider)
        if not is_valid:
            errors.append(f"Embeddings Provider '{emb_provider}': {key_msg}")
    
    # Validate model names (basic checks)
    model = llm_config.get('model', '')
    if provider == 'openai' and model and not any(x in model.lower() for x in ['gpt', 'o1']):
        warnings.append(f"Model '{model}' may not be a valid OpenAI model")
    
    if provider == 'ollama' and model and len(model) < 3:
        warnings.append(f"Model '{model}' may not be a valid Ollama model")
    
    if provider == 'gemini' and model and not any(x in model.lower() for x in ['gemini']):
        warnings.append(f"Model '{model}' may not be a valid Gemini model (should contain 'gemini')")
    
    # Validate temperature
    temperature = llm_config.get('temperature', 0.1)
    if not (0.0 <= temperature <= 2.0):
        errors.append(f"Temperature {temperature} is out of valid range [0.0, 2.0]")
    
    # Validate similarity threshold
    similarity_threshold = emb_config.get('similarity_threshold', 0.75)
    if not (0.0 <= similarity_threshold <= 1.0):
        errors.append(f"Similarity threshold {similarity_threshold} is out of valid range [0.0, 1.0]")
    
    return errors + warnings

def health_check() -> Dict[str, any]:
    """
    Run comprehensive health check of configuration and dependencies.
    
    Returns:
        Dict with 'status', 'errors', 'warnings', and 'info'
    """
    from config_loader import load_config
    
    config = load_config()
    issues = validate_config(config)
    
    errors = [i for i in issues if 'not' in i.lower() or 'invalid' in i.lower() or 'out of' in i.lower()]
    warnings = [i for i in issues if i not in errors]
    
    status = "healthy" if not errors else "unhealthy"
    
    info = {
        'llm_provider': config.get('llm', {}).get('provider', 'unknown'),
        'llm_model': config.get('llm', {}).get('model', 'unknown'),
        'embeddings_provider': config.get('embeddings', {}).get('provider', 'unknown'),
        'embeddings_model': config.get('embeddings', {}).get('model', 'unknown'),
    }
    
    return {
        'status': status,
        'errors': errors,
        'warnings': warnings,
        'info': info
    }

if __name__ == "__main__":
    """Run health check when executed directly."""
    result = health_check()
    
    print("="*60)
    print("🔍 CONFIGURATION HEALTH CHECK")
    print("="*60)
    
    print(f"\nStatus: {result['status'].upper()}")
    
    print(f"\n📋 Configuration:")
    for key, value in result['info'].items():
        print(f"   {key}: {value}")
    
    if result['errors']:
        print(f"\n❌ Errors ({len(result['errors'])}):")
        for error in result['errors']:
            print(f"   - {error}")
    
    if result['warnings']:
        print(f"\n⚠️  Warnings ({len(result['warnings'])}):")
        for warning in result['warnings']:
            print(f"   - {warning}")
    
    if not result['errors'] and not result['warnings']:
        print("\n✅ All checks passed!")
    
    print("="*60)

