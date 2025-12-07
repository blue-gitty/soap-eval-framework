"""
Configuration loader for DeepScribe Evals.
Reads settings from config.yaml so you don't need to edit core files.
Automatically loads .env file if present for secure API key management.
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any

# Try to load .env file automatically
try:
    from dotenv import load_dotenv
    # Load .env from project root (parent of src)
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        print(f"✅ Loaded environment variables from {env_path}")
except ImportError:
    # python-dotenv not installed, skip .env loading
    pass

# Config path is in root directory (parent of src)
DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"

_config_cache = None


def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Security: API keys are never loaded from config.yaml if they exist in environment variables.
    This prevents accidental exposure of keys in version control.
    
    Args:
        config_path: Path to config file (default: config.yaml in project root)
        
    Returns:
        Dict with configuration settings (API keys from env vars override config values)
    """
    global _config_cache
    
    if _config_cache is not None:
        return _config_cache
    
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH
    
    if not os.path.exists(config_path):
        print(f"Warning: Config file not found at {config_path}, using defaults")
        return get_default_config()
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Security: Override API keys from config with environment variables if they exist
    # This ensures env vars always take precedence
    if 'gemini_api_key' in config:
        env_key = os.getenv('GEMINI_API_KEY')
        if env_key:
            config['gemini_api_key'] = ''  # Clear config value, use env var instead
        elif config.get('gemini_api_key'):
            # Warn if key is in config file (security risk)
            print("⚠️  Warning: GEMINI_API_KEY found in config.yaml. Consider using environment variable instead.")
    
    if 'openai_api_key' in config:
        env_key = os.getenv('OPENAI_API_KEY')
        if env_key:
            config['openai_api_key'] = ''  # Clear config value, use env var instead
        elif config.get('openai_api_key'):
            # Warn if key is in config file (security risk)
            print("⚠️  Warning: OPENAI_API_KEY found in config.yaml. Consider using environment variable instead.")
    
    _config_cache = config
    return config


def get_default_config() -> Dict[str, Any]:
    """Return default configuration if config.yaml is missing."""
    return {
        'llm': {
            'provider': 'ollama',
            'model': 'gemma3:4b',
            'prompt_strategy': 'few-shot',
            'temperature': 0.1,
            'max_retries': 3
        },
        'openai_api_key': '',
        'evaluation': {
            'min_recall': 0.7,
            'max_hallucination_rate': 0.1,
            'min_f1': 0.7
        },
        'output': {
            'results_dir': 'results',
            'save_detailed_results': True
        }
    }


def get_llm_config() -> Dict[str, Any]:
    """Get just the LLM configuration."""
    config = load_config()
    return config.get('llm', get_default_config()['llm'])


def reload_config():
    """Force reload config from file (useful if you changed config.yaml)."""
    global _config_cache
    _config_cache = None
    return load_config()
