"""
Embeddings module for semantic similarity.
Supports Gemini, OpenAI, and local SentenceTransformer.
"""

import os
import numpy as np
from typing import List, Union

# Lazy-loaded clients
_gemini_model = None
_openai_client = None
_local_model = None


def get_embeddings_config():
    """Load embeddings config from config.yaml."""
    from config_loader import load_config
    config = load_config()
    return config.get('embeddings', {})


def get_gemini_api_key():
    """Get Gemini API key from env or config."""
    key = os.getenv('GEMINI_API_KEY')
    if not key:
        from config_loader import load_config
        config = load_config()
        key = config.get('gemini_api_key', '')
    return key


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    dot = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


def get_gemini_embeddings(texts: List[str], model: str = "text-embedding-004") -> np.ndarray:
    """Get embeddings from Google Gemini API."""
    global _gemini_model
    
    import google.generativeai as genai
    
    # Optimization: Configure only once
    if not hasattr(get_gemini_embeddings, "_configured"):
        api_key = get_gemini_api_key()
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY environment variable not set. "
                "Set it via: export GEMINI_API_KEY='your-key' or create .env file"
            )
        genai.configure(api_key=api_key)
        get_gemini_embeddings._configured = True
    
    # Get embeddings for each text
    embeddings = []
    for text in texts:
        result = genai.embed_content(
            model=f"models/{model}",
            content=text,
            task_type="SEMANTIC_SIMILARITY"
        )
        embeddings.append(result['embedding'])
    
    return np.array(embeddings)


def get_openai_embeddings(texts: List[str], model: str = "text-embedding-3-small") -> np.ndarray:
    """Get embeddings from OpenAI API."""
    global _openai_client
    
    import openai
    
    if _openai_client is None:
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            from config_loader import load_config
            config = load_config()
            api_key = config.get('openai_api_key', '')
        _openai_client = openai.OpenAI(api_key=api_key)
    
    response = _openai_client.embeddings.create(
        model=model,
        input=texts
    )
    
    embeddings = [item.embedding for item in response.data]
    return np.array(embeddings)


def get_local_embeddings(texts: List[str], model: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """Get embeddings from local SentenceTransformer."""
    global _local_model
    
    if _local_model is None:
        from sentence_transformers import SentenceTransformer
        _local_model = SentenceTransformer(model)
    
    embeddings = _local_model.encode(texts, convert_to_numpy=True)
    return embeddings


def get_embeddings(
    texts: List[str],
    provider: str = None,
    model: str = None
) -> np.ndarray:
    """
    Get embeddings using configured provider.
    
    Args:
        texts: List of texts to embed
        provider: "gemini", "openai", or "local" (default: from config)
        model: Model name (default: from config)
    
    Returns:
        numpy array of shape (len(texts), embedding_dim)
    """
    config = get_embeddings_config()
    
    if provider is None:
        provider = config.get('provider', 'gemini')
    if model is None:
        model = config.get('model', 'text-embedding-004')
    
    if provider == "gemini":
        return get_gemini_embeddings(texts, model)
    elif provider == "openai":
        return get_openai_embeddings(texts, model)
    elif provider == "local":
        return get_local_embeddings(texts, model)
    else:
        raise ValueError(f"Unknown embeddings provider: {provider}")


def compute_similarity_matrix(
    texts1: List[str],
    texts2: List[str],
    provider: str = None,
    model: str = None
) -> np.ndarray:
    """
    Compute pairwise cosine similarity between two lists of texts.
    
    Args:
        texts1: First list of texts
        texts2: Second list of texts
        
    Returns:
        numpy array of shape (len(texts1), len(texts2)) with similarities
    """
    if not texts1 or not texts2:
        return np.array([])
    
    # Get embeddings for both lists
    emb1 = get_embeddings(texts1, provider, model)
    emb2 = get_embeddings(texts2, provider, model)
    
    # Normalize for cosine similarity
    emb1_norm = emb1 / np.linalg.norm(emb1, axis=1, keepdims=True)
    emb2_norm = emb2 / np.linalg.norm(emb2, axis=1, keepdims=True)
    
    # Compute similarity matrix
    similarity = np.dot(emb1_norm, emb2_norm.T)
    
    return similarity


def get_similarity_threshold() -> float:
    """Get similarity threshold from config."""
    config = get_embeddings_config()
    return config.get('similarity_threshold', 0.75)


# Example usage:
# embeddings = get_embeddings(["joint pain", "knee problems"])
# sim_matrix = compute_similarity_matrix(["pain in knee"], ["knee pain", "headache"])
