"""
Semantic similarity evaluation using sentence embeddings.
Reference-free metric for production monitoring.
"""

from sentence_transformers import SentenceTransformer, util
from typing import Dict


class SemanticEvaluator:
    """Evaluate semantic similarity using sentence embeddings."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity between two texts."""
        emb1 = self.model.encode(text1, convert_to_tensor=True)
        emb2 = self.model.encode(text2, convert_to_tensor=True)
        
        similarity = util.cos_sim(emb1, emb2).item()
        return similarity
    
    def evaluate_note(
        self,
        transcript: str,
        generated_note: str,
        ground_truth_note: str = None
    ) -> Dict[str, float]:
        """Evaluate semantic similarity of generated note."""
        results = {}
        
        results['transcript_to_generated'] = self.calculate_similarity(
            transcript, generated_note
        )
        
        if ground_truth_note:
            results['generated_to_ground_truth'] = self.calculate_similarity(
                generated_note, ground_truth_note
            )
            results['average_similarity'] = (
                results['transcript_to_generated'] + 
                results['generated_to_ground_truth']
            ) / 2
        else:
            results['average_similarity'] = results['transcript_to_generated']
        
        return results
