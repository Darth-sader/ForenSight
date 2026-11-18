"""Embedding package (doc §13)."""
from embeddings.similarity import (FusedEmbedder, FragmentEmbedder, cosine_similarity,
                                   knn_successors, pair_similarities)

__all__ = ["FragmentEmbedder", "FusedEmbedder", "cosine_similarity", "knn_successors",
           "pair_similarities"]
