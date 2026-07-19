import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, model_name: str, max_seq_length: int = 1024):
        self.model = SentenceTransformer(model_name)
        self.model.max_seq_length = max_seq_length

    def encode(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(
            texts,
            batch_size=16,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)
