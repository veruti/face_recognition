from pathlib import Path

import numpy as np
from src.core.setting import settings

import faiss
from faiss import IndexFlatIP, normalize_L2

# TODO: add metadata support (if possible)
class IndexRepository:
    def __init__(self):
        self.index = self._load_index()

    def _load_index(self) -> IndexFlatIP:
        if Path(settings.FAISS_INDEX_PATH).exists():
            return faiss.read_index(settings.FAISS_INDEX_PATH)
        else:
            return IndexFlatIP(settings.REID_VECTOR_DIMENSIONS)

    def save_index(self):
        faiss.write_index(self.index, settings.FAISS_INDEX_PATH)

    def add_vectors(self, vectors: np.array):
        normalize_L2(vectors)
        self.index.add(vectors)

    def search_vector(self, vector: np.array) -> tuple[float, int]:
        normalize_L2(vector)
        output = self.index.search(vector, 1)
        confidence = output[0][0][0]
        vector_id = output[1][0][0]

        return confidence, vector_id
