from enum import Enum

from sentence_transformers import SentenceTransformer


class Model(Enum):
    ALL = "all-MiniLM-L6-v2"
    paraphrase = "paraphrase-MiniLM-L6-v2"


class Embedder:
    def __init__(self, model: Model = Model.ALL):
        self.model = SentenceTransformer(model.value)

    def embed(self, sentence: str):
        return self.model.encode(sentence)

    def embed_batch(self, sentences: list[str]):
        return self.model.encode(sentences)
