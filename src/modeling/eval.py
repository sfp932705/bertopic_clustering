import numpy as np
from bertopic import BERTopic
from gensim import corpora
from gensim.models.coherencemodel import CoherenceModel


def get_coherence_score(topic_model: BERTopic, docs: list[str], total_topics: int):
    preprocessed_docs = topic_model._preprocess_text(np.array(docs))
    tokenizer = topic_model.vectorizer_model.build_tokenizer()
    words = topic_model.vectorizer_model.get_feature_names_out()
    tokens = [tokenizer(doc) for doc in preprocessed_docs]
    dictionary = corpora.Dictionary(tokens)
    corpus = [dictionary.doc2bow(token) for token in tokens]
    topic_words = [
        [dictionary.token2id[w] for w in words if w in dictionary.token2id]
        for _ in range(total_topics)
    ]
    coherence_model = CoherenceModel(
        topics=topic_words,
        texts=tokens,
        corpus=corpus,
        dictionary=dictionary,
        coherence="c_v",
        processes=1,
    )
    coherence = coherence_model.get_coherence()
    return coherence
