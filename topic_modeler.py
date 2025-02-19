import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

def perform_topic_modeling(docs, n_topics=3, n_top_words=5):
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(docs)
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(X)
    feature_names = vectorizer.get_feature_names_out()
    topics = {}
    for idx, topic in enumerate(lda.components_):
        top_features = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
        topics[f"Topic {idx+1}"] = top_features
    return topics

if __name__ == "__main__":
    docs = []
    n = int(input("Enter the number of documents: "))
    for i in range(n):
        doc = input(f"Enter document {i+1}: ")
        docs.append(doc)
    topics = perform_topic_modeling(docs)
    for topic, words in topics.items():
        print(f"{topic}: {', '.join(words)}")
