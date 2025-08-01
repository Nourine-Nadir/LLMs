import json
def get_data(json_path):
    texts = []
    metadatas = []

    with open(json_path, 'r', encoding='utf-8') as f:
        articles = json.load(f)
    for article in articles:
        for key, content in article.items():
            if key == "metadata":
                current_meta = content
                metadatas.append(current_meta)
                texts.append('No Text')


            else:
                text = f"{key}\n{content}"
                texts.append(text)
                metadatas.append('No Metadata')

    return texts, metadatas

# Find closest Articles :
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def TF_IDF_retrrierval(texts:list, query:str, top_k:int = 10)-> list:
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)  # shape: (num_docs, num_terms)
    question_vec = vectorizer.transform([query])  # shape: (1, num_terms)

    # Step 3: Compute cosine similarity between question and all texts
    cosine_sim = cosine_similarity(question_vec, tfidf_matrix).flatten()

    # Step 4: Get top k most similar documents
    top_indices = np.argsort(cosine_sim)[-top_k:][::-1]  # descending order

    # Step 5: Return the results
    results_dict = [(texts[i], cosine_sim[i]) for i in top_indices]
    texts = [(texts[i]) for i in top_indices]

    # print("==== Top TF-IDF Matches ====")
    # for rank, (text, score) in enumerate(results, 1):
    #     print(f"\n--- Rank {rank} (Score: {score:.4f}) ---")
    #     print(text[:500])  # Show first 500 chars
    #     print("-" * 50)

    return results_dict, texts
