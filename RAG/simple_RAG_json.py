from os import close

from langchain_community.vectorstores import FAISS
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM  # Note the changed class name
from langchain_text_splitters import RecursiveCharacterTextSplitter
import textwrap
import json
import os
from parser import Parser
from args_config import PARSER_CONFIG
try:
    parser = Parser(prog='Design Of Experiments',
                    description='Tools for design of experiments purposes')
    args = parser.get_args(
        PARSER_CONFIG
    )
except ValueError as e:
    args = None
    print(e)

# Initialize models
embedding_model = OllamaEmbeddings(model=args.embedding_model_name)
llm = OllamaLLM(model=args.model_name)  # Using OllamaLLM instead of Ollama

json_path = args.data_filepath

with open(json_path, 'r', encoding='utf-8') as f:
    articles = json.load(f)

texts = []
metadatas = []

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

# Find closest Articles :
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def TF_IDF_retrrierval(texts:list, query:str, top_k:int = 10)-> list:
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)  # shape: (num_docs, num_terms)
    question_vec = vectorizer.transform([question])  # shape: (1, num_terms)

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




def rag_query(question, vectorstore):
    # Retrieve relevant documents
    docs = vectorstore.similarity_search(question, k=10)
    # lambda_mult : degree of diversity among the results 0 = max diversity, 1 = min diversity

    print("==== Retrieved Documents (Context Chunks) ====")
    for i, doc in enumerate(docs, 1):
        print(f"\n--- Document {i} (Length: {len(doc.page_content)} chars) ---")
        print(doc.page_content)
        print("Metadata:", doc.metadata)
        print("-" * 50)

    # Combine context from retrieved documents
    context = "\n\n".join([doc.page_content for doc in docs])

    # Create prompt with context
    prompt = f"""
    This are the three rules to follow before answering
    1. Answer in french
    2. Reformulate the question correctly
    3. Synthesize the answer
    Answer the question following exclusively this context:


    {context}


    Question: {question}

    """

    # Generate answer
    answer = llm.invoke(prompt)

    return answer


# Example usage
question = "Quand est-ce que la peine de mort est applique ?  "
if args.tf_idf:
    print('Using TF-IDF')
    _, close_texts = TF_IDF_retrrierval(texts, question, top_k=args.tf_idf_topk)
    print(f'close texts : {close_texts}')
    vectorstore = FAISS.from_texts(texts=close_texts, embedding=embedding_model)
#
# else:
#     print('No TF-IDF')
#     # Split into chunks
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=1000,
#         chunk_overlap=200,
#         separators=["\n\n"," ", ""]
#     )
#     texts = text_splitter.split_text(texts)
#     vectorstore = FAISS.from_texts(texts=texts, embedding=embedding_model)


# Create FAISS vector store
#
answer = rag_query(question, vectorstore)

print("Question:", question)
print("Answer:")
print(textwrap.fill(answer, width=80))