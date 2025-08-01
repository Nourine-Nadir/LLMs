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
from utils import get_data, TF_IDF_retrrierval
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




texts, metadatas = get_data(json_path)

def rag_query(question, vectorstore):
    # Retrieve relevant documents
    print(f'retrieving docs similarity...')
    docs = vectorstore.similarity_search(question, k=10)
    print(f'Docs retrieved ! ')
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
question = "Quel est la taxe d'achat d'un yacht ou bateau de plaisance ?"
if args.tf_idf:
    print('Using TF-IDF')
    _, close_texts = TF_IDF_retrrierval(texts, question, top_k=args.tf_idf_topk)
    print(f'close texts : {close_texts}')
    print(f'retrieving docs similarity in TF-IDF...')
    vectorstore = FAISS.from_texts(texts=close_texts, embedding=embedding_model)

    context = ("\n\n".join(close_texts))

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
    answer = llm.invoke(prompt)

else:
    print('No TF-IDF')
    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n"," ", ""]
    )
    texts = text_splitter.split_text("\n\n".join(texts))
    vectorstore = FAISS.from_texts(texts=texts, embedding=embedding_model)
    # Create FAISS vector store

    answer = rag_query(question, vectorstore)

print("Question:", question)
print("Answer:")
print(textwrap.fill(answer, width=80))