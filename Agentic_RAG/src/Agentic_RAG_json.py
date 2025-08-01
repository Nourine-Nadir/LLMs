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
tfidf_texts = texts.copy()

text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n"," ", ""]
    )
texts = text_splitter.split_text("\n\n".join(texts))


from langchain.tools import Tool

def tfidf_tool_func(query: str) -> str:
    print(f'TF-IDF retrieval...')
    _, close_texts = TF_IDF_retrrierval(tfidf_texts, query, top_k=args.tf_idf_topk)
    print(f'TF-IDF retrieval Finished ! ')
    return "\n\n".join(close_texts)



def faiss_tool_func(query: str) -> str:
    print(f'Creating embbedding ...')
    vectorstore = FAISS.from_texts(texts=texts, embedding=embedding_model)
    print(f'Finished embbedding ! ')
    print(f'faiss vectorstore searching...')
    faiss_results = vectorstore.similarity_search(query, k=5)
    print(f'faiss vectorstore Finished !')
    return "\n\n".join([doc.page_content for doc in faiss_results])

tfidf_tool = Tool(
    name="TFIDF Retriever",
    func=tfidf_tool_func,
    description="Useful for retrieving relevant legal texts using keyword matching"
)

faiss_tool = Tool(
    name="FAISS Retriever",
    func=faiss_tool_func,
    description="Useful for semantic similarity-based retrieval of laws"
)


# Example usage
question = "Quand est-ce que la peine de mort est applique ?  "


from langchain.agents import initialize_agent
from langchain.agents.agent_types import AgentType

tools = [tfidf_tool, faiss_tool]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True
)
# Create FAISS vector store
# docs = vectorstore.similarity_search(question, k=10)
# # lambda_mult : degree of diversity among the results 0 = max diversity, 1 = min diversity
#
# print("==== Retrieved Documents (Context Chunks) ====")
# for i, doc in enumerate(docs, 1):
#     print(f"\n--- Document {i} (Length: {len(doc.page_content)} chars) ---")
#     print(doc.page_content)
#     print("Metadata:", doc.metadata)
#     print("-" * 50)
#
# # Combine context from retrieved documents
# context = "\n\n".join([doc.page_content for doc in docs])

# Create prompt with context
prompt = """Follow these rules:
    1. Answer in french
    2. Reformulate the question correctly
    3. Synthesize the answer


Use tools to gather relevant context before answering.
"""

while True:
    question = input("Question (or 'exit' to quit): ")
    if question.lower() == 'exit':
        break
    agent_response = agent.invoke(prompt + question)
    # 5. Convert final answer to French
    french_answer = llm.invoke(
        f"Traduis et reformule en français: {agent_response['output']}"
    )
    print(textwrap.fill(french_answer, width=80))


