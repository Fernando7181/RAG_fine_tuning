from langchain_community.document_loaders import PyPDFLoader
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from huggingface_hub import login
from pydantic import BaseModel
from langchain_ollama import ChatOllama
from langchain.vectorstores import Chroma
import os
import sys
import chromadb


access_token_read = ""
access_token_write = ""
login(token=access_token_read)

def pull_model(model_name: str):

    print(f"Pulling model {model_name}...")
    os.system(f"ollama pull {model_name}")


def rag_chain():
    model_name = "llama3.1"
    pull_model(model_name)
    
    model = ChatOllama(model=model_name)
    
    prompt = PromptTemplate.from_template(
        """
        <s> [Instructions] You are a friendly assistant. Answer the question based only on the following context. 
        If you don't know the answer, then reply, No Context available for this question {input}. [/Instructions] </s> 
        [Instructions] Question: {input} 
        Context: {context} 
        Answer: [/Instructions] 
        """
    )
    
    embedding = FastEmbedEmbeddings()
    vector_store = Chroma(
        persist_directory="./sql_chroma_db", 
        embedding_function=embedding
    )

    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": 3,
            "score_threshold": 0.2,
        },
    )

    document_chain = create_stuff_documents_chain(model, prompt)
    chain = create_retrieval_chain(retriever, document_chain)

    return chain

def create_retrieval_chain(retriever, document_chain):
     return {
        "retriever": retriever,
        "document_chain": document_chain,
        "run": lambda inputs: (
            document_chain.invoke({
                "context": retriever.invoke(inputs["input"]),
                "input": inputs["input"]
            })
            if retriever.invoke(inputs["input"])
            else "No relevant documents found."
        )
    }

def ask(query: str):
    chain = rag_chain()
    result = chain["run"]({"input": query})
    print(result)

ask("What is 2 + 2?")