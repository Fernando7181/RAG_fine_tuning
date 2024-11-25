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


def ingest():
    loader = PyPDFLoader("assets/2409.12122v1.pdf")
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(pages)
    print(f"Split {len(pages)} pages into {len(chunks)} chunks.")

   
    embedding = FastEmbedEmbeddings()

    Chroma.from_documents(
        documents=chunks, 
        embedding=embedding,
        persist_directory="./sql_chroma"
)
ingest()