# rag_engine.py
from langchain.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

def create_vectorstore(directory_path='data', index_path='faiss_index'):
    documents = []
    for file in os.listdir(directory_path):
        if file.endswith('.pdf'):
            loader = PyPDFLoader(os.path.join(directory_path, file))
            documents.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(index_path)
