import os
from langchain_community.vectorstores import Chroma
from src.config import get_embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader

persist_directory = "./chroma_db"

def build_vectorstore(urls):
    if not urls:
        return None
    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50
    )
    doc_splits = text_splitter.split_documents(docs_list)
    
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name="crag_collection",
        embedding=get_embeddings(),
        persist_directory=persist_directory,
    )
    
    return vectorstore.as_retriever()

def get_retriever():
    if not os.path.exists(persist_directory):
        return None
    vectorstore = Chroma(
        collection_name="crag_collection",
        embedding_function=get_embeddings(),
        persist_directory=persist_directory,
    )
    return vectorstore.as_retriever()
