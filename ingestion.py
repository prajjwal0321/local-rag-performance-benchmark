import os
import time
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

def process_and_index_pdf(pdf_path, chunk_size, chunk_overlap):
    """
    Loads a PDF, extracts pages while preserving source metadata, splits text,
    and indexes embeddings into a local Chroma vector database.
    """
    # 1. Load the PDF using LangChain's PyPDFLoader
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    
    # Extract the actual original filename to use in citations
    original_filename = os.path.basename(pdf_path)
    if original_filename == "temp.pdf":
        original_filename = "Uploaded_Research_Paper.pdf"
        
    # Explicitly ensure page metadata is cleanly structured
    for doc in docs:
        doc.metadata["source"] = original_filename
        # PyPDFLoader uses 0-indexed pages; add 1 to make it human-readable (Page 1, Page 2...)
        if "page" in doc.metadata:
            doc.metadata["page"] = int(doc.metadata["page"]) + 1
        else:
            doc.metadata["page"] = 1

    # 2. Split text based on target parameters
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    splits = text_splitter.split_documents(docs)
    
    # 3. Create Local Vector Embeddings using HuggingFace
    start_embed_time = time.time()
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    end_embed_time = time.time()
    
    embedding_latency = end_embed_time - start_embed_time
    return vectorstore, embedding_latency
