import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
import time

st.set_page_config(page_title="Local AI Research Assistant", layout="wide")
st.title("📄 Local AI Academic Document Assistant")

# Sidebar for experimental parameters (Crucial for your Research Paper!)
st.sidebar.header("🔬 Research Parameters")
chunk_size = st.sidebar.slider("Chunk Size (Tokens)", 200, 2000, 500, step=100)
chunk_overlap = st.sidebar.slider("Chunk Overlap", 20, 200, 50)

uploaded_file = st.file_uploader("Upload an Academic PDF", type=["pdf"])

if uploaded_file is not None:
    # Save the file temporarily
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    st.success("File uploaded successfully! Starting indexing...")
    
    # 1. Load the PDF
    loader = PyPDFLoader("temp.pdf")
    docs = loader.load()
    
    # 2. Split text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    splits = text_splitter.split_documents(docs)
    
    # 3. Create Local Vector Embeddings using HuggingFace
    start_embed_time = time.time()
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    end_embed_time = time.time()
    
    st.info(f"⚡ Time taken to embed and index document: {end_embed_time - start_embed_time:.2f} seconds")
    
    # 4. Set up the Retriever and Modern QA Chain
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = Ollama(model="llama3")
    
    # Define a custom system prompt for the assistant
    system_prompt = (
        "You are an expert academic research assistant. Use the following pieces of retrieved "
        "context to answer the question. If you don't know the answer, say that you don't know.\n\n"
        "{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    # Create the modern chain structure using the classic backbone
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    # User Query Section
    user_query = st.text_input("Ask a question about this research paper:")
    if user_query:
        start_query_time = time.time()
        with st.spinner("Analyzing document locally..."):
            response = rag_chain.invoke({"input": user_query})
        end_query_time = time.time()
        
        st.markdown(f"### 🤖 Answer:\n{response['answer']}")
        st.caption(f"⏱️ Response Generation Time: {end_query_time - start_query_time:.2f} seconds")
