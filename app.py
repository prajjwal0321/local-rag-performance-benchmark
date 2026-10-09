import streamlit as st
import os
import time

# Import our system modular components safely
from src.ingestion import process_and_index_pdf
from src.retrieval import configure_retriever
from src.generation import execute_rag_inference

# Configure a clean, professional, academic UI theme configuration
st.set_page_config(
    page_title="Local AI Research Assistant", 
    layout="centered",
    initial_sidebar_state="expanded"
)

st.title("📄 Local AI Academic Document Assistant")
st.markdown("---")

# Sidebar - Structured System Configuration Controls 
st.sidebar.header("🔬 Pipeline Parameters")
st.sidebar.markdown("---")

st.sidebar.subheader("🔀 Retrieval Configuration")
chunk_size = st.sidebar.slider("Chunk Size (Tokens)", 200, 2000, 500, step=100)
chunk_overlap = st.sidebar.slider("Chunk Overlap (Tokens)", 20, 200, 50)
top_k = st.sidebar.slider("Retrieval Depth (Top-K)", 1, 8, 3)

st.sidebar.markdown("---")
st.sidebar.subheader("🤖 Model Configuration")
selected_model = st.sidebar.selectbox("Local Inference LLM", ["llama3"])
temperature = st.sidebar.slider("Inference Temperature", 0.0, 1.0, 0.0, step=0.1)

# Primary Document Drop Zone Interface Layer
st.subheader("1. Ingest Academic Artifacts")
uploaded_file = st.file_uploader("Upload target manuscript PDF for system parsing", type=["pdf"])

temp_pdf_path = "temp.pdf"

# Phase 16: Robust Error Handling & Session State Validation Guardrails
if uploaded_file is not None:
    try:
        # Check if a brand-new file is dropped to prevent redundant embedding operations
        if "current_file" not in st.session_state or st.session_state.current_file != uploaded_file.name:
            with open(temp_pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
                
            # Verify the file is not empty or corrupted before indexing
            if os.path.getsize(temp_pdf_path) == 0:
                st.error("❌ Ingestion Failure: The uploaded PDF file appears to be empty or corrupted.")
            else:
                st.success(f"✅ Target file '{uploaded_file.name}' staged successfully! Initializing pipeline indexing...")
                
                with st.spinner("Executing dynamic text segmentation and mapping vector space matrix..."):
                    vectorstore, embedding_latency = process_and_index_pdf(
                        temp_pdf_path, chunk_size, chunk_overlap
                    )
                    
                    # Lock data indices securely into persistent session container memory
                    st.session_state.vectorstore = vectorstore
                    st.session_state.embedding_latency = embedding_latency
                    st.session_state.current_file = uploaded_file.name
                    
                st.info(f"⏱️ Vector Database Indexing Latency: {st.session_state.embedding_latency:.2f} seconds")
                
    except Exception as e:
        st.error(f"❌ Critical Pipeline Failure during Ingestion: {str(e)}")
        st.info("💡 Troubleshooting tip: Ensure the Ollama background daemon service is fully running on your Mac.")

    # 2. Query Inference Interface Layer - Only unlocks if database mapping is successful
    if "vectorstore" in st.session_state:
        st.markdown("---")
        st.subheader("2. Execute Grounded Knowledge Extraction")
        user_query = st.text_input("Enter your research question regarding the document content:")
        
        if user_query:
            if user_query.strip() == "":
                st.warning("⚠️ Query Validation Warning: Please input text characters before running query evaluation loops.")
            else:
                with st.spinner("Retrieving local similarity records and processing inference loops..."):
                    try:
                        # Pass all variable sliders down to the localized inference engine
                        answer, citations, generation_latency = execute_rag_inference(
                            st.session_state.vectorstore, top_k, user_query
                        )
                        
                        # Render Clean Academic Terminal Interface Layout 
                        st.markdown("#### 🤖 Grounded System Response")
                        st.info(answer)
                        st.caption(f"⏱️ Inference Engine Response Latency: {generation_latency:.2f} seconds")
                        
                        # Render Verified System Citations Metadata Array
                        st.markdown("---")
                        st.markdown("#### 📄 Verifiable Academic Citations")
                        
                        seen_citations = set()
                        has_citations = False
                        
                        for cite in citations:
                            cite_str = f"{cite['source']} — *Page {cite['page']}*"
                            if cite_str not in seen_citations:
                                # Ensure we don't display citations if the model answered with a fallback statement
                                if "I could not find this information" not in answer:
                                    st.markdown(f"**[{cite['index']}]** {cite['source']} — *Page {cite['page']}*")
                                    has_citations = True
                                seen_citations.add(cite_str)
                                
                        if not has_citations:
                            st.caption("ℹ️ No document source citations mapped for this query response.")
                            
                    except Exception as e:
                        st.error(f"❌ System Inference Error: {str(e)}")
                        st.info("💡 Troubleshooting tip: Verify your local Llama 3 model weights are initialized via terminal.")
e.g., Refactor UI layer to add robust parameter controls and pipeline error configurations
