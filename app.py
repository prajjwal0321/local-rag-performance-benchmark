import streamlit as st
import os
import time

# Explicit path tracking fix for local execution mapping
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

# Import modular processing layers safely
from ingestion import process_and_index_pdf
from retrieval import configure_retriever
from generation import execute_rag_inference

st.set_page_config(
    page_title="Local AI Research Assistant", 
    layout="centered",
    initial_sidebar_state="expanded"
)

st.title("📄 Local AI Academic Document Assistant")
st.markdown("---")

# Sidebar - Configuration Controls
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

# Monitor parameter state changes to force a re-index if sliders move (Issue 5 Fix)
current_params = f"{chunk_size}_{chunk_overlap}"
if "last_compiled_params" not in st.session_state:
    st.session_state.last_compiled_params = ""

st.subheader("1. Ingest Academic Artifacts")
uploaded_file = st.file_uploader("Upload target manuscript PDF for system parsing", type=["pdf"])

temp_pdf_path = os.path.join(BASE_DIR, "temp.pdf")

if uploaded_file is not None:
    try:
        # Re-index if either the file name OR the chunk parameters change
        file_changed = ("current_file" not in st.session_state or st.session_state.current_file != uploaded_file.name)
        params_changed = (st.session_state.last_compiled_params != current_params)
        
        if file_changed or params_changed:
            with open(temp_pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
                
            if os.path.getsize(temp_pdf_path) == 0:
                st.error("❌ Ingestion Failure: File is empty or corrupted.")
            else:
                st.success(f"✅ Target file '{uploaded_file.name}' staged! Rebuilding index maps...")
                
                with st.spinner("Recomputing vector space matrix..."):
                    vectorstore, embedding_latency = process_and_index_pdf(
                        temp_pdf_path, chunk_size, chunk_overlap
                    )
                    st.session_state.vectorstore = vectorstore
                    st.session_state.embedding_latency = embedding_latency
                    st.session_state.current_file = uploaded_file.name
                    st.session_state.last_compiled_params = current_params
                    
                st.info(f"⏱️ Vector Database Indexing Latency: {st.session_state.embedding_latency:.2f} seconds")
                
    except Exception as e:
        st.error(f"❌ Ingestion Error: {str(e)}")

    if "vectorstore" in st.session_state:
        st.markdown("---")
        st.subheader("2. Execute Grounded Knowledge Extraction")
        user_query = st.text_input("Enter your research question:")
        
        if user_query:
            if user_query.strip() != "":
                with st.spinner("Processing local inference loops..."):
                    try:
                        # PASSING ALL ACTIVE MODEL CONTROLS DOWN (Issue 5 Fix)
                        answer, citations, gen_latency = execute_rag_inference(
                            st.session_state.vectorstore, 
                            top_k, 
                            user_query,
                            selected_model,
                            temperature
                        )
                        
                        st.markdown("#### 🤖 Grounded System Response")
                        st.info(answer)
                        st.caption(f"⏱️ Inference Processing Latency: {gen_latency:.2f} seconds")
                        
                        st.markdown("---")
                        st.markdown("#### 📄 Verifiable Academic Citations")
                        
                        seen_citations = set()
                        has_citations = False
                        
                        for cite in citations:
                            cite_str = f"{cite['source']} — *Page {cite['page']}*"
                            if cite_str not in seen_citations:
                                if "I could not find this information" not in answer:
                                    st.markdown(f"**[{cite['index']}]** {cite['source']} — *Page {cite['page']}*")
                                    has_citations = True
                                seen_citations.add(cite_str)
                                
                        if not has_citations:
                            st.caption("ℹ️ No document source citations mapped for this query response.")
                            
                    except Exception as e:
                        st.error(f"❌ System Inference Error: {str(e)}")
