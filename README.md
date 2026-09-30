# 📄 Local RAG Performance Analysis Framework

An empirical evaluation framework designed to benchmark system latency against textual chunk-size configurations in Retrieval-Augmented Generation (RAG) pipelines using 100% local consumer hardware.

## 🚀 System Architecture
- **Frontend UI:** Streamlit
- **Orchestration:** LangChain
- **Vector Database:** ChromaDB
- **Embedding Engine:** HuggingFace (`all-MiniLM-L6-v2`)
- **Local Inference LLM:** Ollama (`Llama 3`)

## 🔬 Research Focus
This project serves as the technical validation layer for the pre-print research paper: 
*"An Empirical Evaluation of Local Retrieval-Augmented Generation Pipeline Architectures on Consumer Hardware."* 

We analyze how adjusting hyper-parameters (such as token chunk sizes: 200, 500, 1000) directly scales embedding speed, indexing time, and context synthesis accuracy under zero-cost, privacy-first computing limits.

## 🛠️ Local Installation & Setup

1. Clone the project folder:
   ```bash
   git clone https://github.com
   cd local-rag-performance-benchmark
   ```
2. Activate a virtual sandbox environment and install dependencies:
   ```bash
   source venv/bin/activate
   pip install streamlit langchain-classic langchain-community chromadb pypdf langchain-huggingface sentence-transformers
   ```
3. Run the dashboard:
   ```bash
   streamlit run app.py
   ```
