# Local RAG Performance Analysis Framework

An empirical evaluation framework designed to benchmark system latency metrics against token-aware chunk configurations in Retrieval-Augmented Generation (RAG) pipelines using 100% local consumer hardware.

## 🔬 Core Research Question
> *How do varying token chunk constraints affect document data indexing timelines and local LLM generation latencies inside a privacy-preserving consumer hardware environment?*

## 📊 Experimental Parameters
- **Target Configurations Evaluated:** 200 Tokens, 400 Tokens, and 600 Tokens
- **Fixed System Constraints:** Chunk Overlap = 50 tokens, Retrieval Depth (Top-K) = 3
- **Verification Trials:** 3 continuous inference query trials per configuration block to log runtime cache fluctuations, while database vector indexing is performed exactly once per parameter tier.

## 🛠️ Complete Installation & Local Replication

1. **Prerequisites:** Setup [Ollama](https://ollama.com) and download the required weights:
   ```bash
   ollama pull llama3
   ```
2. **Clone the Repository Workspace:**
   ```bash
   git clone https://github.com
   cd local-rag-performance-benchmark
   ```
3. **Setup Dependencies:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install streamlit langchain-classic langchain-community chromadb pypdf langchain-huggingface sentence-transformers pandas matplotlib seaborn tiktoken
   ```
4. **Boot Up the Dashboard:**
   ```bash
   streamlit run app.py
   ```

## 🔬 Framework Limitations & Scope
This independent benchmarking workbench tracks computational processing latencies within an isolated hardware testing environment. The recorded data sheets evaluate structural context window grounding profiles rather than an automated quantitative answer-quality accuracy rating.
