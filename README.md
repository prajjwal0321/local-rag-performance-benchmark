# 📄 Local RAG Performance Analysis Framework

An empirical evaluation framework designed to analyze how structural text chunk segmentation boundaries influence index generation times and inference latencies within a 100% offline, local Retrieval-Augmented Generation (RAG) pipeline.

## 🔬 Core Research Question
> *How do varying token chunk constraints affect document data indexing timelines and local LLM generation latencies inside a privacy-preserving consumer hardware environment?*

## 🚀 System Architecture Flowchart

## 📊 Experimental Parameters & Dataset
- **Evaluation Document:** *ScienceAgentBench* Technical Manuscript (13 Pages)
- **Target Configurations Evaluated:** 200 Tokens, 400 Tokens, and 600 Tokens
- **Fixed System Constraints:** Chunk Overlap = 50 tokens, Retrieval Depth (Top-K) = 3
- **Verification Trials:** 3 continuous operational runs per question block to isolate background CPU noise and calculate mean standard deviation values.

## 🛠️ Complete Installation & Local Replication

1. **Prerequisites:** Install [Ollama](https://ollama.com) on your local machine and fetch the required model weights via terminal:
   ```bash
   ollama pull llama3
   ```
2. **Clone the Workspace:**
   ```bash
   git clone https://github.com
   cd local-rag-performance-benchmark
   ```
3. **Setup Virtual Sandbox & Libraries:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install streamlit langchain-classic langchain-community chromadb pypdf langchain-huggingface sentence-transformers pandas matplotlib seaborn
   ```
4. **Boot Up the Dashboard:**
   ```bash
   streamlit run app.py
   ```

## 🔬 Experimental Limitations
This performance benchmark evaluation was conducted strictly within an isolated consumer hardware environment using the Llama 3 architecture. System latency vectors and processing timelines recorded in this repository reflect specific hardware caching characteristics and may not generalize uniformly to deep multi-model server clusters or cloud API networks.

