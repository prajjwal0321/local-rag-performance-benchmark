
Local RAG Performance Analysis Framework

An empirical evaluation framework designed to benchmark system latency against textual chunk-size configurations in Retrieval-Augmented Generation (RAG) pipelines using 100% local consumer hardware.

🔬 Core Research Question

How do varying token chunk constraints affect document data indexing timelines and local LLM generation latencies inside a privacy-preserving consumer hardware environment?

📊 Experimental Parameters

• Target Configurations: 200, 400, and 600 Tokens
• Fixed Constraints: Overlap = 50 tokens, Retrieval Depth (Top-K) = 3
• Verification Trials: 3 continuous operational runs per question block to isolate background CPU noise and calculate stable mean latencies.

🛠️ Quick Replicated Setup

1. Download Ollama and pull weights: ollama pull llama3
2. Clone repository and install requirements via pip.
3. Boot up the dashboard: streamlit run app.py
