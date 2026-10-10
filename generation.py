from langchain_community.llms import Ollama
import time

def execute_rag_inference(vectorstore, top_k, user_query):
    """
    Manually retrieves top-k documents, extracts text + metadata citations,
    and invokes local Llama 3 inference under strict anti-hallucination grounding guidelines.
    """
    llm = Ollama(model="llama3")
    
    # 1. Manually pull the top-k most similar text chunks from Chroma
    retrieved_docs = vectorstore.similarity_search(user_query, k=top_k)
    
    # 2. Build the context string and compile clean citations arrays
    context_parts = []
    citations = []
    
    for i, doc in enumerate(retrieved_docs, 1):
        context_parts.append(doc.page_content)
        
        # Extract metadata tracked from ingestion
        source_name = doc.metadata.get("source", "Unknown_Document.pdf")
        page_num = doc.metadata.get("page", "Unknown")
        
        citations.append({
            "index": i,
            "source": source_name,
            "page": page_num
        })
        
    combined_context = "\n\n---\n\n".join(context_parts)
    
    # 3. Construct the clean, grounded anti-hallucination system prompt layout (Phase 3)
    system_prompt = (
        "You are an expert academic research assistant operating under strict validation constraints.\n"
        "Instructions:\n"
        "1. Answer the user question ONLY using the provided text context sections listed below.\n"
        "2. Do NOT use outside knowledge, external historical facts, or assumptions.\n"
        "3. Do NOT make up or extrapolate facts. Keep statements completely literal to the text.\n"
        "4. If the exact answer cannot be explicitly found in the context blocks, you MUST respond exactly with:\n"
        "'I could not find this information in the uploaded document.'\n\n"
        f"Retrieved Document Context:\n{combined_context}"
    )
    
    formatted_prompt = f"System: {system_prompt}\n\nHuman: {user_query}\n\nAssistant:"
    
    # 4. Invoke local model execution
    start_time = time.time()
    answer = llm.invoke(formatted_prompt)
    generation_latency = time.time() - start_time
    
    return answer, citations, generation_latency
