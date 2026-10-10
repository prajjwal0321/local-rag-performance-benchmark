from langchain_community.llms import Ollama
import time

def execute_rag_inference(vectorstore, top_k, user_query, model_name="llama3", temperature=0.0):
    """
    Retrieves top-k documents and invokes local inference under strict 
    grounding guidelines using adjustable hardware parameters.
    """
    # BINDING THE ACTUAL USER CONTROLS DIRECTLY TO OLLAMA (Issue 5 Fix)
    llm = Ollama(model=model_name, temperature=temperature)
    
    # 1. Similarity index extraction lookup
    retrieved_docs = vectorstore.similarity_search(user_query, k=top_k)
    
    context_parts = []
    citations = []
    
    for i, doc in enumerate(retrieved_docs, 1):
        context_parts.append(doc.page_content)
        source_name = doc.metadata.get("source", "Unknown_Document.pdf")
        page_num = doc.metadata.get("page", "Unknown")
        
        citations.append({
            "index": i,
            "source": source_name,
            "page": page_num
        })
        
    combined_context = "\n\n---\n\n".join(context_parts)
    
    # 2. Strict grounding instructions template
    system_prompt = (
        "You are an expert academic research assistant operating under strict validation constraints.\n"
        "Instructions:\n"
        "1. Answer the user question ONLY using the provided text context sections listed below.\n"
        "2. Do NOT use outside knowledge or assumptions.\n"
        "3. If the exact answer cannot be explicitly found in the context blocks, you MUST respond exactly with:\n"
        "'I could not find this information in the uploaded document.'\n\n"
        f"Retrieved Document Context:\n{combined_context}"
    )
    
    formatted_prompt = f"System: {system_prompt}\n\nHuman: {user_query}\n\nAssistant:"
    
    # 3. Model tracking calculation loop
    start_time = time.time()
    answer = llm.invoke(formatted_prompt)
    generation_latency = time.time() - start_time
    
    return answer, citations, generation_latency
