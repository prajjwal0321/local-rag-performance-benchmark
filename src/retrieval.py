def configure_retriever(vectorstore, top_k=3):
    """
    Configures the vectorstore retriever component with a variable depth constraint.
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    return retriever
