from typing import List
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings  # Updated import
from langchain.schema import Document

def semantic_chunk_text(text: str, return_text_only: bool = True) -> List[str] | List[Document]:
    """
    Perform semantic chunking using free HuggingFace embeddings
    
    Args:
        text (str): The input text from PDF to be chunked
        return_text_only (bool): If True, returns only text content without metadata
        
    Returns:
        List[str] or List[Document]: List of chunked text or Document objects
    """
    
    # Initialize HuggingFace embeddings (free) - Updated import to avoid deprecation
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Create semantic chunker
    text_splitter = SemanticChunker(
        embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=95
    )
    
    # Split the text into semantic chunks
    chunks = text_splitter.create_documents([text])
    
    # Return only text content if requested
    if return_text_only:
        return [chunk.page_content for chunk in chunks]
    else:
        return chunks