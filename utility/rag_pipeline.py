from typing import List, Dict, Any
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_experimental.text_splitter import SemanticChunker
import os

class SemanticRAGPipeline:
    """Complete pipeline: PDF text -> Semantic Chunking -> Vector Store -> Search"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize the complete RAG pipeline"""
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
        self.vector_store = None
        self.chunks = []
        
    def semantic_chunk_text(self, text: str) -> List[str]:
        """Perform semantic chunking using free HuggingFace embeddings"""
        # Create semantic chunker
        text_splitter = SemanticChunker(
            self.embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=95
        )
        
        # Split the text into semantic chunks
        chunks = text_splitter.create_documents([text])
        
        # Return only text content
        return [chunk.page_content for chunk in chunks]
    
    def create_vector_store(self, pdf_text: str) -> None:
        """Complete pipeline: chunk text and create vector store"""
        print("🔄 Starting semantic chunking...")
        
        # Step 1: Semantic chunking
        self.chunks = self.semantic_chunk_text(pdf_text)
        print(f"✅ Created {len(self.chunks)} semantic chunks")
        
        # Step 2: Create embeddings and index in FAISS
        print("🔄 Creating embeddings and indexing...")
        documents = [Document(page_content=chunk, metadata={"chunk_id": i}) 
                    for i, chunk in enumerate(self.chunks)]
        
        self.vector_store = FAISS.from_documents(documents, self.embeddings)
        print(f"✅ Successfully indexed {len(self.chunks)} chunks in FAISS vector store")
    
    def search(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """Search for similar chunks"""
        if not self.vector_store:
            raise ValueError("Vector store not initialized. Call create_vector_store first.")
            
        docs = self.vector_store.similarity_search(query, k=k)
        return [{"content": doc.page_content, "metadata": doc.metadata} for doc in docs]
    
    def search_with_scores(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """Search with similarity scores"""
        if not self.vector_store:
            raise ValueError("Vector store not initialized. Call create_vector_store first.")
            
        docs_with_scores = self.vector_store.similarity_search_with_score(query, k=k)
        return [{"content": doc.page_content, "score": float(score), "metadata": doc.metadata} 
                for doc, score in docs_with_scores]
    
    def save_vector_store(self, path: str = "./faiss_index") -> None:
        """Save vector store to disk"""
        if self.vector_store:
            self.vector_store.save_local(path)
            print(f"✅ Vector store saved to {path}")
        else:
            print("❌ No vector store to save")
    
    def load_vector_store(self, path: str = "./faiss_index") -> None:
        """Load vector store from disk"""
        if os.path.exists(path):
            self.vector_store = FAISS.load_local(
                path, 
                self.embeddings, 
                allow_dangerous_deserialization=True
            )
            print(f"✅ Vector store loaded from {path}")
        else:
            print(f"❌ Path {path} does not exist")
    
    def get_chunk_info(self) -> Dict[str, Any]:
        """Get information about chunks"""
        if not self.chunks:
            return {"total_chunks": 0, "avg_length": 0}
            
        lengths = [len(chunk) for chunk in self.chunks]
        return {
            "total_chunks": len(self.chunks),
            "avg_length": sum(lengths) / len(lengths),
            "min_length": min(lengths),
            "max_length": max(lengths)
        }   