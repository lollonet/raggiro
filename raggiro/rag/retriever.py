"""Retrieval module for RAG integration."""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np

# For embeddings
from sentence_transformers import SentenceTransformer

# For vector search
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False

class VectorRetriever:
    """Retrieves relevant chunks from vector stores for RAG."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the vector retriever.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Configure retrieval settings
        retrieval_config = self.config.get("retrieval", {})
        self.embedding_model = retrieval_config.get("embedding_model", "all-MiniLM-L6-v2")
        self.vector_db = retrieval_config.get("vector_db", "faiss")  # "faiss", "qdrant"
        self.qdrant_url = retrieval_config.get("qdrant_url", "http://localhost:6333")
        self.qdrant_collection = retrieval_config.get("qdrant_collection", "raggiro")
        self.top_k = retrieval_config.get("top_k", 5)
        
        # Initialize embedding model with error handling
        try:
            # Set torch config to avoid init_empty_weights error
            import os
            os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Avoid parallelism warnings
            
            # Check for transformer version compatibility issue
            try:
                # Try importing transformers to check version
                import transformers
                # If TORCH_ENABLE_MPS is set, it can cause issues with init_empty_weights
                if "TORCH_ENABLE_MPS" in os.environ:
                    del os.environ["TORCH_ENABLE_MPS"]
            except ImportError:
                pass
            
            # Try loading with normal method
            self.model = SentenceTransformer(self.embedding_model)
            print(f"Successfully loaded sentence transformer model: {self.embedding_model}")
        except Exception as e:
            error_msg = str(e)
            print(f"Warning: Failed to load sentence transformer model: {error_msg}")
            
            if "init_empty_weights" in error_msg:
                print("The error is related to 'init_empty_weights' which is a compatibility issue.")
                print("Trying alternative initialization method...")
                # Fallback to a more compatible model
                try:
                    # Import a simple version without compatibility issues
                    from sentence_transformers import SentenceTransformer
                    # Try with an older stable model
                    backup_model = "paraphrase-MiniLM-L6-v2"
                    print(f"Trying backup model: {backup_model}")
                    self.model = SentenceTransformer(backup_model)
                    print(f"Successfully loaded backup sentence transformer model: {backup_model}")
                    # Update the model name to reflect what was actually loaded
                    self.embedding_model = backup_model
                except Exception as e2:
                    print(f"CRITICAL: Also failed to load backup model: {str(e2)}")
                    raise RuntimeError(f"Failed to initialize sentence transformer models: {error_msg} AND {str(e2)}")
            else:
                print(f"ERROR: Failed to load sentence transformer model: {error_msg}")
                raise
        
        # Initialize vector database
        self.index = None
        self.document_lookup = {}
        
        if self.vector_db == "qdrant" and QDRANT_AVAILABLE:
            self.client = QdrantClient(url=self.qdrant_url)
    
    def load_index(self, input_path: Union[str, Path]) -> Dict:
        """Load a FAISS index and document lookup from disk.
        
        Args:
            input_path: Path to the saved index and lookup
            
        Returns:
            Load result
        """
        input_path = Path(input_path)
        
        # Only FAISS indices can be loaded
        if self.vector_db != "faiss" or not FAISS_AVAILABLE:
            return {
                "input_path": str(input_path),
                "success": False,
                "error": "Only FAISS indices can be loaded",
            }
        
        try:
            # Check if the files exist
            index_path = input_path / "index.faiss"
            lookup_path = input_path / "lookup.json"
            
            if not index_path.exists():
                return {
                    "input_path": str(input_path),
                    "success": False,
                    "error": f"Index file not found: {index_path}",
                }
            
            if not lookup_path.exists():
                return {
                    "input_path": str(input_path),
                    "success": False,
                    "error": f"Lookup file not found: {lookup_path}",
                }
            
            # Load FAISS index
            self.index = faiss.read_index(str(index_path))
            
            # Load document lookup
            with open(lookup_path, "r", encoding="utf-8") as f:
                # Convert string keys back to integers
                lookup_data = json.load(f)
                self.document_lookup = {int(k): v for k, v in lookup_data.items()}
            
            return {
                "input_path": str(input_path),
                "total_vectors": self.index.ntotal,
                "success": True,
            }
        except Exception as e:
            return {
                "input_path": str(input_path),
                "success": False,
                "error": f"Failed to load index: {str(e)}",
            }
    
    def retrieve(self, query: str, top_k: Optional[int] = None) -> Dict:
        """Retrieve relevant chunks for a query.
        
        Args:
            query: Query string
            top_k: Number of chunks to retrieve (overrides config)
            
        Returns:
            Retrieval results
        """
        if not query:
            return {
                "query": query,
                "success": False,
                "error": "Query is empty",
            }
        
        # Use provided top_k if given, otherwise use the one from config
        k = top_k if top_k is not None else self.top_k
        
        try:
            # Compute query embedding
            query_embedding = self.model.encode(query)
            
            # Retrieve from the appropriate vector database
            if self.vector_db == "faiss" and FAISS_AVAILABLE:
                result = self._retrieve_from_faiss(query_embedding, k)
            elif self.vector_db == "qdrant" and QDRANT_AVAILABLE:
                result = self._retrieve_from_qdrant(query_embedding, k)
            else:
                return {
                    "query": query,
                    "success": False,
                    "error": f"Vector database {self.vector_db} is not available",
                }
            
            # Add query to the result
            result["query"] = query
            result["success"] = True
            
            return result
        except Exception as e:
            return {
                "query": query,
                "success": False,
                "error": f"Failed to retrieve results: {str(e)}",
            }
    
    def _retrieve_from_faiss(self, query_embedding: np.ndarray, k: int) -> Dict:
        """Retrieve from FAISS index.
        
        Args:
            query_embedding: Query embedding
            k: Number of results to retrieve
            
        Returns:
            Retrieval results
        """
        if self.index is None:
            return {
                "success": False,
                "error": "No index loaded",
            }
        
        # Reshape embedding for FAISS
        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
        
        # Search the index
        distances, indices = self.index.search(query_embedding, k)
        
        # Get the chunks for the indices
        chunks = []
        
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx in self.document_lookup:
                chunk_data = self.document_lookup[idx]
                # Create result chunk with basic info
                result_chunk = {
                    "rank": i + 1,
                    "score": float(1.0 - distance / 2.0),  # Convert L2 distance to a similarity score
                    "text": chunk_data["text"],
                    "metadata": chunk_data["metadata"],
                }
                
                # Add summary if available - useful for context explanation
                if "summary" in chunk_data:
                    result_chunk["summary"] = chunk_data["summary"]
                    
                chunks.append(result_chunk)
        
        return {
            "chunks": chunks,
            "total_chunks": len(chunks),
        }
    
    def _retrieve_from_qdrant(self, query_embedding: np.ndarray, k: int) -> Dict:
        """Retrieve from Qdrant.
        
        Args:
            query_embedding: Query embedding
            k: Number of results to retrieve
            
        Returns:
            Retrieval results
        """
        # Search the collection
        search_result = self.client.search(
            collection_name=self.qdrant_collection,
            query_vector=query_embedding.tolist(),
            limit=k,
        )
        
        # Get the chunks for the results
        chunks = []
        
        for i, result in enumerate(search_result):
            # Create result chunk with basic info
            result_chunk = {
                "rank": i + 1,
                "score": float(result.score),
                "text": result.payload.get("text", ""),
                "metadata": {k: v for k, v in result.payload.items() if k not in ["text", "summary"]},
            }
            
            # Add summary if available - useful for context explanation
            if "summary" in result.payload:
                result_chunk["summary"] = result.payload["summary"]
                
            chunks.append(result_chunk)
        
        return {
            "chunks": chunks,
            "total_chunks": len(chunks),
        }