"""Retrieval module for RAG integration."""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np

# For embeddings
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

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
        # Set environment variables before any imports or model loading
        import os
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        self.config = config or {}
        
        # Configure retrieval settings
        retrieval_config = self.config.get("retrieval", {})
        self.embedding_model = retrieval_config.get("embedding_model", "all-MiniLM-L6-v2")
        self.vector_db = retrieval_config.get("vector_db", "faiss")  # "faiss", "qdrant"
        self.qdrant_url = retrieval_config.get("qdrant_url", "http://localhost:6333")
        self.qdrant_collection = retrieval_config.get("qdrant_collection", "raggiro")
        self.top_k = retrieval_config.get("top_k", 5)
        
        # Summary relevance filtering settings
        self.use_summary_filtering = retrieval_config.get("use_summary_filtering", True)
        self.summary_relevance_threshold = retrieval_config.get("summary_relevance_threshold", 0.4)
        self.summary_boost_factor = retrieval_config.get("summary_boost_factor", 0.2)
        
        # Initialize components
        self.index = None
        self.document_lookup = {}
        self.model = None
        
        # Check if the package is available
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            print("CRITICAL: sentence-transformers package is not available. Vector operations will not work.")
            return
        
        # Initialize sentence transformer model with robust error handling
        print(f"Loading sentence transformer model: {self.embedding_model}")
        
        try:
            # Fix for 'init_empty_weights' error: Import specific modules before loading the model
            try:
                # Make sure to import torch and transformers directly
                import torch
                import transformers
                
                # Import specific modules that might be needed for initialization
                from transformers import AutoModel, AutoTokenizer
                from transformers.modeling_utils import PreTrainedModel
            except ImportError as import_err:
                print(f"Warning: Could not import required dependencies: {str(import_err)}")
                
            # Use direct loading but with silent flag to reduce console output
            self.model = SentenceTransformer(self.embedding_model, device='cpu')
            print(f"Successfully loaded sentence transformer model: {self.embedding_model}")
        except Exception as e:
            error_msg = str(e)
            print(f"Warning: Failed to load model {self.embedding_model}: {error_msg}")
            
            # Try a known-compatible backup model with simplified loading
            try:
                backup_model = "paraphrase-MiniLM-L6-v2"
                print(f"Trying backup model: {backup_model}")
                
                # More careful loading to avoid issues
                try:
                    # Pre-check for torch and related dependencies
                    import torch
                    import transformers
                    
                    # Try with specific device setting
                    self.model = SentenceTransformer(backup_model, device='cpu')
                except ImportError:
                    # If torch isn't available, try with basic import
                    self.model = SentenceTransformer(backup_model)
                    
                print(f"Successfully loaded backup model: {backup_model}")
                self.embedding_model = backup_model
            except Exception as e2:
                self.model = None
                print(f"CRITICAL: All model loading attempts failed. Vector operations will not work.")
                print(f"Error details: {error_msg} AND {str(e2)}")
        
        if self.vector_db == "qdrant" and QDRANT_AVAILABLE:
            self.client = QdrantClient(url=self.qdrant_url)
    
    def load_index(self, input_path: Union[str, Path]) -> Dict:
        """Load a FAISS index and document lookup from disk.
        
        Args:
            input_path: Path to the directory containing the index
            
        Returns:
            Dictionary with loading status
        """
        if self.vector_db != "faiss":
            return {
                "input_path": str(input_path),
                "success": False,
                "error": "Loading from disk only supported for FAISS",
            }
        
        input_path = Path(input_path)
        
        # Check if index exists
        index_path = input_path / "index.faiss"
        lookup_path = input_path / "document_lookup.json"
        
        if not index_path.exists() or not lookup_path.exists():
            return {
                "input_path": str(input_path),
                "success": False,
                "error": "Index or document lookup not found",
            }
        
        try:
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
        
        # Check if model is available
        if self.model is None:
            return {
                "query": query, 
                "success": False,
                "error": "Embedding model not available. Vector search cannot be performed.",
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
            
            # Apply summary-based filtering and reranking if enabled
            if self.use_summary_filtering and result.get("chunks", []):
                result = self._apply_summary_filtering(query, result)
            
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
        
    def _apply_summary_filtering(self, query: str, result: Dict) -> Dict:
        """Apply filtering and reranking based on summary relevance.
        
        This method:
        1. Calculates similarity between query and each chunk's summary
        2. Filters out chunks with low summary relevance
        3. Boosts scores of chunks with high summary relevance
        
        Args:
            query: Original query string
            result: Original retrieval result
            
        Returns:
            Filtered and reranked result
        """
        chunks = result.get("chunks", [])
        if not chunks:
            return result
            
        # Extract chunks that have summaries
        chunks_with_summaries = [chunk for chunk in chunks if "summary" in chunk and chunk["summary"]]
        
        # If no summaries, return original result
        if not chunks_with_summaries:
            return result
            
        # Check if model is available
        if self.model is None:
            return result
            
        # Encode query once
        query_embedding = self.model.encode(query)
        
        # Compute similarity between query and each summary
        enhanced_chunks = []
        filtered_chunks = []
        
        for chunk in chunks:
            # If no summary, keep the chunk as is
            if "summary" not in chunk or not chunk["summary"]:
                filtered_chunks.append(chunk)
                continue
                
            # Compute summary relevance
            summary_embedding = self.model.encode(chunk["summary"])
            summary_similarity = self._compute_cosine_similarity(query_embedding, summary_embedding)
            
            # Add summary similarity to chunk for transparency
            chunk["summary_relevance"] = float(summary_similarity)
            
            # Filter low-relevance chunks
            if summary_similarity < self.summary_relevance_threshold:
                continue
                
            # Boost score based on summary relevance
            original_score = chunk["score"]
            boost = summary_similarity * self.summary_boost_factor
            boosted_score = min(1.0, original_score + boost)  # Cap at 1.0
            
            # Create enhanced chunk
            enhanced_chunk = chunk.copy()
            enhanced_chunk["original_score"] = original_score
            enhanced_chunk["score"] = boosted_score
            enhanced_chunk["score_boost"] = boost
            
            filtered_chunks.append(enhanced_chunk)
            
        # Sort by new scores
        reranked_chunks = sorted(filtered_chunks, key=lambda x: x["score"], reverse=True)
        
        # Return with the same format as original
        return {
            "chunks": reranked_chunks,
            "total_chunks": len(reranked_chunks),
            "query": result.get("query", ""),
            "success": result.get("success", True),
            "filtering_applied": True,
            "original_chunk_count": len(chunks)
        }
        
    def _compute_cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors.
        
        Args:
            v1: First vector
            v2: Second vector
            
        Returns:
            Cosine similarity (0-1)
        """
        # Ensure vectors are normalized
        v1_norm = v1 / np.linalg.norm(v1)
        v2_norm = v2 / np.linalg.norm(v2)
        
        # Compute cosine similarity
        return np.dot(v1_norm, v2_norm)