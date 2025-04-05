"""Vector indexing module for RAG integration."""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
from tqdm import tqdm

# For embeddings
from sentence_transformers import SentenceTransformer

# For vector storage
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

class VectorIndexer:
    """Indexes processed documents into vector stores for RAG."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the vector indexer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Configure indexing settings
        indexing_config = self.config.get("indexing", {})
        self.chunk_level = indexing_config.get("chunk_level", "chunks")  # "chunks", "paragraphs", "sections"
        self.embedding_model = indexing_config.get("embedding_model", "all-MiniLM-L6-v2")
        self.vector_db = indexing_config.get("vector_db", "faiss")  # "faiss", "qdrant"
        self.qdrant_url = indexing_config.get("qdrant_url", "http://localhost:6333")
        self.qdrant_collection = indexing_config.get("qdrant_collection", "raggiro")
        
        # Initialize embedding model
        self.model = SentenceTransformer(self.embedding_model)
        
        # Initialize vector database
        self.index = None
        self.document_lookup = {}
        
        if self.vector_db == "qdrant" and QDRANT_AVAILABLE:
            self.client = QdrantClient(url=self.qdrant_url)
            
            # Create collection if it doesn't exist
            try:
                self.client.get_collection(self.qdrant_collection)
            except:
                self.client.create_collection(
                    collection_name=self.qdrant_collection,
                    vectors_config=models.VectorParams(
                        size=self.model.get_sentence_embedding_dimension(),
                        distance=models.Distance.COSINE,
                    ),
                )
    
    def _extract_chunks(self, document: Dict) -> List[Dict]:
        """Extract chunks from a document based on configuration.
        
        Args:
            document: Document dictionary
            
        Returns:
            List of chunks with text and metadata
        """
        chunks = []
        
        if self.chunk_level == "chunks" and "chunks" in document:
            # Use the pre-segmented chunks
            for i, chunk in enumerate(document["chunks"]):
                chunks.append({
                    "id": chunk.get("id", f"chunk_{i}"),
                    "text": chunk["text"],
                    "metadata": {
                        "document_id": document["metadata"]["file"]["hash"],
                        "document_title": document["metadata"].get("title", ""),
                        "document_path": document["metadata"]["file"]["path"],
                        "chunk_id": chunk.get("id", f"chunk_{i}"),
                        "sections": [seg["text"] for seg in chunk.get("segments", []) if seg["type"] == "header"],
                    }
                })
        elif self.chunk_level == "paragraphs" and "segments" in document:
            # Use paragraph segments
            for i, segment in enumerate(document["segments"]):
                if segment["type"] == "paragraph":
                    chunks.append({
                        "id": f"para_{i}",
                        "text": segment["text"],
                        "metadata": {
                            "document_id": document["metadata"]["file"]["hash"],
                            "document_title": document["metadata"].get("title", ""),
                            "document_path": document["metadata"]["file"]["path"],
                            "section": segment.get("section", ""),
                        }
                    })
        elif self.chunk_level == "sections" and "segments" in document:
            # Use section segments
            for i, segment in enumerate(document["segments"]):
                if segment["type"] == "section":
                    chunks.append({
                        "id": f"section_{i}",
                        "text": segment["text"],
                        "metadata": {
                            "document_id": document["metadata"]["file"]["hash"],
                            "document_title": document["metadata"].get("title", ""),
                            "document_path": document["metadata"]["file"]["path"],
                            "section_title": segment.get("title", ""),
                        }
                    })
        else:
            # Fallback to the full text
            chunks.append({
                "id": "full_text",
                "text": document["text"],
                "metadata": {
                    "document_id": document["metadata"]["file"]["hash"],
                    "document_title": document["metadata"].get("title", ""),
                    "document_path": document["metadata"]["file"]["path"],
                }
            })
        
        return chunks
    
    def index_document(self, document_path: Union[str, Path]) -> Dict:
        """Index a processed document.
        
        Args:
            document_path: Path to the processed document JSON file
            
        Returns:
            Indexing result
        """
        document_path = Path(document_path)
        
        # Check if the document exists
        if not document_path.exists():
            return {
                "document_path": str(document_path),
                "success": False,
                "error": "Document does not exist",
            }
        
        # Load document
        try:
            with open(document_path, "r", encoding="utf-8") as f:
                document = json.load(f)
        except Exception as e:
            return {
                "document_path": str(document_path),
                "success": False,
                "error": f"Failed to load document: {str(e)}",
            }
        
        # Extract chunks
        chunks = self._extract_chunks(document)
        
        if not chunks:
            return {
                "document_path": str(document_path),
                "success": False,
                "error": "No chunks extracted from document",
            }
        
        # Compute embeddings
        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.model.encode(texts)
        
        # Index the document
        result = {
            "document_path": str(document_path),
            "chunks_indexed": len(chunks),
            "success": False,
            "error": None,
        }
        
        try:
            if self.vector_db == "faiss" and FAISS_AVAILABLE:
                result.update(self._index_with_faiss(chunks, embeddings))
            elif self.vector_db == "qdrant" and QDRANT_AVAILABLE:
                result.update(self._index_with_qdrant(chunks, embeddings))
            else:
                result["error"] = f"Vector database {self.vector_db} is not available"
        except Exception as e:
            result["error"] = f"Failed to index document: {str(e)}"
        
        # Set success flag
        result["success"] = result["error"] is None
        
        return result
    
    def _index_with_faiss(self, chunks: List[Dict], embeddings: np.ndarray) -> Dict:
        """Index chunks with FAISS.
        
        Args:
            chunks: List of chunks with text and metadata
            embeddings: Chunk embeddings
            
        Returns:
            Indexing result
        """
        if self.index is None:
            # Initialize FAISS index
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
        
        # Get current index size
        current_size = self.index.ntotal
        
        # Add embeddings to the index
        self.index.add(embeddings.astype(np.float32))
        
        # Store chunk metadata in lookup dictionary
        for i, chunk in enumerate(chunks):
            self.document_lookup[current_size + i] = {
                "text": chunk["text"],
                "metadata": chunk["metadata"],
            }
        
        return {
            "vector_db": "faiss",
            "total_vectors": self.index.ntotal,
        }
    
    def _index_with_qdrant(self, chunks: List[Dict], embeddings: np.ndarray) -> Dict:
        """Index chunks with Qdrant.
        
        Args:
            chunks: List of chunks with text and metadata
            embeddings: Chunk embeddings
            
        Returns:
            Indexing result
        """
        # Get current collection size
        collection_info = self.client.get_collection(self.qdrant_collection)
        current_size = collection_info.vectors_count
        
        # Prepare points for Qdrant
        points = []
        
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            point_id = current_size + i
            
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=embedding.tolist(),
                    payload={
                        "text": chunk["text"],
                        **chunk["metadata"],
                    }
                )
            )
        
        # Upload points in batches
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i+batch_size]
            self.client.upsert(
                collection_name=self.qdrant_collection,
                points=batch,
            )
        
        # Get updated collection size
        collection_info = self.client.get_collection(self.qdrant_collection)
        
        return {
            "vector_db": "qdrant",
            "total_vectors": collection_info.vectors_count,
        }
    
    def index_directory(self, directory_path: Union[str, Path]) -> Dict:
        """Index all processed documents in a directory.
        
        Args:
            directory_path: Path to the directory with processed documents
            
        Returns:
            Indexing results
        """
        directory_path = Path(directory_path)
        
        # Check if the directory exists
        if not directory_path.exists():
            return {
                "directory_path": str(directory_path),
                "success": False,
                "error": "Directory does not exist",
            }
        
        # Find all JSON files
        json_files = list(directory_path.glob("**/*.json"))
        
        if not json_files:
            return {
                "directory_path": str(directory_path),
                "success": False,
                "error": "No JSON files found in directory",
            }
        
        # Index each document
        results = []
        
        for json_file in tqdm(json_files, desc="Indexing documents"):
            result = self.index_document(json_file)
            results.append(result)
        
        # Generate indexing summary
        successful = sum(1 for r in results if r["success"])
        failed = len(results) - successful
        total_chunks = sum(r.get("chunks_indexed", 0) for r in results if r["success"])
        
        summary = {
            "directory_path": str(directory_path),
            "total_files": len(results),
            "successful_files": successful,
            "failed_files": failed,
            "success_rate": round(successful / len(results) * 100, 2) if results else 0,
            "total_chunks_indexed": total_chunks,
        }
        
        return {
            "directory_path": str(directory_path),
            "success": True,
            "summary": summary,
            "results": results,
        }
    
    def save_index(self, output_path: Union[str, Path]) -> Dict:
        """Save the FAISS index and document lookup to disk.
        
        Args:
            output_path: Path to save the index and lookup
            
        Returns:
            Save result
        """
        output_path = Path(output_path)
        
        # Only FAISS indices can be saved
        if self.vector_db != "faiss" or not FAISS_AVAILABLE:
            return {
                "output_path": str(output_path),
                "success": False,
                "error": "Only FAISS indices can be saved",
            }
        
        # Check if the index exists
        if self.index is None:
            return {
                "output_path": str(output_path),
                "success": False,
                "error": "No index exists to save",
            }
        
        try:
            # Create output directory if it doesn't exist
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save FAISS index
            index_path = output_path / "index.faiss"
            faiss.write_index(self.index, str(index_path))
            
            # Save document lookup
            lookup_path = output_path / "lookup.json"
            with open(lookup_path, "w", encoding="utf-8") as f:
                json.dump(self.document_lookup, f, ensure_ascii=False)
            
            return {
                "output_path": str(output_path),
                "index_path": str(index_path),
                "lookup_path": str(lookup_path),
                "total_vectors": self.index.ntotal,
                "success": True,
            }
        except Exception as e:
            return {
                "output_path": str(output_path),
                "success": False,
                "error": f"Failed to save index: {str(e)}",
            }
    
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