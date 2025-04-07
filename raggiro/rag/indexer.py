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
        
        # Summary/dual embedding settings
        self.use_dual_embeddings = indexing_config.get("use_dual_embeddings", True)  # Whether to use both text and summary
        self.summary_weight = indexing_config.get("summary_weight", 0.3)  # Weight for summary vs full text
        
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
                # Create basic chunk
                chunk_data = {
                    "id": chunk.get("id", f"chunk_{i}"),
                    "text": chunk["text"],
                    "metadata": {
                        "document_id": document["metadata"]["file"]["hash"],
                        "document_title": document["metadata"].get("title", ""),
                        "document_path": document["metadata"]["file"]["path"],
                        "chunk_id": chunk.get("id", f"chunk_{i}"),
                        "sections": [seg["text"] for seg in chunk.get("segments", []) if seg["type"] == "header"],
                    }
                }
                
                # Add summary if available
                if "summary" in chunk:
                    chunk_data["summary"] = chunk["summary"]
                    # Also add to metadata so it's stored in vector db payload
                    chunk_data["metadata"]["summary"] = chunk["summary"]
                    
                chunks.append(chunk_data)
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
            # Handle both dictionary and list formats
            if isinstance(document, dict) and "text" in document:
                text = document["text"]
                metadata = document["metadata"]
            elif isinstance(document, list):
                # If document is a list, join all text content
                text = "\n".join([item.get("text", "") for item in document if isinstance(item, dict)])
                # Use metadata from the first item if available
                metadata = next((item.get("metadata", {}) for item in document if isinstance(item, dict) and "metadata" in item), {})
            else:
                # Default values if neither format matches
                text = ""
                metadata = {}
                
            chunks.append({
                "id": "full_text",
                "text": text,
                "metadata": {
                    "document_id": metadata.get("file", {}).get("hash", "unknown"),
                    "document_title": metadata.get("title", ""),
                    "document_path": metadata.get("file", {}).get("path", "unknown"),
                }
            })
        
        return chunks
        
    def _compute_enhanced_embeddings(self, chunks: List[Dict]) -> np.ndarray:
        """Compute embeddings for chunks, optionally enhanced with summaries.
        
        This method creates embeddings that blend the full text with the summary
        if available, giving more weight to the key concepts in the summary.
        
        Args:
            chunks: List of chunks with text and optional summaries
            
        Returns:
            NumPy array of embeddings
        """
        # First, get the regular text embeddings
        texts = [chunk["text"] for chunk in chunks]
        text_embeddings = self.model.encode(texts)
        
        # Check if we should use dual embeddings with summaries
        if not self.use_dual_embeddings:
            return text_embeddings
            
        # Check if any chunks have summaries
        has_summaries = any("summary" in chunk and chunk["summary"] for chunk in chunks)
        if not has_summaries:
            return text_embeddings
            
        # Get summaries, use empty string for chunks without summary
        summaries = [chunk.get("summary", "") if chunk.get("summary", "").strip() else chunk["text"][:200] 
                     for chunk in chunks]
                     
        # Calculate summary embeddings
        summary_embeddings = self.model.encode(summaries)
        
        # Blend the embeddings with configurable weights
        text_weight = 1 - self.summary_weight
        enhanced_embeddings = (text_weight * text_embeddings) + (self.summary_weight * summary_embeddings)
        
        # Normalize the blended embeddings to unit length
        norms = np.linalg.norm(enhanced_embeddings, axis=1, keepdims=True)
        normalized_embeddings = enhanced_embeddings / norms
        
        return normalized_embeddings
    
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
        
        # Compute embeddings with optional summary enhancement
        embeddings = self._compute_enhanced_embeddings(chunks)
        
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
    
    def _index_md_files(self, md_files: List[Path]) -> Dict:
        """Index markdown files directly.
        
        Args:
            md_files: List of markdown file paths
            
        Returns:
            Indexing results
        """
        results = []
        
        for md_file in tqdm(md_files, desc="Indexing markdown files"):
            try:
                # Read the markdown file
                with open(md_file, "r", encoding="utf-8") as f:
                    content = f.read()
                
                # Create a simple document structure
                file_name = md_file.name
                document = {
                    "text": content,
                    "metadata": {
                        "title": file_name,
                        "file": {
                            "path": str(md_file),
                            "hash": f"md_{hash(content) & 0xFFFFFFFF}",  # Simple hash for uniqueness
                            "name": file_name
                        }
                    },
                    # Create a single chunk from the content
                    "chunks": [
                        {
                            "id": f"{file_name}_chunk_1",
                            "text": content,
                            "segments": []  # No segments for simplicity
                        }
                    ]
                }
                
                # Calculate embeddings for the chunk
                chunk_data = {
                    "text": document["chunks"][0]["text"],
                    # Add empty summary as we don't have one for direct markdown files
                    "summary": ""
                }
                embedding = self._compute_enhanced_embeddings([chunk_data])[0]
                
                # Index the document
                chunk = {
                    "id": document["chunks"][0]["id"],
                    "text": text,
                    "metadata": {
                        "document_id": document["metadata"]["file"]["hash"],
                        "document_title": document["metadata"]["title"],
                        "document_path": document["metadata"]["file"]["path"],
                        "chunk_id": document["chunks"][0]["id"],
                        "sections": []
                    }
                }
                
                # Add to index using appropriate method
                if self.vector_db == "faiss" and FAISS_AVAILABLE:
                    # Initialize FAISS index if needed
                    if self.index is None:
                        dimension = len(embedding)
                        self.index = faiss.IndexFlatL2(dimension)
                    
                    # Get current index size
                    current_size = self.index.ntotal
                    
                    # Add embedding to index
                    self.index.add(embedding.reshape(1, -1).astype(np.float32))
                    
                    # Store chunk metadata
                    self.document_lookup[current_size] = {
                        "text": chunk["text"],
                        "metadata": chunk["metadata"]
                    }
                
                results.append({
                    "document_path": str(md_file),
                    "chunks_indexed": 1,
                    "success": True
                })
                
            except Exception as e:
                results.append({
                    "document_path": str(md_file),
                    "success": False,
                    "error": f"Failed to index markdown file: {str(e)}"
                })
        
        # Generate indexing summary
        successful = sum(1 for r in results if r["success"])
        failed = len(results) - successful
        total_chunks = sum(r.get("chunks_indexed", 0) for r in results if r["success"])
        
        summary = {
            "total_files": len(results),
            "successful_files": successful,
            "failed_files": failed,
            "success_rate": round(successful / len(results) * 100, 2) if results else 0,
            "total_chunks_indexed": total_chunks,
        }
        
        return {
            "success": successful > 0,
            "summary": summary,
            "results": results,
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
        
        # Find all JSON files (using different patterns to cover both document files and raw dumps)
        json_files = list(directory_path.glob("**/*.json"))
        
        # If directory itself is a JSON file, add it directly (special case for test scripts)
        if directory_path.suffix.lower() == '.json' and directory_path.is_file():
            json_files = [directory_path]
        
        # Special case: if the directory itself doesn't have JSON files but it has subdirectories
        # with "chunks" in their names, look for JSON files there
        if not json_files:
            chunk_dirs = [d for d in directory_path.glob("*chunks*") if d.is_dir()]
            for chunk_dir in chunk_dirs:
                json_files.extend(list(chunk_dir.glob("**/*.json")))
        
        # If still no JSON files, check if there are MD files that can be used directly
        if not json_files:
            md_files = list(directory_path.glob("**/*.md"))
            if md_files:
                # Create an index with text from MD files
                print(f"No JSON files found, but found {len(md_files)} Markdown files. Creating index from MD content.")
                return self._index_md_files(md_files)
        
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