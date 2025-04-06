"""Complete RAG pipeline integration."""

from typing import Dict, List, Optional, Set, Tuple, Union

from raggiro.rag.indexer import VectorIndexer
from raggiro.rag.retriever import VectorRetriever
from raggiro.rag.rewriter import QueryRewriter
from raggiro.rag.generator import ResponseGenerator

class RagPipeline:
    """Complete RAG pipeline that integrates all components."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the RAG pipeline.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Initialize components
        self.indexer = VectorIndexer(self.config)
        self.retriever = VectorRetriever(self.config)
        self.rewriter = QueryRewriter(self.config)
        self.generator = ResponseGenerator(self.config)
        
        # Configure pipeline settings
        pipeline_config = self.config.get("pipeline", {})
        self.use_query_rewriting = pipeline_config.get("use_query_rewriting", True)
        self.top_k = pipeline_config.get("top_k", 5)
    
    def query(self, query: str, top_k: Optional[int] = None, document_language: str = None) -> Dict:
        """Process a query through the entire RAG pipeline.
        
        Args:
            query: User query
            top_k: Number of chunks to retrieve (overrides config)
            document_language: Optional language of the document (for better query rewriting and response generation)
            
        Returns:
            Complete RAG result
        """
        # Use provided top_k if given, otherwise use the one from config
        k = top_k if top_k is not None else self.top_k
        
        # Initialize the result dictionary
        result = {
            "original_query": query,
            "success": False,
            "steps": [],
            "document_language": document_language,
        }
        
        # Step 1: Query rewriting (optional)
        if self.use_query_rewriting:
            rewrite_result = self.rewriter.rewrite(query, document_language)
            result["steps"].append({
                "step": "rewrite",
                "result": rewrite_result,
            })
            
            if rewrite_result["success"] and rewrite_result.get("modified", False):
                query = rewrite_result["rewritten_query"]
                result["rewritten_query"] = query
                # If document language wasn't provided but was detected during rewriting
                if not document_language and "query_language" in rewrite_result:
                    document_language = rewrite_result["query_language"]
                    result["document_language"] = document_language
        
        # Step 2: Retrieval
        retrieval_result = self.retriever.retrieve(query, k)
        result["steps"].append({
            "step": "retrieve",
            "result": retrieval_result,
        })
        
        if not retrieval_result["success"]:
            result["error"] = "Failed to retrieve relevant documents"
            return result
        
        # Step 3: Response generation
        chunks = retrieval_result["chunks"]
        
        if not chunks:
            result["steps"].append({
                "step": "generate",
                "result": {
                    "success": False,
                    "error": "No relevant chunks found",
                },
            })
            result["response"] = "I don't have enough information to answer this question."
            result["error"] = "No relevant chunks found"
            return result
        
        generation_result = self.generator.generate(query, chunks)
        result["steps"].append({
            "step": "generate",
            "result": generation_result,
        })
        
        if not generation_result["success"]:
            result["error"] = generation_result.get("error", "Failed to generate a response")
            result["response"] = "I'm sorry, but I encountered an error while generating a response."
            return result
        
        # Build the final result
        result["response"] = generation_result["response"]
        result["success"] = True
        result["chunks_used"] = generation_result["chunks_used"]
        
        return result