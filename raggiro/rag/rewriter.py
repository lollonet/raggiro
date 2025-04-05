"""Query rewriting module for RAG integration."""

import os
import subprocess
from typing import Dict, List, Optional, Set, Tuple, Union

class QueryRewriter:
    """Rewrites user queries for better retrieval performance using LLMs."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the query rewriter.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Configure rewriting settings
        rewriting_config = self.config.get("rewriting", {})
        self.enabled = rewriting_config.get("enabled", True)
        self.llm_type = rewriting_config.get("llm_type", "ollama")  # "ollama", "llamacpp"
        self.model_name = rewriting_config.get("model_name", "llama3")
        self.temperature = rewriting_config.get("temperature", 0.1)
        self.max_tokens = rewriting_config.get("max_tokens", 200)
        self.ollama_base_url = rewriting_config.get("ollama_base_url", "http://localhost:11434")
        self.llamacpp_path = rewriting_config.get("llamacpp_path", "")
        
        # Rewriting prompt
        self.prompt_template = rewriting_config.get("prompt_template", """
You are a helpful assistant that improves user queries for a retrieval system. Your task is to:

1. Understand the original query
2. Make it more specific, detailed, and precise
3. Expand ambiguous terms while preserving the core meaning
4. Add any missing but implied context that would help retrieval
5. Keep the rewritten query concise, focused, and in the form of a question

Original Query: {query}

Rewritten Query: 
""")
    
    def rewrite(self, query: str) -> Dict:
        """Rewrite a query for better retrieval performance.
        
        Args:
            query: Original query string
            
        Returns:
            Rewriting result
        """
        if not self.enabled:
            return {
                "original_query": query,
                "rewritten_query": query,
                "success": True,
                "modified": False,
            }
        
        try:
            if self.llm_type == "ollama":
                result = self._rewrite_with_ollama(query)
            elif self.llm_type == "llamacpp":
                result = self._rewrite_with_llamacpp(query)
            else:
                return {
                    "original_query": query,
                    "rewritten_query": query,
                    "success": False,
                    "error": f"Unsupported LLM type: {self.llm_type}",
                    "modified": False,
                }
            
            return {
                "original_query": query,
                "rewritten_query": result["rewritten_query"],
                "success": True,
                "modified": query != result["rewritten_query"],
            }
        except Exception as e:
            return {
                "original_query": query,
                "rewritten_query": query,
                "success": False,
                "error": f"Failed to rewrite query: {str(e)}",
                "modified": False,
            }
    
    def _rewrite_with_ollama(self, query: str) -> Dict:
        """Rewrite a query using Ollama.
        
        Args:
            query: Original query string
            
        Returns:
            Rewriting result
        """
        try:
            import requests
            
            # Format the prompt
            prompt = self.prompt_template.format(query=query)
            
            # Call Ollama API
            response = requests.post(
                f"{self.ollama_base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                    "stream": False,
                },
            )
            
            if response.status_code != 200:
                return {
                    "rewritten_query": query,
                    "error": f"Ollama API error: {response.text}",
                }
            
            # Extract the generated text
            generated_text = response.json().get("response", "").strip()
            
            # If the response is empty, return the original query
            if not generated_text:
                return {"rewritten_query": query}
            
            return {"rewritten_query": generated_text}
        except ImportError:
            return {
                "rewritten_query": query,
                "error": "requests module not installed",
            }
        except Exception as e:
            return {
                "rewritten_query": query,
                "error": str(e),
            }
    
    def _rewrite_with_llamacpp(self, query: str) -> Dict:
        """Rewrite a query using llama.cpp.
        
        Args:
            query: Original query string
            
        Returns:
            Rewriting result
        """
        # Format the prompt
        prompt = self.prompt_template.format(query=query)
        
        try:
            # Check if llamacpp_path is set
            if not self.llamacpp_path or not os.path.exists(self.llamacpp_path):
                return {
                    "rewritten_query": query,
                    "error": "llama.cpp executable path not set or does not exist",
                }
            
            # Call llama.cpp
            result = subprocess.run(
                [
                    self.llamacpp_path,
                    "--model", self.model_name,
                    "--temp", str(self.temperature),
                    "--n-predict", str(self.max_tokens),
                    "--prompt", prompt,
                    "--no-mmap",
                    "--quiet"
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            
            # Extract the generated text (after the prompt)
            output = result.stdout.strip()
            
            # Find where the prompt ends and the response begins
            response_start = output.find("Rewritten Query:") + len("Rewritten Query:")
            
            if response_start == -1 + len("Rewritten Query:"):
                return {"rewritten_query": query}
            
            generated_text = output[response_start:].strip()
            
            # If the response is empty, return the original query
            if not generated_text:
                return {"rewritten_query": query}
            
            return {"rewritten_query": generated_text}
        except subprocess.CalledProcessError as e:
            return {
                "rewritten_query": query,
                "error": f"llama.cpp error: {e.stderr}",
            }
        except Exception as e:
            return {
                "rewritten_query": query,
                "error": str(e),
            }