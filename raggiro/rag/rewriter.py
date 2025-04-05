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
        self.llm_type = rewriting_config.get("llm_type", "ollama")  # "ollama", "llamacpp", "openai"
        
        # Provider-specific model names
        self.ollama_model = rewriting_config.get("ollama_model", "llama3")
        self.llamacpp_model = rewriting_config.get("llamacpp_model", "llama3")
        self.openai_model = rewriting_config.get("openai_model", "gpt-3.5-turbo")
        
        # Set model name based on provider type
        if self.llm_type == "ollama":
            self.model_name = self.ollama_model
        elif self.llm_type == "llamacpp":
            self.model_name = self.llamacpp_model
        elif self.llm_type == "openai":
            self.model_name = self.openai_model
        else:
            self.model_name = "llama3"  # Default fallback
            
        self.temperature = rewriting_config.get("temperature", 0.1)
        self.max_tokens = rewriting_config.get("max_tokens", 200)
        self.ollama_base_url = rewriting_config.get("ollama_base_url", self.config.get("llm", {}).get("ollama_base_url", "http://localhost:11434"))
        self.llamacpp_path = rewriting_config.get("llamacpp_path", "")
        
        # API settings (for OpenAI, etc.)
        self.api_key = rewriting_config.get("api_key", "") or self.config.get("llm", {}).get("api_key", "")
        self.api_url = rewriting_config.get("api_url", "") or self.config.get("llm", {}).get("api_url", "")
        
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
            elif self.llm_type == "openai":
                result = self._rewrite_with_openai(query)
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
    
    def _rewrite_with_openai(self, query: str) -> Dict:
        """Rewrite a query using OpenAI API.
        
        Args:
            query: Original query string
            
        Returns:
            Rewriting result
        """
        try:
            import openai
            
            # Check if API key is set
            if not self.api_key:
                return {
                    "rewritten_query": query,
                    "error": "OpenAI API key not set",
                }
            
            # Configure OpenAI client
            client = openai.OpenAI(api_key=self.api_key)
            
            # Set custom API URL if provided
            if self.api_url:
                client.base_url = self.api_url
            
            # Format the prompt
            prompt = self.prompt_template.format(query=query)
            
            # Call OpenAI API
            response = client.chat.completions.create(
                model=self.openai_model,  # Use openai_model for API calls
                messages=[
                    {"role": "system", "content": "You rewrite search queries to make them more effective for retrieval."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            
            # Extract the generated text
            generated_text = response.choices[0].message.content.strip()
            
            # If the response is empty, return the original query
            if not generated_text:
                return {"rewritten_query": query}
            
            # Check if the response contains the expected format
            if "Rewritten Query:" in generated_text:
                lines = generated_text.split("Rewritten Query:", 1)
                generated_text = lines[1].strip()
            
            return {"rewritten_query": generated_text}
        except ImportError:
            return {
                "rewritten_query": query,
                "error": "OpenAI package not installed. Install with: pip install openai",
            }
        except Exception as e:
            return {
                "rewritten_query": query,
                "error": f"OpenAI API error: {str(e)}",
            }