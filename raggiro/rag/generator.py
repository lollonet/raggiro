"""Response generation module for RAG integration."""

import os
import subprocess
from typing import Dict, List, Optional, Set, Tuple, Union

class ResponseGenerator:
    """Generates responses based on retrieved chunks using LLMs."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the response generator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Configure generation settings
        generation_config = self.config.get("generation", {})
        self.llm_type = generation_config.get("llm_type", "ollama")  # "ollama", "llamacpp", "openai"
        
        # Provider-specific model names
        self.ollama_model = generation_config.get("ollama_model", "mistral")
        self.llamacpp_model = generation_config.get("llamacpp_model", "mistral")
        self.openai_model = generation_config.get("openai_model", "gpt-3.5-turbo")
        
        # Set model name based on provider type
        if self.llm_type == "ollama":
            self.model_name = self.ollama_model
        elif self.llm_type == "llamacpp":
            self.model_name = self.llamacpp_model
        elif self.llm_type == "openai":
            self.model_name = self.openai_model
        else:
            self.model_name = "mistral"  # Default fallback
            
        self.temperature = generation_config.get("temperature", 0.1)
        self.max_tokens = generation_config.get("max_tokens", 1000)
        # Get ollama URL from config with explicit print for debugging
        llm_config = self.config.get("llm", {})
        self.ollama_base_url = generation_config.get("ollama_base_url") or llm_config.get("ollama_base_url", "http://localhost:11434")
        print(f"Generator using Ollama URL: {self.ollama_base_url}")
        self.llamacpp_path = generation_config.get("llamacpp_path", "")
        
        # API settings (for OpenAI, etc.)
        self.api_key = generation_config.get("api_key", "") or self.config.get("llm", {}).get("api_key", "")
        self.api_url = generation_config.get("api_url", "") or self.config.get("llm", {}).get("api_url", "")
        
        # Response generation prompt
        self.prompt_template = generation_config.get("prompt_template", """
You are a helpful assistant that answers questions based on the provided context. Your task is to:

1. Read and understand the user's question
2. Analyze the provided document chunks for relevant information
3. Pay special attention to the summaries provided for each chunk, which highlight key information
4. Generate a comprehensive, accurate answer based ONLY on the provided chunks
5. If the chunks don't contain enough information to answer the question, state this clearly
6. Include specific citations in your answer referencing the source documents
7. Format your response clearly with proper paragraphs, bullet points, or numbered lists as appropriate
8. IMPORTANT: Your response MUST be in the same language as the document chunks and the user query

User Question: {query}

Document Language: {document_language}

Context Chunks:
{chunks}

{additional_instructions}

Your Answer (include citations to specific documents):
""")
    
    def generate(self, query: str, chunks: List[Dict]) -> Dict:
        """Generate a response based on retrieved chunks.
        
        Args:
            query: User query
            chunks: Retrieved chunks with metadata
            
        Returns:
            Generation result
        """
        if not chunks:
            return {
                "query": query,
                "response": "I don't have enough information to answer this question.",
                "success": False,
                "error": "No context chunks provided",
            }
        
        try:
            # Format context chunks for the prompt
            formatted_chunks = self._format_chunks(chunks)
            
            # Detect document language from chunks
            document_language = self._detect_document_language(chunks)
            
            # Prepare additional instructions based on document language
            additional_instructions = ""
            if document_language:
                language_name = self._get_language_name(document_language)
                additional_instructions = f"IMPORTANT: Your response MUST be in {language_name}."
            
            # Generate response with the appropriate LLM
            if self.llm_type == "ollama":
                result = self._generate_with_ollama(query, formatted_chunks, document_language, additional_instructions)
            elif self.llm_type == "llamacpp":
                result = self._generate_with_llamacpp(query, formatted_chunks, document_language, additional_instructions)
            elif self.llm_type == "openai":
                result = self._generate_with_openai(query, formatted_chunks, document_language, additional_instructions)
            else:
                return {
                    "query": query,
                    "response": "I'm unable to generate a response due to configuration issues.",
                    "success": False,
                    "error": f"Unsupported LLM type: {self.llm_type}",
                }
            
            if "error" in result:
                return {
                    "query": query,
                    "response": "I'm sorry, but I encountered an error while generating a response.",
                    "success": False,
                    "error": result["error"],
                }
            
            return {
                "query": query,
                "response": result["response"],
                "success": True,
                "chunks_used": len(chunks),
            }
        except Exception as e:
            return {
                "query": query,
                "response": "I'm sorry, but I encountered an error while generating a response.",
                "success": False,
                "error": str(e),
            }
    
    def _format_chunks(self, chunks: List[Dict]) -> str:
        """Format chunks for inclusion in the prompt.
        
        Args:
            chunks: Retrieved chunks with metadata
            
        Returns:
            Formatted chunks as a string
        """
        formatted_chunks = []
        
        for i, chunk in enumerate(chunks):
            # Format document source information
            doc_title = chunk["metadata"].get("document_title", "")
            doc_path = chunk["metadata"].get("document_path", "").split("/")[-1]
            section = ""
            
            if "sections" in chunk["metadata"] and chunk["metadata"]["sections"]:
                section = f" - Section: {chunk['metadata']['sections'][0]}"
            elif "section_title" in chunk["metadata"] and chunk["metadata"]["section_title"]:
                section = f" - Section: {chunk['metadata']['section_title']}"
            
            source = f"Document {i+1}: {doc_title or doc_path}{section}"
            
            # Format the chunk text
            chunk_text = chunk["text"].strip()
            
            # Include summary if available
            summary = ""
            if "summary" in chunk and chunk["summary"]:
                summary = f"Summary: {chunk['summary'].strip()}\n\n"
            
            # Add to formatted chunks
            formatted_chunks.append(f"--- {source} ---\n{summary}Content: {chunk_text}\n")
        
        return "\n".join(formatted_chunks)
    
    def _detect_document_language(self, chunks: List[Dict]) -> str:
        """Detect the language of the document from chunks.
        
        Args:
            chunks: Document chunks
            
        Returns:
            Language code or None if detection fails
        """
        # First check if language is specified in metadata
        for chunk in chunks:
            if "metadata" in chunk and chunk["metadata"].get("language"):
                return chunk["metadata"].get("language")
        
        # If not found in metadata, try to detect from text
        all_text = " ".join([chunk.get("text", "") for chunk in chunks])
        return self._detect_language(all_text)
    
    def _detect_language(self, text: str) -> str:
        """Detect the language of a text.
        
        Args:
            text: Text to detect language
            
        Returns:
            Language code or None if detection fails
        """
        try:
            import langdetect
            return langdetect.detect(text)
        except:
            # If langdetect is not available or fails, return None
            return None
            
    def _get_language_name(self, language_code: str) -> str:
        """Get language name from code.
        
        Args:
            language_code: ISO language code
            
        Returns:
            Full language name
        """
        language_map = {
            "en": "English",
            "it": "Italian",
            "es": "Spanish",
            "fr": "French",
            "de": "German",
            "pt": "Portuguese",
            "ru": "Russian",
            "zh": "Chinese",
            "ja": "Japanese"
        }
        
        return language_map.get(language_code, language_code)
    
    def _generate_with_ollama(self, query: str, formatted_chunks: str, document_language: str = None, additional_instructions: str = "") -> Dict:
        """Generate a response using Ollama.
        
        Args:
            query: User query
            formatted_chunks: Formatted context chunks
            document_language: Detected language of the document
            additional_instructions: Additional instructions for the LLM
            
        Returns:
            Generation result
        """
        try:
            import requests
            
            # Format the prompt
            prompt = self.prompt_template.format(
                query=query,
                chunks=formatted_chunks,
                document_language=document_language or "Not detected",
                additional_instructions=additional_instructions,
            )
            
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
                return {"error": f"Ollama API error: {response.text}"}
            
            # Extract the generated text
            generated_text = response.json().get("response", "").strip()
            
            return {"response": generated_text}
        except ImportError:
            return {"error": "requests module not installed"}
        except Exception as e:
            return {"error": str(e)}
    
    def _generate_with_llamacpp(self, query: str, formatted_chunks: str, document_language: str = None, additional_instructions: str = "") -> Dict:
        """Generate a response using llama.cpp.
        
        Args:
            query: User query
            formatted_chunks: Formatted context chunks
            document_language: Detected language of the document
            additional_instructions: Additional instructions for the LLM
            
        Returns:
            Generation result
        """
        # Format the prompt
        prompt = self.prompt_template.format(
            query=query,
            chunks=formatted_chunks,
            document_language=document_language or "Not detected",
            additional_instructions=additional_instructions,
        )
        
        try:
            # Check if llamacpp_path is set
            if not self.llamacpp_path or not os.path.exists(self.llamacpp_path):
                return {"error": "llama.cpp executable path not set or does not exist"}
            
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
            response_start = output.find("Your Answer") + len("Your Answer")
            
            if response_start == -1 + len("Your Answer"):
                return {"error": "Failed to extract response from llama.cpp output"}
            
            generated_text = output[response_start:].strip()
            
            return {"response": generated_text}
        except subprocess.CalledProcessError as e:
            return {"error": f"llama.cpp error: {e.stderr}"}
        except Exception as e:
            return {"error": str(e)}
            
    def _generate_with_openai(self, query: str, formatted_chunks: str, document_language: str = None, additional_instructions: str = "") -> Dict:
        """Generate a response using OpenAI API.
        
        Args:
            query: User query
            formatted_chunks: Formatted context chunks
            document_language: Detected language of the document
            additional_instructions: Additional instructions for the LLM
            
        Returns:
            Generation result
        """
        try:
            import openai
            
            # Check if API key is set
            if not self.api_key:
                return {"error": "OpenAI API key not set"}
            
            # Configure OpenAI client
            client = openai.OpenAI(api_key=self.api_key)
            
            # Set custom API URL if provided
            if self.api_url:
                client.base_url = self.api_url
            
            # Format the prompt
            prompt = self.prompt_template.format(
                query=query,
                chunks=formatted_chunks,
                document_language=document_language or "Not detected",
                additional_instructions=additional_instructions,
            )
            
            # Create a system message with language instructions
            system_content = "You are a helpful assistant that answers questions based on provided context. Pay special attention to the summaries provided for each chunk, which highlight key information."
            if document_language:
                language_name = self._get_language_name(document_language)
                system_content += f" IMPORTANT: Your response MUST be in {language_name}."
            
            # Call OpenAI API
            response = client.chat.completions.create(
                model=self.openai_model,  # Use openai_model for API calls
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            
            # Extract the generated text
            generated_text = response.choices[0].message.content.strip()
            
            # If the response is empty, return error
            if not generated_text:
                return {"error": "Empty response from OpenAI API"}
            
            # Check if the response contains the expected format marker and clean if needed
            if "Your Answer" in generated_text:
                parts = generated_text.split("Your Answer", 1)
                generated_text = parts[1].strip().lstrip(":")
            
            return {"response": generated_text}
        except ImportError:
            return {"error": "OpenAI package not installed. Install with: pip install openai"}
        except Exception as e:
            return {"error": f"OpenAI API error: {str(e)}"}