"""Module for text segmentation into logical chunks."""

import re
import numpy as np
from typing import Dict, List, Optional, Set, Tuple, Union
from sklearn.metrics.pairwise import cosine_similarity

import spacy
from spacy.language import Language

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

class Segmenter:
    """Segments text into logical chunks like paragraphs, sections, titles, etc."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the segmenter.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Configure segmenter settings
        segmentation_config = self.config.get("segmentation", {})
        self.use_spacy = segmentation_config.get("use_spacy", True)
        self.spacy_model = segmentation_config.get("spacy_model", "en_core_web_sm")
        self.min_section_length = segmentation_config.get("min_section_length", 100)
        self.max_chunk_size = segmentation_config.get("max_chunk_size", 1000)
        self.chunk_overlap = segmentation_config.get("chunk_overlap", 200)
        
        # Semantic chunking settings
        self.semantic_chunking = segmentation_config.get("semantic_chunking", False)
        self.chunking_strategy = segmentation_config.get("chunking_strategy", "hybrid")
        self.semantic_similarity_threshold = segmentation_config.get("semantic_similarity_threshold", 0.65)
        
        # Initialize sentence transformer model for semantic chunking
        self.sentence_transformer = None
        if self.semantic_chunking and SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception as e:
                print(f"Warning: Failed to load sentence transformer model: {e}")
                self.semantic_chunking = False
        elif self.semantic_chunking and not SENTENCE_TRANSFORMERS_AVAILABLE:
            print("Warning: sentence-transformers package not available. Semantic chunking disabled.")
            self.semantic_chunking = False
        
        # Common section header patterns
        self.section_header_patterns = [
            r"^(?:\d+\.)?\s*[A-Z][A-Z\s]+$",  # ALL CAPS HEADERS
            r"^(?:\d+\.)*\d+\s+[A-Z][a-zA-Z\s]+$",  # Numbered headers like "1.2.3 Header"
            r"^[A-Z][a-z]+\s+\d+\s*[:.]\s*[A-Z][a-zA-Z\s]+$",  # "Section 1: Header"
            r"^(?:Chapter|Section|Part|Appendix)\s+[IVXLCDM]+\s*[:.]\s*[A-Z][a-zA-Z\s]+$",  # "Chapter IV: Header"
        ]
        
        # Load spaCy model if enabled
        self.nlp = None
        if self.use_spacy:
            try:
                self.nlp = spacy.load(self.spacy_model, disable=["ner", "parser"])
            except Exception as e:
                # Fallback to a smaller model or disable spaCy
                try:
                    self.nlp = spacy.blank("en")
                except:
                    self.use_spacy = False
        
        # Precompile regex patterns
        self.section_header_regex = [re.compile(pattern) for pattern in self.section_header_patterns]
    
    def segment(self, document: Dict) -> Dict:
        """Segment a document into logical parts.
        
        Args:
            document: Document dictionary with text
            
        Returns:
            Document with added segments
        """
        result = document.copy()
        
        # Extract structural segments
        segments = self._extract_segments(document["text"])
        result["segments"] = segments
        
        # Create chunks from segments
        chunks = self._create_chunks(segments)
        result["chunks"] = chunks
        
        return result
    
    def _extract_segments(self, text: str) -> List[Dict]:
        """Extract logical segments from text.
        
        Args:
            text: Text to segment
            
        Returns:
            List of segment dictionaries
        """
        if not text:
            return []
        
        segments = []
        
        # Split by paragraphs with multiple approaches to catch different formatting styles
        # First try standard double newline approach
        paragraphs = re.split(r"\n\s*\n", text)
        
        # If we only got a few large paragraphs, try additional splitting approaches
        if len(paragraphs) < 10 and len(text) > 5000:
            # Try splitting by single newlines followed by indentation or bullet points
            paragraphs = re.split(r"\n(?=\s{2,}|\t|•|\*|\-|[0-9]+\.|\([0-9]+\))", text)
            
            # If still insufficient, try splitting by sentences for very large paragraphs
            if len(paragraphs) < 15 and any(len(p) > 1000 for p in paragraphs):
                new_paragraphs = []
                for p in paragraphs:
                    if len(p) > 1000:
                        # Split large paragraphs into sentence groups
                        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', p)
                        # Group sentences into smaller paragraph units (3-5 sentences per unit)
                        for i in range(0, len(sentences), 3):
                            group = " ".join(sentences[i:i+3])
                            if group.strip():
                                new_paragraphs.append(group)
                    else:
                        new_paragraphs.append(p)
                paragraphs = new_paragraphs
        
        current_section = None
        current_section_content = []
        
        for i, paragraph in enumerate(paragraphs):
            paragraph = paragraph.strip()
            if not paragraph:
                continue
                
            # Check if it's a header
            is_header = any(pattern.match(paragraph) for pattern in self.section_header_regex)
            
            # Add additional header detection using spaCy if available
            if not is_header and self.use_spacy and self.nlp and len(paragraph) < 200:
                # Use spaCy to check if this might be a header
                # Headers usually have certain linguistic properties
                doc = self.nlp(paragraph)
                
                # Headers often have a higher ratio of proper nouns, no verbs
                total_tokens = len(doc)
                if total_tokens > 0:
                    proper_nouns = sum(1 for token in doc if token.pos_ == "PROPN")
                    verbs = sum(1 for token in doc if token.pos_ == "VERB")
                    
                    proper_noun_ratio = proper_nouns / total_tokens
                    has_few_verbs = verbs <= 1
                    
                    if proper_noun_ratio > 0.5 and has_few_verbs and total_tokens < 10:
                        is_header = True
            
            if is_header:
                # Save the previous section if we have one
                if current_section and current_section_content:
                    section_text = "\n".join(current_section_content)
                    segments.append({
                        "type": "section",
                        "title": current_section,
                        "text": section_text,
                        "length": len(section_text),
                    })
                
                # Start a new section
                current_section = paragraph
                current_section_content = []
                
                # Add the header as a segment
                segments.append({
                    "type": "header",
                    "text": paragraph,
                    "level": self._estimate_header_level(paragraph),
                    "length": len(paragraph),
                })
            else:
                # Add to current section content
                current_section_content.append(paragraph)
                
                # Also add as a paragraph segment
                segments.append({
                    "type": "paragraph",
                    "text": paragraph,
                    "section": current_section,
                    "length": len(paragraph),
                })
        
        # Add the last section if we have one
        if current_section and current_section_content:
            section_text = "\n".join(current_section_content)
            segments.append({
                "type": "section",
                "title": current_section,
                "text": section_text,
                "length": len(section_text),
            })
        
        # If no sections were found, create a default one
        if not any(segment["type"] == "section" for segment in segments):
            segments.append({
                "type": "section",
                "title": None,
                "text": text,
                "length": len(text),
            })
        
        return segments
    
    def _estimate_header_level(self, header: str) -> int:
        """Estimate the level of a header based on its format.
        
        Args:
            header: Header text
            
        Returns:
            Estimated header level (1-6)
        """
        # Check for numbered headers
        if re.match(r"^\d+\.\d+\.\d+\.\d+", header):
            return 4
        elif re.match(r"^\d+\.\d+\.\d+", header):
            return 3
        elif re.match(r"^\d+\.\d+", header):
            return 2
        elif re.match(r"^\d+\.?", header):
            return 1
        
        # Check for section/chapter headers
        if re.match(r"^(?:Chapter|CHAPTER)", header):
            return 1
        elif re.match(r"^(?:Section|SECTION)", header):
            return 2
        elif re.match(r"^(?:Subsection|SUBSECTION)", header):
            return 3
        
        # Check for formatting cues
        if header.isupper():
            return 1
        elif header.istitle():
            return 2
        
        # Default
        return 3
    
    def _create_chunks(self, segments: List[Dict]) -> List[Dict]:
        """Create chunks from segments for RAG processing.
        
        Args:
            segments: List of segment dictionaries
            
        Returns:
            List of chunk dictionaries
        """
        # Choose the appropriate chunking strategy
        if self.semantic_chunking and self.chunking_strategy == "semantic":
            return self._create_semantic_chunks(segments)
        elif self.semantic_chunking and self.chunking_strategy == "hybrid":
            return self._create_hybrid_chunks(segments)
        else:
            return self._create_size_based_chunks(segments)
    
    def _create_size_based_chunks(self, segments: List[Dict]) -> List[Dict]:
        """Create chunks based on size limit.
        
        Args:
            segments: List of segment dictionaries
            
        Returns:
            List of chunk dictionaries
        """
        chunks = []
        
        current_chunk_text = ""
        current_chunk_segments = []
        
        for segment in segments:
            # Skip extremely short segments (like single words)
            if segment["length"] < 5:
                continue
                
            # Always keep headers with the following content
            if segment["type"] == "header":
                current_chunk_segments.append(segment)
                current_chunk_text += segment["text"] + "\n\n"
                continue
            
            # For sections and paragraphs, check if adding would exceed max size
            if len(current_chunk_text) + segment["length"] > self.max_chunk_size and current_chunk_text:
                # Save current chunk
                chunks.append({
                    "text": current_chunk_text.strip(),
                    "segments": current_chunk_segments,
                    "length": len(current_chunk_text),
                })
                
                # Start new chunk with overlap
                if self.chunk_overlap > 0 and current_chunk_segments:
                    # Find segments to include for overlap
                    overlap_size = 0
                    overlap_segments = []
                    
                    for seg in reversed(current_chunk_segments):
                        if overlap_size >= self.chunk_overlap:
                            break
                            
                        if seg["type"] == "paragraph":
                            overlap_segments.insert(0, seg)
                            overlap_size += seg["length"]
                    
                    # Initialize new chunk with overlap
                    current_chunk_segments = overlap_segments
                    current_chunk_text = "\n\n".join(seg["text"] for seg in overlap_segments)
                    if current_chunk_text:
                        current_chunk_text += "\n\n"
                else:
                    # Start new chunk without overlap
                    current_chunk_segments = []
                    current_chunk_text = ""
            
            # Add current segment to chunk
            current_chunk_segments.append(segment)
            current_chunk_text += segment["text"] + "\n\n"
        
        # Add the last chunk if not empty
        if current_chunk_text:
            chunks.append({
                "text": current_chunk_text.strip(),
                "segments": current_chunk_segments,
                "length": len(current_chunk_text),
            })
        
        # Add chunk IDs
        for i, chunk in enumerate(chunks):
            chunk["id"] = f"chunk_{i+1}"
        
        return chunks
    
    def _create_semantic_chunks(self, segments: List[Dict]) -> List[Dict]:
        """Create chunks based on semantic similarity between segments.
        
        Args:
            segments: List of segment dictionaries
            
        Returns:
            List of chunk dictionaries
        """
        if not self.sentence_transformer:
            # Fall back to size-based chunking if semantic model is not available
            return self._create_size_based_chunks(segments)
        
        # Filter out very short segments
        filtered_segments = [seg for seg in segments if seg["length"] >= 5]
        if not filtered_segments:
            return []
        
        # Create embeddings for each segment
        segment_texts = [seg["text"] for seg in filtered_segments]
        embeddings = self.sentence_transformer.encode(segment_texts)
        
        # Normalize embeddings
        norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized_embeddings = embeddings / norm
        
        # Compute similarity matrix
        similarity_matrix = cosine_similarity(normalized_embeddings)
        
        # Group segments into chunks based on semantic similarity
        chunks = []
        used_segments = set()
        
        # First process all headers as chunk starters
        for i, segment in enumerate(filtered_segments):
            if segment["type"] == "header" and i not in used_segments:
                chunk_segments = [segment]
                chunk_text = segment["text"] + "\n\n"
                used_segments.add(i)
                
                # Find semantically related segments for this header
                for j, other_segment in enumerate(filtered_segments):
                    if j != i and j not in used_segments:
                        # Check semantic similarity and add if above threshold
                        if similarity_matrix[i, j] >= self.semantic_similarity_threshold:
                            chunk_segments.append(other_segment)
                            chunk_text += other_segment["text"] + "\n\n"
                            used_segments.add(j)
                
                chunks.append({
                    "text": chunk_text.strip(),
                    "segments": chunk_segments,
                    "length": len(chunk_text),
                })
        
        # Process remaining segments
        for i, segment in enumerate(filtered_segments):
            if i not in used_segments:
                chunk_segments = [segment]
                chunk_text = segment["text"] + "\n\n"
                used_segments.add(i)
                
                # Find semantically related segments
                for j, other_segment in enumerate(filtered_segments):
                    if j != i and j not in used_segments:
                        # Check semantic similarity and chunk size limit
                        if (similarity_matrix[i, j] >= self.semantic_similarity_threshold and 
                            len(chunk_text) + other_segment["length"] <= self.max_chunk_size * 1.5):  # Allow slightly larger chunks for semantic coherence
                            chunk_segments.append(other_segment)
                            chunk_text += other_segment["text"] + "\n\n"
                            used_segments.add(j)
                
                # Only create the chunk if it has content
                if chunk_segments:
                    chunks.append({
                        "text": chunk_text.strip(),
                        "segments": chunk_segments,
                        "length": len(chunk_text),
                    })
        
        # Add chunk IDs
        for i, chunk in enumerate(chunks):
            chunk["id"] = f"chunk_{i+1}"
            # Add semantic flag
            chunk["semantic"] = True
        
        return chunks
    
    def _create_hybrid_chunks(self, segments: List[Dict]) -> List[Dict]:
        """Create chunks using both size and semantic approaches.
        
        Args:
            segments: List of segment dictionaries
            
        Returns:
            List of chunk dictionaries
        """
        # First create chunks based on size
        size_chunks = self._create_size_based_chunks(segments)
        
        # If semantic chunking is not possible, return size-based chunks
        if not self.sentence_transformer:
            return size_chunks
        
        # Create embeddings for the size-based chunks
        chunk_texts = [chunk["text"] for chunk in size_chunks]
        embeddings = self.sentence_transformer.encode(chunk_texts)
        
        # Normalize embeddings
        norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized_embeddings = embeddings / norm
        
        # Compute similarity matrix
        similarity_matrix = cosine_similarity(normalized_embeddings)
        
        # Merge similar chunks that are below the size limit
        final_chunks = []
        used_chunks = set()
        
        for i, chunk in enumerate(size_chunks):
            if i in used_chunks:
                continue
                
            merged_chunk = {
                "text": chunk["text"],
                "segments": chunk["segments"].copy(),
                "length": chunk["length"],
            }
            used_chunks.add(i)
            
            # Look for similar chunks to merge
            for j, other_chunk in enumerate(size_chunks):
                if j != i and j not in used_chunks:
                    # Check semantic similarity and combined size
                    if (similarity_matrix[i, j] >= self.semantic_similarity_threshold and
                        merged_chunk["length"] + other_chunk["length"] <= self.max_chunk_size * 1.5):
                        
                        # Merge the chunks
                        merged_chunk["text"] += "\n\n" + other_chunk["text"]
                        merged_chunk["segments"].extend(other_chunk["segments"])
                        merged_chunk["length"] = len(merged_chunk["text"])
                        used_chunks.add(j)
            
            # Add semantic flag
            merged_chunk["semantic"] = True
            final_chunks.append(merged_chunk)
        
        # Add chunk IDs
        for i, chunk in enumerate(final_chunks):
            chunk["id"] = f"chunk_{i+1}"
        
        return final_chunks