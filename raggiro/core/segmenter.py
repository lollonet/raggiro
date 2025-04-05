"""Module for text segmentation into logical chunks."""

import re
from typing import Dict, List, Optional, Set, Tuple, Union

import spacy
from spacy.language import Language

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