"""Module for text segmentation into logical chunks."""

import re
import numpy as np
import logging
from typing import Dict, List, Optional, Set, Tuple, Union
from sklearn.metrics.pairwise import cosine_similarity

import spacy
from spacy.language import Language
import nltk

# Set up logger
logger = logging.getLogger("raggiro.segmenter")
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    logger.info("Downloading nltk punkt tokenizer...")
    nltk.download('punkt', quiet=True)

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
        
        # Summary generation settings
        self.generate_summaries = segmentation_config.get("generate_summaries", True)
        self.summary_max_length = segmentation_config.get("summary_max_length", 150)
        self.summary_sentences = segmentation_config.get("summary_sentences", 2)
        
        # Initialize sentence transformer model for semantic chunking
        self.sentence_transformer = None
        if self.semantic_chunking and SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception as e:
                logger.warning(f"Failed to load sentence transformer model: {e}")
                self.semantic_chunking = False
        elif self.semantic_chunking and not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.warning("sentence-transformers package not available. Semantic chunking disabled.")
            self.semantic_chunking = False
        
        # Common section header patterns
        self.section_header_patterns = [
            r"^(?:\d+\.)?\s*[A-Z][A-Z\s]+$",  # ALL CAPS HEADERS
            r"^(?:\d+\.)*\d+\s+[A-Z][a-zA-Z\s]+$",  # Numbered headers like "1.2.3 Header"
            r"^[A-Z][a-z]+\s+\d+\s*[:.]\s*[A-Z][a-zA-Z\s]+$",  # "Section 1: Header"
            r"^(?:Chapter|Section|Part|Appendix)\s+[IVXLCDM]+\s*[:.]\s*[A-Z][a-zA-Z\s]+$",  # "Chapter IV: Header"
            r"^(?:\d+\.)+\s*[A-Z]",  # Hierarchical numbered headers like "1.2.3. Title"
            r"^[A-Z][A-Za-z\s]{2,20}$",  # Short capitalized titles
            r"^(?:[A-Z]{1,2}|[IVX]+)\.(?:[A-Z]{1,2}|[IVX]+)\.",  # Hierarchical references like "A.I." or "II.B."
            r"^[A-Z][a-z]+ \d{1,2}$",  # Simple section identifiers like "Figure 1" or "Table 2"
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
        
        # Pass extraction method info to segments if available
        if "extraction_method" in document:
            extraction_method = document["extraction_method"]
            for segment in segments:
                segment["extraction_method"] = extraction_method
                
            # Log info about OCR documents to help with debugging
            if extraction_method in ["pdf_ocr", "image_ocr"]:
                logger.info(f"Processing OCR document with {len(segments)} segments")
                
        result["segments"] = segments
        
        # Create chunks from segments
        chunks = self._create_chunks(segments)
        result["chunks"] = chunks
        
        # Log chunking results
        logger.info(f"Document segmentation complete: {len(segments)} segments → {len(chunks)} chunks")
        if len(chunks) <= 2 and len(document["text"]) > 1000:
            logger.warning(f"Document was segmented into only {len(chunks)} chunks despite having {len(document['text'])} characters")
            
        # Log if summaries were generated
        summaries_count = sum(1 for chunk in chunks if "summary" in chunk)
        if summaries_count > 0:
            logger.info(f"Generated {summaries_count} chunk summaries")
        
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
        
        # For OCR documents, be more aggressive with segmentation
        is_likely_ocr = len(text) > 1000 and (len(text.split()) / len(text.split('\n')) > 20)
        
        # If we only got a few large paragraphs or likely OCR text, try additional splitting approaches
        if (len(paragraphs) < 10 and len(text) > 3000) or is_likely_ocr:
            # Try splitting by single newlines followed by indentation or bullet points
            paragraphs = re.split(r"\n(?=\s{2,}|\t|•|\*|\-|[0-9]+\.|\([0-9]+\))", text)
            
            # For OCR documents, also look for potential section breaks
            if is_likely_ocr:
                # Look for sentences that might indicate section breaks (all caps terms, etc.)
                ocr_para_splits = re.split(r'(?<=[.!?])\s+(?=[A-Z]{2,})', text)
                
                # If this gives us more reasonable segments, use it
                if len(ocr_para_splits) > len(paragraphs) * 1.2:
                    paragraphs = ocr_para_splits
            
            # If still insufficient, try splitting by sentences for very large paragraphs
            if len(paragraphs) < 15 and any(len(p) > 800 for p in paragraphs):
                new_paragraphs = []
                for p in paragraphs:
                    if len(p) > 800:
                        # Split large paragraphs into sentence groups
                        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', p)
                        # Group sentences into smaller paragraph units (3-5 sentences per unit)
                        for i in range(0, len(sentences), 4):
                            group = " ".join(sentences[i:i+4])
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
    
    def _generate_chunk_summary(self, chunk_text: str) -> str:
        """Generate a summary for a chunk of text.
        
        This uses a simple extractive summarization approach by selecting
        the most representative sentences from the chunk.
        
        Args:
            chunk_text: The text content of the chunk
            
        Returns:
            A short summary of the chunk content
        """
        if not chunk_text or len(chunk_text) < 100:
            return chunk_text[:self.summary_max_length] if chunk_text else ""
        
        try:
            # Use NLTK to tokenize sentences
            from nltk.tokenize import sent_tokenize
            
            try:
                sentences = sent_tokenize(chunk_text)
            except LookupError:
                # Download punkt data if not available
                logger.info("Downloading NLTK punkt tokenizer...")
                import nltk
                nltk.download('punkt', quiet=True)
                sentences = sent_tokenize(chunk_text)
            
            if not sentences:
                # Fallback if tokenization returns empty list
                sentences = [s.strip() + "." for s in chunk_text.split('.') if s.strip()]
                
            if not sentences:
                return chunk_text[:self.summary_max_length]
                
            if len(sentences) <= self.summary_sentences:
                # If we have fewer sentences than requested, use them all
                return " ".join(sentences)
            
            # For extractive summarization, we'll use a simple approach:
            # 1. First sentence is often important (topic sentence)
            # 2. For the remaining summary, pick sentences with important keywords
            
            summary = [sentences[0]]  # Always include the first sentence
            
            # For the rest, let's find sentences with the highest keyword density
            # This is a simple approach - more sophisticated NLP would be better
            
            # Get keyword frequency (excluding stopwords implicitly by min length)
            word_freq = {}
            for sentence in sentences:
                for word in sentence.lower().split():
                    # Only count words longer than 3 chars (implicit stopword filtering)
                    if len(word) > 3:
                        # Clean punctuation and normalize quotes/apostrophes
                        word = word.strip('.,;:!?()[]{}')
                        # Normalize various quote and apostrophe characters
                        word = word.replace('"', '').replace('"', '').replace('"', '')
                        word = word.replace('\'', '').replace(''', '').replace(''', '')
                        if word:
                            word_freq[word] = word_freq.get(word, 0) + 1
            
            # Score sentences based on keyword frequency
            sentence_scores = {}
            for i, sentence in enumerate(sentences):
                if i == 0:  # Skip the first sentence, already included
                    continue
                    
                score = 0
                for word in sentence.lower().split():
                    # Clean and normalize in the same way as when building word_freq
                    word = word.strip('.,;:!?()[]{}')
                    # Normalize various quote and apostrophe characters
                    word = word.replace('"', '').replace('"', '').replace('"', '')
                    word = word.replace('\'', '').replace(''', '').replace(''', '')
                    if len(word) > 3 and word in word_freq:
                        score += word_freq[word]
                
                # Normalize by sentence length to avoid favoring long sentences
                sentence_scores[i] = score / max(1, len(sentence.split()))
            
            # Get the top scoring sentences
            top_indices = sorted(sentence_scores.keys(), 
                                key=lambda i: sentence_scores[i], 
                                reverse=True)[:self.summary_sentences-1]
            
            # Add the top sentences to the summary (in original order)
            for i in sorted(top_indices):
                summary.append(sentences[i])
            
            # Join the summary sentences and ensure it's not too long
            full_summary = " ".join(summary)
            if len(full_summary) > self.summary_max_length:
                # Truncate to the max length at a word boundary
                truncated = full_summary[:self.summary_max_length]
                last_space = truncated.rfind(' ')
                if last_space > 0:
                    truncated = truncated[:last_space] + "..."
                else:
                    truncated = truncated + "..."
                return truncated
            
            return full_summary
            
        except Exception as e:
            logger.warning(f"Failed to generate summary: {str(e)}", exc_info=True)
            # Fallback to a simple first N chars approach
            return chunk_text[:self.summary_max_length] + "..." if len(chunk_text) > self.summary_max_length else chunk_text
    
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
        
        # Special handling for OCR documents: look for indicators
        is_ocr_document = False
        extraction_method = None
        
        # Check if we have extraction method info in the first segment
        if segments and len(segments) > 0:
            for segment in segments:
                if "extraction_method" in segment:
                    extraction_method = segment.get("extraction_method")
                    if extraction_method in ["pdf_ocr", "image_ocr"]:
                        is_ocr_document = True
                        logger.info(f"Detected OCR document with method: {extraction_method}")
                        break
        
        # Adjust chunk size target based on document type
        effective_max_size = self.max_chunk_size
        if is_ocr_document:
            # For OCR documents, we want to ensure we have enough content in each chunk
            # but avoid having only 1-2 large chunks
            if len(segments) < 5:
                # Very few segments, use a smaller max size to force more chunks
                effective_max_size = min(self.max_chunk_size, 800)
                logger.info(f"OCR document with few segments detected, using reduced max size: {effective_max_size}")
        
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
            if len(current_chunk_text) + segment["length"] > effective_max_size and current_chunk_text:
                # Save current chunk
                chunk_text = current_chunk_text.strip()
                chunk_dict = {
                    "text": chunk_text,
                    "segments": current_chunk_segments,
                    "length": len(chunk_text),
                }
                
                # Generate summary if enabled
                if self.generate_summaries:
                    chunk_dict["summary"] = self._generate_chunk_summary(chunk_text)
                
                chunks.append(chunk_dict)
                
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
            chunk_text = current_chunk_text.strip()
            chunk_dict = {
                "text": chunk_text,
                "segments": current_chunk_segments,
                "length": len(chunk_text),
            }
            
            # Generate summary if enabled
            if self.generate_summaries:
                chunk_dict["summary"] = self._generate_chunk_summary(chunk_text)
            
            chunks.append(chunk_dict)
        
        # Check if we ended up with too few chunks for the document size
        total_text = sum(len(seg["text"]) for seg in segments)
        min_expected_chunks = max(2, total_text // 1000)
        
        # If we have fewer chunks than expected for the document size, subdivide the largest chunks
        if len(chunks) < min_expected_chunks and total_text > 2000:
            logger.info(f"Document produced only {len(chunks)} chunks, expected at least {min_expected_chunks}. Subdividing...")
            
            # Sort chunks by size to identify the largest ones
            chunks_by_size = sorted(chunks, key=lambda c: c["length"], reverse=True)
            
            # Take the largest chunks and split them further
            for i in range(min(3, len(chunks_by_size))):
                large_chunk = chunks_by_size[i]
                
                # Only process truly large chunks
                if large_chunk["length"] < 800:
                    continue
                    
                # Split the chunk text into sentences
                sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', large_chunk["text"])
                
                # If we can meaningfully split this chunk
                if len(sentences) >= 6:
                    # Remove the original chunk from the list
                    chunks.remove(large_chunk)
                    
                    # Split into two roughly equal parts
                    mid_point = len(sentences) // 2
                    
                    # Create two new chunks
                    first_half_text = " ".join(sentences[:mid_point])
                    first_half = {
                        "text": first_half_text,
                        "segments": large_chunk["segments"][:len(large_chunk["segments"])//2],
                        "length": len(first_half_text),
                    }
                    
                    second_half_text = " ".join(sentences[mid_point:])
                    second_half = {
                        "text": second_half_text,
                        "segments": large_chunk["segments"][len(large_chunk["segments"])//2:],
                        "length": len(second_half_text),
                    }
                    
                    # Generate summaries for the new chunks if enabled
                    if self.generate_summaries:
                        first_half["summary"] = self._generate_chunk_summary(first_half_text)
                        second_half["summary"] = self._generate_chunk_summary(second_half_text)
                    
                    # Add the new chunks to the list
                    chunks.append(first_half)
                    chunks.append(second_half)
                    
                    logger.info(f"Split a large chunk of {large_chunk['length']} chars into two of {first_half['length']} and {second_half['length']} chars")
        
        # Add chunk IDs
        for i, chunk in enumerate(chunks):
            chunk["id"] = f"chunk_{i+1}"
        
        logger.info(f"Created {len(chunks)} chunks from {len(segments)} segments")
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
        
        # Check if this is an OCR document that might need special handling
        is_ocr_document = False
        for segment in filtered_segments:
            if segment.get("extraction_method") in ["pdf_ocr", "image_ocr"]:
                is_ocr_document = True
                logger.info("Using adjusted semantic similarity for OCR document")
                break
                
        # Adjust similarity threshold for OCR documents
        similarity_threshold = self.semantic_similarity_threshold
        if is_ocr_document:
            # Use a lower threshold for OCR documents since they might have errors
            similarity_threshold = max(0.45, similarity_threshold - 0.15)
            
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
                        if similarity_matrix[i, j] >= similarity_threshold:
                            chunk_segments.append(other_segment)
                            chunk_text += other_segment["text"] + "\n\n"
                            used_segments.add(j)
                            
                            # For OCR documents, limit the chunk size more strictly to prevent 
                            # getting just one huge chunk
                            if is_ocr_document and len(chunk_text) > self.max_chunk_size * 0.8:
                                break
                
                chunk_text_cleaned = chunk_text.strip()
                chunk_dict = {
                    "text": chunk_text_cleaned,
                    "segments": chunk_segments,
                    "length": len(chunk_text_cleaned),
                }
                
                # Generate summary if enabled
                if self.generate_summaries:
                    chunk_dict["summary"] = self._generate_chunk_summary(chunk_text_cleaned)
                
                chunks.append(chunk_dict)
        
        # Process remaining segments
        for i, segment in enumerate(filtered_segments):
            if i not in used_segments:
                chunk_segments = [segment]
                chunk_text = segment["text"] + "\n\n"
                used_segments.add(i)
                
                # Determine max allowed chunk size
                max_size_multiplier = 1.5  # Allow slightly larger chunks for semantic coherence
                if is_ocr_document:
                    # For OCR, use a smaller multiplier and lower absolute limit
                    max_size_multiplier = 1.2
                    max_allowed_size = min(self.max_chunk_size * max_size_multiplier, 1200)
                else:
                    max_allowed_size = self.max_chunk_size * max_size_multiplier
                
                # Find semantically related segments
                for j, other_segment in enumerate(filtered_segments):
                    if j != i and j not in used_segments:
                        # Check semantic similarity and chunk size limit
                        if (similarity_matrix[i, j] >= similarity_threshold and 
                            len(chunk_text) + other_segment["length"] <= max_allowed_size):
                            chunk_segments.append(other_segment)
                            chunk_text += other_segment["text"] + "\n\n"
                            used_segments.add(j)
                
                # Only create the chunk if it has content
                if chunk_segments:
                    chunk_text_cleaned = chunk_text.strip()
                    chunk_dict = {
                        "text": chunk_text_cleaned,
                        "segments": chunk_segments,
                        "length": len(chunk_text_cleaned),
                    }
                    
                    # Generate summary if enabled
                    if self.generate_summaries:
                        chunk_dict["summary"] = self._generate_chunk_summary(chunk_text_cleaned)
                    
                    chunks.append(chunk_dict)
        
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
        
        # Check if this is an OCR document
        is_ocr_document = False
        for chunk in size_chunks:
            for segment in chunk.get("segments", []):
                if segment.get("extraction_method") in ["pdf_ocr", "image_ocr"]:
                    is_ocr_document = True
                    logger.info("Using adjusted hybrid chunking strategy for OCR document")
                    break
            if is_ocr_document:
                break
        
        # Adjust similarity threshold and size limits for OCR documents
        similarity_threshold = self.semantic_similarity_threshold
        max_size_multiplier = 1.5
        if is_ocr_document:
            # For OCR documents, be more conservative with merging to prevent very large chunks
            similarity_threshold = max(0.5, similarity_threshold - 0.1)
            max_size_multiplier = 1.2
            logger.info(f"Adjusted OCR similarity threshold: {similarity_threshold}, size multiplier: {max_size_multiplier}")
        
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
            
            # Determine max allowed size for merged chunks
            if is_ocr_document:
                max_allowed_size = min(self.max_chunk_size * max_size_multiplier, 1200)
            else:
                max_allowed_size = self.max_chunk_size * max_size_multiplier
            
            # Look for similar chunks to merge
            for j, other_chunk in enumerate(size_chunks):
                if j != i and j not in used_chunks:
                    # Check semantic similarity and combined size
                    if (similarity_matrix[i, j] >= similarity_threshold and
                        merged_chunk["length"] + other_chunk["length"] <= max_allowed_size):
                        
                        # Merge the chunks
                        merged_chunk["text"] += "\n\n" + other_chunk["text"]
                        merged_chunk["segments"].extend(other_chunk["segments"])
                        merged_chunk["length"] = len(merged_chunk["text"])
                        used_chunks.add(j)
                        
                        # For OCR documents, limit the number of merges to prevent very large chunks
                        if is_ocr_document and merged_chunk["length"] > self.max_chunk_size * 0.8:
                            break
            
            # Add semantic flag
            merged_chunk["semantic"] = True
            merged_chunk["hybrid"] = True
            
            # Generate summary if enabled
            if self.generate_summaries and "text" in merged_chunk:
                merged_chunk["summary"] = self._generate_chunk_summary(merged_chunk["text"])
                
            final_chunks.append(merged_chunk)
            
        # Log chunking statistics
        logger.info(f"Hybrid chunking: {len(size_chunks)} size chunks → {len(final_chunks)} final chunks")
        
        # Add chunk IDs
        for i, chunk in enumerate(final_chunks):
            chunk["id"] = f"chunk_{i+1}"
        
        return final_chunks