"""Module for text segmentation into logical chunks."""

import re
import numpy as np
import logging
import unicodedata
from typing import Dict, List, Optional, Set, Tuple, Union
from sklearn.metrics.pairwise import cosine_similarity

import spacy
from spacy.language import Language
import nltk

# Import Unicode normalizer if available
try:
    from ..utils.unicode_normalizer import UnicodeNormalizer
    UNICODE_NORMALIZER_AVAILABLE = True
except ImportError:
    UNICODE_NORMALIZER_AVAILABLE = False

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
                # Fix for 'init_empty_weights' error: Import specific modules before loading the model
                try:
                    # Make sure to import torch and transformers directly
                    import torch
                    import transformers
                    
                    # Import specific modules that might be needed for initialization
                    from transformers import AutoModel, AutoTokenizer
                    from transformers.modeling_utils import PreTrainedModel
                    
                    # Explicitly import the initialization functions to fix the 'init_empty_weights' error
                    try:
                        from transformers.modeling_utils import init_empty_weights
                    except ImportError:
                        # In older transformers versions, this might be in a different location
                        try:
                            from transformers.utils import init_empty_weights
                        except ImportError:
                            logger.warning("Could not import init_empty_weights from transformers")
                            
                    # Import other potentially needed functions
                    try:
                        from transformers import __version__ as transformers_version
                        logger.info(f"Using transformers version: {transformers_version}")
                    except:
                        pass
                except ImportError as import_err:
                    logger.warning(f"Could not import required dependencies for SentenceTransformer: {str(import_err)}")
                
                # Explicitly set the device to CPU to avoid CUDA related issues
                self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
                logger.info(f"Successfully loaded sentence transformer model: all-MiniLM-L6-v2")
            except Exception as e:
                logger.warning(f"Failed to load sentence transformer model: {e}")
                
                # Try loading a backup model with more minimal settings
                try:
                    backup_model = "paraphrase-MiniLM-L6-v2"
                    logger.info(f"Trying backup model: {backup_model}")
                    
                    # Pre-import the required modules
                    import torch
                    import transformers
                    
                    # Try importing the init_empty_weights function from different possible locations
                    try:
                        from transformers.modeling_utils import init_empty_weights
                    except ImportError:
                        try:
                            from transformers.utils import init_empty_weights
                        except ImportError:
                            logger.warning("Could not import init_empty_weights from transformers")
                    
                    # Try an alternative way to initialize the model
                    try:
                        from sentence_transformers import __version__ as sbert_version
                        logger.info(f"Using sentence-transformers version: {sbert_version}")
                    except:
                        pass
                    
                    # Explicit device setting to use CPU
                    self.sentence_transformer = SentenceTransformer(backup_model, device='cpu')
                    logger.info(f"Successfully loaded backup model: {backup_model}")
                except Exception as e2:
                    logger.error(f"Failed to load backup model: {e2}")
                    
                    # Try one last fallback approach with absolute minimal requirements
                    try:
                        logger.info("Attempting final fallback to basic model...")
                        
                        # Set environment variable to disable parallelism
                        import os
                        os.environ["TOKENIZERS_PARALLELISM"] = "false"
                        
                        # Import model directly with minimal settings
                        from sentence_transformers import SentenceTransformer
                        
                        # Use a very basic model
                        last_resort_model = "distiluse-base-multilingual-cased"
                        self.sentence_transformer = SentenceTransformer(last_resort_model, device='cpu')
                        logger.info(f"Successfully loaded fallback model: {last_resort_model}")
                    except Exception as e3:
                        logger.error(f"All sentence transformer loading attempts failed: {e3}")
                        self.semantic_chunking = False
        elif self.semantic_chunking and not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.warning("sentence-transformers package not available. Semantic chunking disabled.")
            self.semantic_chunking = False
        
        # Common section header patterns with support for multiple European languages
        self.section_header_patterns = [
            # Format-based patterns (work in any language)
            r"^(?:\d+\.)?\s*[A-Z][A-Z\s]+$",  # ALL CAPS HEADERS
            r"^(?:\d+\.)*\d+\s+[A-Z][a-zA-Z\s]+$",  # Numbered headers like "1.2.3 Header"
            r"^(?:\d+[\.\-])+\s*[A-Z]",  # Hierarchical numbered headers like "1.2.3. Title" or "1-2-3 Title"
            r"^[A-Z][A-Za-z\s]{2,20}$",  # Short capitalized titles
            r"^(?:[A-Z]{1,2}|[IVX]+)\.(?:[A-Z]{1,2}|[IVX]+)\.",  # Hierarchical references like "A.I." or "II.B."
            
            # English patterns
            r"^[A-Z][a-z]+\s+\d+\s*[:.]\s*[A-Z][a-zA-Z\s]+$",  # "Section 1: Header"
            r"^(?:Chapter|Section|Part|Appendix)\s+[IVXLCDM]+\s*[:.]\s*[A-Z][a-zA-Z\s]+$",  # "Chapter IV: Header"
            r"^[A-Z][a-z]+ \d{1,2}$",  # Simple section identifiers like "Figure 1" or "Table 2"
            
            # Italian patterns
            r"^(?:Capitolo|Sezione|Parte|Appendice)\s+[IVXLCDM\d]+\s*[:.]\s*.+$",  # "Capitolo IV: Titolo"
            r"^(?:Figura|Tabella) \d{1,2}$",  # "Figura 1" or "Tabella 2"
            
            # Spanish patterns
            r"^(?:Capítulo|Sección|Parte|Apéndice)\s+[IVXLCDM\d]+\s*[:.]\s*.+$",  # "Capítulo IV: Título"
            r"^(?:Figura|Tabla) \d{1,2}$",  # "Figura 1" or "Tabla 2"
            
            # French patterns
            r"^(?:Chapitre|Section|Partie|Annexe)\s+[IVXLCDM\d]+\s*[:.]\s*.+$",  # "Chapitre IV: Titre"
            r"^(?:Figure|Tableau) \d{1,2}$",  # "Figure 1" or "Tableau 2"
            
            # German patterns
            r"^(?:Kapitel|Abschnitt|Teil|Anhang)\s+[IVXLCDM\d]+\s*[:.]\s*.+$",  # "Kapitel IV: Titel"
            r"^(?:Abbildung|Tabelle) \d{1,2}$",  # "Abbildung 1" or "Tabelle 2"
            
            # Portuguese patterns
            r"^(?:Capítulo|Seção|Parte|Apêndice)\s+[IVXLCDM\d]+\s*[:.]\s*.+$",  # "Capítulo IV: Título"
            r"^(?:Figura|Tabela) \d{1,2}$",  # "Figura 1" or "Tabela 2"
            
            # Generic patterns for other European languages
            r"^(?:Hoofdstuk|Rozdział|Kapittel|Κεφάλαιο|Luku|Capitol|Fejezet|Kapitola)\s+[\dIVX]+\s*[:.]\s*.+$",
            r"^(?:Afbeelding|Rysunek|Figur|Εικόνα|Kuva|Imagine|Ábra|Obrázek)\s+\d+$",
            
            # Special patterns for handling common nested section indicators in various languages
            r"^[\d\.\-]+\s+[A-ZÀÁÈÉÌÍÒÓÙÚÄËÏÖÜŸÂÊÎÔÛÇÑ]",  # Hierarchical numbered headers with accented capitals
            r"^[A-ZÀÁÈÉÌÍÒÓÙÚÄËÏÖÜŸÂÊÎÔÛÇÑ][a-zàáèéìíòóùúäëïöüÿâêîôûçñ\s]{2,25}$"  # Short capitalized titles with accents
        ]
        
        # Patterns to identify table of contents or indexes in multiple European languages
        self.toc_patterns = [
            # English
            r"^(?:Table\s+of\s+Contents|Contents|Index)$",
            
            # Italian
            r"^(?:Indice|Sommario|Contenuti|Indice\s+generale|Indice\s+analitico|Indice\s+degli\s+argomenti)$",
            
            # Spanish
            r"^(?:Índice|Índice\s+general|Tabla\s+de\s+contenidos|Contenidos|Sumario)$",
            
            # French
            r"^(?:Table\s+des\s+matières|Sommaire|Index|Table\s+des\s+contenus)$",
            
            # German
            r"^(?:Inhaltsverzeichnis|Inhalt|Index|Stichwortverzeichnis)$",
            
            # Portuguese
            r"^(?:Índice|Sumário|Conteúdo|Tabela\s+de\s+conteúdos)$",
            
            # Dutch
            r"^(?:Inhoudsopgave|Inhoud|Index|Trefwoordenregister)$",
            
            # Polish
            r"^(?:Spis\s+treści|Indeks|Zawartość)$",
            
            # Swedish
            r"^(?:Innehållsförteckning|Innehåll|Index|Sakregister)$",
            
            # Danish
            r"^(?:Indholdsfortegnelse|Indhold|Indeks|Stikordsregister)$",
            
            # Greek
            r"^(?:Πίνακας\s+περιεχομένων|Περιεχόμενα|Ευρετήριο)$",
            
            # Finnish
            r"^(?:Sisällysluettelo|Sisältö|Hakemisto)$",
            
            # Romanian
            r"^(?:Cuprins|Tabla\s+de\s+materii|Indice|Conținut)$",
            
            # Hungarian
            r"^(?:Tartalomjegyzék|Tartalom|Index|Tárgymutató)$",
            
            # Czech
            r"^(?:Obsah|Rejstřík|Obsah\s+knihy)$",
            
            # All caps versions (any language)
            r"^(?:INDICE|SOMMARIO|ÍNDICE|TABLE\s+OF\s+CONTENTS|CONTENTS|INDEX|INHALTSVERZEICHNIS|TABLA\s+DE\s+CONTENIDOS|TABLE\s+DES\s+MATIÈRES|SUMÁRIO|SPIS\s+TREŚCI|OBSAH|CUPRINS|INHOUD|ΠΕΡΙΕΧΟΜΕΝΑ)$",
        ]
        
        # Patterns to identify entries in a table of contents with support for multiple languages and formats
        self.toc_entry_patterns = [
            # Basic formats with decimal numbering
            r"^\s*(?:\d+\.)*\d+\s+.+\s+\d+\s*$",  # Numbered section with page number at end (e.g., "1.2 Section Title 42")
            r"^\s*(?:\d+[\.\-])*\d+[\.\-]\s+.+\s+\d+\s*$",  # Supports variations like "1-2-3 Title 42" as well as "1.2.3 Title 42"
            
            # Formats with leader characters
            r"^.+\s+\.{2,}\s+\d+\s*$",  # Entry with dots leading to page number (e.g., "Section Title .... 42")
            r"^.+\s+[\.\-_]{2,}\s+\d+\s*$",  # Entry with dots, dashes or underscores leading to page number
            
            # Simple formats
            r"^.+\s+\d+\s*$",  # Simple entry with page number at end (e.g., "Section Title 42")
            r"^.+\s+[Ss]eite\s+\d+\s*$",  # German style with "Seite" (page) (e.g., "Kapitel Seite 42")
            r"^.+\s+[Pp]age\s+\d+\s*$",  # With explicit "page" word (e.g., "Chapter page 42")
            r"^.+\s+[Pp]ágina\s+\d+\s*$",  # Spanish "página" (e.g., "Sección página 42")
            r"^.+\s+[Pp]agina\s+\d+\s*$",  # Italian "pagina" (e.g., "Sezione pagina 42")
            r"^.+\s+[Pp]\.?\s+\d+\s*$",  # Abbreviated page indicator (e.g., "Chapter p. 42")
            
            # Formats with Roman numerals and letters for sections
            r"^\s*(?:[IVX]+|[A-Z])\.\s+.+\s+\d+\s*$",  # Roman numeral or lettered sections with page number (e.g., "IV. Section Title 42")
            r"^\s*(?:[IVX]+|[A-Z])[\.\-]\s+.+\s+\d+\s*$",  # With dash or dot (e.g., "IV- Section Title 42")
            r"^\s*[A-Z](?:[IVX]+|[0-9]+)\.\s+.+\s+\d+\s*$",  # Format like "A1. Title 42" or "AIV. Title 42"
            
            # Formats with brackets or parentheses
            r"^\s*[\[\(](?:\d+|[A-Z]|[IVX]+)[\]\)]\s+.+\s+\d+\s*$",  # Bracketed numbers or letters (e.g., "[1] Title 42" or "(A) Title 42")
            
            # Formats with chapter/section indicators
            r"^\s*(?:Chapter|Chap\.|Ch\.|Section|Sect\.|Sec\.|Kapitel|Kap\.|Capítulo|Cap\.|Chapitre|Capitolo|Hoofdstuk)\s+[\dIVX]+\s*[:\.\-—–]\s*.+\s+\d+\s*$",
            
            # Indented formats (common in many TOCs)
            r"^\s{2,}.+\s+\d+\s*$",  # Indented entry with page number (e.g., "    Subsection 42")
            
            # Multilevel entries with indents or repeated symbols 
            r"^(?:\s{2,}|\t+|[»•◦▪‣◘○●·]+\s+).+\s+\d+\s*$",  # Indented or bulleted subentries
            
            # Formats with special diacritics and characters for European languages
            r"^.+\s+(?:страница|стр\.)\s+\d+\s*$",  # Cyrillic "страница" (page) (Russian, Bulgarian)
            r"^.+\s+(?:strona|str\.)\s+\d+\s*$",  # Polish "strona" (page)
            r"^.+\s+(?:σελίδα|σελ\.)\s+\d+\s*$",  # Greek "σελίδα" (page)
            r"^.+\s+oldal\s+\d+\s*$",  # Hungarian "oldal" (page)
            r"^.+\s+(?:sivu|s\.)\s+\d+\s*$",  # Finnish "sivu" (page)
            r"^.+\s+(?:sida|s\.)\s+\d+\s*$",  # Swedish "sida" (page)
            r"^.+\s+(?:strana|str\.)\s+\d+\s*$",  # Czech "strana" (page)
            r"^.+\s+(?:pagină|pag\.)\s+\d+\s*$",  # Romanian "pagină" (page)
            
            # Catch-all pattern for any reasonable TOC entry format
            r"^[^\.]+(?:\.\d+)*(?:[\.\s]+|\s+)[^\d]+\s+\d+\s*$"  # Generic pattern to catch other formats
        ]
        
        # Load spaCy model if enabled
        self.nlp = None
        if self.use_spacy:
            try:
                self.nlp = spacy.load(self.spacy_model, disable=["ner", "parser"])
                logger.info(f"Caricato modello spaCy: {self.spacy_model}")
            except OSError as e:
                logger.warning(f"Modello spaCy {self.spacy_model} non trovato. Errore: {e}")
                # Prova a caricare altri modelli in ordine di preferenza
                alternative_models = ["xx_sent_ud_sm", "it_core_news_sm", "en_core_web_sm"]
                for model in alternative_models:
                    if model != self.spacy_model:  # Non provare di nuovo lo stesso modello
                        try:
                            self.nlp = spacy.load(model, disable=["ner", "parser"])
                            logger.info(f"Caricato modello spaCy alternativo: {model}")
                            break
                        except OSError:
                            continue
                
                # Se ancora nessun modello è stato caricato, prova con blank
                if self.nlp is None:
                    try:
                        logger.warning("Utilizzo modello spaCy blank come fallback")
                        self.nlp = spacy.blank("en")
                    except Exception as e:
                        logger.error(f"Impossibile caricare modello spaCy blank: {e}")
                        self.use_spacy = False
            except Exception as e:
                logger.error(f"Errore non previsto nel caricamento di spaCy: {e}")
                self.use_spacy = False
        
        # Precompile regex patterns
        self.section_header_regex = [re.compile(pattern) for pattern in self.section_header_patterns]
        self.toc_regex = [re.compile(pattern, re.IGNORECASE) for pattern in self.toc_patterns]
        self.toc_entry_regex = [re.compile(pattern) for pattern in self.toc_entry_patterns]
        
        # Table of contents detection settings
        toc_config = self.config.get("toc_detection", {})
        self.detect_toc = toc_config.get("enabled", True)
        self.min_toc_entries = toc_config.get("min_entries", 3)
        self.max_toc_entries = toc_config.get("max_entries", 150)  # Increased from 100 to catch longer TOCs
        self.max_toc_section_length = toc_config.get("max_section_length", 8000)  # Increased from 5000 to handle larger sections
        # How aggressively to try to find TOCs - higher means more attempts but might have more false positives
        self.toc_detection_aggressiveness = toc_config.get("aggressiveness", 2)  # 1 = conservative, 2 = normal, 3 = aggressive
    
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
        
        # Detect and extract table of contents if enabled
        if self.detect_toc:
            # First try to extract TOC from text content
            toc_info = self._detect_table_of_contents(document["text"], segments)
            
            # If no TOC was found in the text and this is a PDF document with document-level TOC,
            # try to use the PDF's built-in bookmarks as a potential TOC
            if not (toc_info and toc_info["entries"]) and document.get("extraction_method") == "pdf":
                # Check if we have document metadata with outline/bookmarks
                if "pdf_outline" in document or "pdf_bookmarks" in document:
                    toc_info = self._extract_toc_from_pdf_outline(document)
                    if toc_info:
                        logger.info(f"Extracted TOC from PDF document outline with {len(toc_info['entries'])} entries")
            
            # Set the TOC info in the result if available
            if toc_info and toc_info["entries"]:
                result["table_of_contents"] = toc_info
                logger.info(f"Detected table of contents with {len(toc_info['entries'])} entries")
        
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
    
    def _detect_table_of_contents(self, text: str, segments: List[Dict]) -> Optional[Dict]:
        """Detect and extract a table of contents or index from document text.
        
        Supports detection of tables of contents in multiple European languages.
        
        Args:
            text: Document text
            segments: List of segments already extracted
            
        Returns:
            Dictionary with table of contents information or None if not detected
        """
        if not text or not self.detect_toc:
            return None
            
        # First check if we have a TOC header in the first ~25% of the document
        # Tables of contents are typically at the start of documents
        first_part_limit = min(len(text), 15000)  # Increased to 15K chars to better handle larger documents
        first_part = text[:first_part_limit]
        paragraphs = first_part.split('\n\n')
        
        toc_start_idx = None
        toc_title = None
        
        # Look for TOC header markers
        # Check more paragraphs (30 instead of 20) to better handle documents with longer introductions
        for i, para in enumerate(paragraphs[:30]):
            para_lines = para.strip().split('\n')
            for line in para_lines:
                line_clean = line.strip()
                # Check if this line matches a TOC header pattern
                if any(pattern.search(line_clean) for pattern in self.toc_regex):
                    toc_start_idx = i
                    toc_title = line_clean
                    logger.debug(f"Found potential TOC header: '{toc_title}' at paragraph {i}")
                    break
            if toc_start_idx is not None:
                break
        
        # Fallback 1: if no TOC header was found by exact pattern match, 
        # try looking for common TOC words anywhere in short paragraphs
        if toc_start_idx is None:
            common_toc_words = [
                "contents", "index", "indice", "sommario", "tabla", "inhaltsverzeichnis", 
                "table des matières", "índice", "sumário", "spis treści", "obsah", "inhoud",
                "περιεχόμενα", "sisällys", "cuprins", "tartalomjegyzék"
            ]
            for i, para in enumerate(paragraphs[:50]):  # Look in more paragraphs for the fallback
                para_text = para.strip().lower()
                # Only check short paragraphs (likely to be headers)
                if 3 <= len(para_text) <= 100:
                    for word in common_toc_words:
                        if word in para_text:
                            toc_start_idx = i
                            toc_title = para.strip()
                            logger.debug(f"Found potential TOC header via fallback 1: '{toc_title}' at paragraph {i}")
                            break
                    if toc_start_idx is not None:
                        break
        
        # Fallback 2: if still no TOC header found, try detecting consecutive paragraphs with page numbers
        # This catches TOCs that don't have a clear header or where the header wasn't recognized
        if toc_start_idx is None:
            for i in range(min(50, len(paragraphs))):
                # Skip very short paragraphs
                if len(paragraphs[i].strip()) < 5:
                    continue
                    
                # Check if this paragraph and the next few look like TOC entries
                toc_like_count = 0
                non_toc_count = 0
                
                # Check up to 10 consecutive paragraphs
                for j in range(i, min(i + 10, len(paragraphs))):
                    para_lines = paragraphs[j].strip().split('\n')
                    para_toc_lines = 0
                    
                    for line in para_lines:
                        line = line.strip()
                        if not line:
                            continue
                            
                        # Check if this line looks like a TOC entry (contains dots/spaces followed by numbers at the end)
                        has_page_num = bool(re.search(r'[\.\s]+\d+\s*$', line))
                        if has_page_num:
                            para_toc_lines += 1
                            
                    if para_toc_lines > 0:
                        toc_like_count += 1
                    else:
                        non_toc_count += 1
                        
                    # If we have 3+ TOC-like paragraphs and not too many non-TOC ones, consider this a TOC
                    if toc_like_count >= 3 and non_toc_count <= 1:
                        toc_start_idx = i
                        toc_title = "Contents"  # Generic title since we don't have a real header
                        logger.debug(f"Found potential TOC via fallback 2 (pattern detection) at paragraph {i}")
                        break
                        
                if toc_start_idx is not None:
                    break
        if toc_start_idx is None:
            return None
            
        # Look for TOC entries in the paragraphs following the header
        toc_entries = []
        consecutive_toc_lines = 0
        non_toc_lines = 0
        
        # Analyze more paragraphs to catch longer TOCs
        max_paragraphs_to_check = min(len(paragraphs), toc_start_idx + 40)
        
        # Start from the paragraph after the TOC header
        for i in range(toc_start_idx + 1, max_paragraphs_to_check):
            para = paragraphs[i]
            para_lines = para.strip().split('\n')
            
            para_toc_lines = 0
            for line in para_lines:
                line = line.strip()
                if not line:
                    continue
                    
                # Check if this line matches a TOC entry pattern
                is_toc_entry = any(pattern.search(line) for pattern in self.toc_entry_regex)
                
                if is_toc_entry:
                    para_toc_lines += 1
                    # Extract the entry information
                    # Try to separate the title from the page number using multiple pattern approaches
                    
                    # Initialize variables for extraction results
                    extracted_title = None
                    extracted_page_num = None
                    extracted_section_num = None
                    extracted_level = 0
                    
                    # Try multiple patterns for extracting TOC information
                    
                    # Pattern 1: Title followed by dots/dashes and page number
                    dots_match = re.search(r'^(.+?)[\.\-_]{2,}\s*(\d+)\s*$', line)
                    
                    # Pattern 2: Title followed directly by page number
                    direct_match = re.search(r'^(.+?)\s+(\d+)\s*$', line)
                    
                    # Pattern 3: Numbered section with title and page number
                    numbered_match = re.search(r'^(\s*(?:\d+[\.\-])*\d+[\.\-]?)\s+(.+?)\s+(\d+)\s*$', line)
                    
                    # Pattern 4: Title with explicit page word
                    page_word_match = re.search(r'^(.+?)\s+(?:[Pp]age|[Pp]ágina|[Pp]agina|[Ss]eite|[Pp]\.?|страница|стр\.|strona|str\.|σελίδα|σελ\.|oldal|sivu|s\.|sida|strana|pagină|pag\.)\s+(\d+)\s*$', line)
                    
                    # Pattern 5: Roman numerals or lettered section headings
                    roman_match = re.search(r'^(\s*(?:[IVX]+|[A-Z])[\.\-])\s+(.+?)\s+(\d+)\s*$', line)
                    
                    # Pattern 6: Bracketed or parenthesized section numbers
                    bracket_match = re.search(r'^(\s*[\[\(](?:\d+|[A-Z]|[IVX]+)[\]\)])\s+(.+?)\s+(\d+)\s*$', line)
                    
                    # Pattern 7: Indented or bulleted entries (detect level from indentation)
                    indent_match = re.search(r'^(\s+|[»•◦▪‣◘○●·]+\s+)(.+?)\s+(\d+)\s*$', line)
                    
                    # Try matching against all patterns and use the first successful match
                    if dots_match:
                        extracted_title = dots_match.group(1).strip()
                        extracted_page_num = int(dots_match.group(2))
                        # Determine level based on indentation or bullet type if present
                        if extracted_title.startswith(('  ', '\t', '»', '•', '◦', '▪', '‣', '◘', '○', '●', '·')):
                            # Count leading whitespace or bullet characters to estimate level
                            leading_space = len(extracted_title) - len(extracted_title.lstrip())
                            extracted_level = min(3, leading_space // 2)
                            extracted_title = extracted_title.lstrip()
                            
                    elif numbered_match:
                        extracted_section_num = numbered_match.group(1).strip()
                        extracted_title = numbered_match.group(2).strip()
                        extracted_page_num = int(numbered_match.group(3))
                        # Determine level based on the depth of section numbering
                        extracted_level = max(extracted_section_num.count('.'), extracted_section_num.count('-')) 
                        
                    elif roman_match:
                        extracted_section_num = roman_match.group(1).strip()
                        extracted_title = roman_match.group(2).strip()
                        extracted_page_num = int(roman_match.group(3))
                        # Roman numerals or single letters typically indicate top-level sections
                        extracted_level = 0
                        
                    elif bracket_match:
                        extracted_section_num = bracket_match.group(1).strip()
                        extracted_title = bracket_match.group(2).strip()
                        extracted_page_num = int(bracket_match.group(3))
                        # Bracketed numbers typically indicate top or second-level sections
                        extracted_level = 1
                        
                    elif page_word_match:
                        extracted_title = page_word_match.group(1).strip()
                        extracted_page_num = int(page_word_match.group(2))
                        
                    elif indent_match:
                        indent = indent_match.group(1)
                        extracted_title = indent_match.group(2).strip()
                        extracted_page_num = int(indent_match.group(3))
                        # Estimate level from indentation
                        if '\t' in indent:
                            extracted_level = indent.count('\t')
                        else:
                            extracted_level = min(3, len(indent) // 2)
                        
                    elif direct_match:
                        # This is the most general pattern and can produce false positives,
                        # so we apply it last and with additional verification
                        title_candidate = direct_match.group(1).strip()
                        page_candidate = int(direct_match.group(2))
                        
                        # Verify this looks like a genuine TOC entry:
                        # - Page number should be reasonable (not too large for most books)
                        # - Title shouldn't be too short or too long
                        if (1 <= page_candidate <= 5000) and (5 <= len(title_candidate) <= 200):
                            extracted_title = title_candidate
                            extracted_page_num = page_candidate
                            
                            # Try to determine level from indentation or formatting
                            leading_space = len(title_candidate) - len(title_candidate.lstrip())
                            if leading_space > 0:
                                extracted_level = min(3, leading_space // 2)
                                extracted_title = extracted_title.lstrip()
                    
                    # Create the TOC entry from extracted information
                    if extracted_title and extracted_page_num:
                        entry = {
                            "title": extracted_title,
                            "page": extracted_page_num,
                            "level": extracted_level,
                            "text": line
                        }
                        
                        # Add section number if available
                        if extracted_section_num:
                            entry["section"] = extracted_section_num
                            
                        toc_entries.append(entry)
                    else:
                        # If extraction failed but the line matched a TOC pattern,
                        # add it with minimal information
                        toc_entries.append({
                            "text": line,
                            "level": 0
                        })
                else:
                    non_toc_lines += 1
            
            consecutive_toc_lines += para_toc_lines
            
            # If we've found a reasonable number of TOC entries but then hit a 
            # paragraph with no TOC-like lines, we've probably reached the end of the TOC
            if consecutive_toc_lines >= self.min_toc_entries and para_toc_lines == 0:
                # If we have enough entries already, allow up to 2 non-TOC paragraphs
                # since some TOCs have section breaks or other formatting
                if len(toc_entries) >= self.min_toc_entries * 2:
                    non_toc_paragraph_count = 0
                    for next_i in range(i, min(i + 3, max_paragraphs_to_check)):
                        para = paragraphs[next_i]
                        para_lines = para.strip().split('\n')
                        has_toc_line = False
                        
                        for line in para_lines:
                            if any(pattern.search(line.strip()) for pattern in self.toc_entry_regex):
                                has_toc_line = True
                                break
                                
                        if has_toc_line:
                            # Continue processing this paragraph in the main loop
                            continue
                        else:
                            non_toc_paragraph_count += 1
                            
                        if non_toc_paragraph_count >= 2:
                            break
                else:
                    # With fewer entries, be more conservative
                    break
                
            # If we have too many non-TOC lines mixed in, or reached max entries, stop
            if non_toc_lines > 15 or len(toc_entries) >= self.max_toc_entries:
                break
        
        # Validate that we found enough entries to consider this a real TOC
        if len(toc_entries) < self.min_toc_entries:
            logger.debug(f"Found only {len(toc_entries)} TOC entries, which is fewer than min_toc_entries ({self.min_toc_entries}). Not treating as TOC.")
            return None
            
        # Post-process TOC entries to ensure consistent level hierarchy
        # This helps with TOCs that don't have explicit level indicators
        self._post_process_toc_levels(toc_entries)
            
        # Mark TOC segments so they can be handled differently in chunking
        if segments:
            # Find the appropriate segments and mark them as TOC
            toc_start_pos = first_part.find(toc_title) if toc_title else -1
            if toc_start_pos >= 0:
                for segment in segments:
                    # Check if this segment is part of the TOC
                    if segment.get("text") and (
                        (toc_title and segment["text"].startswith(toc_title)) or
                        any(entry["text"] in segment["text"] for entry in toc_entries)
                    ):
                        segment["type"] = "toc"
                        segment["is_toc"] = True
                        logger.debug(f"Marked segment as TOC: {segment['text'][:50]}...")
        
        # Add language detection for the TOC if possible
        toc_language = self._detect_toc_language(toc_title, toc_entries)
        
        # Create the final TOC info
        toc_info = {
            "title": toc_title,
            "entries": toc_entries,
            "entry_count": len(toc_entries),
            "language": toc_language
        }
        
        logger.info(f"Detected table of contents in language '{toc_language}' with {len(toc_entries)} entries")
        return toc_info
        
    def _post_process_toc_levels(self, toc_entries: List[Dict]) -> None:
        """Ensure consistent level hierarchy in TOC entries.
        
        Args:
            toc_entries: List of TOC entry dictionaries to process
        """
        if not toc_entries:
            return
            
        # First pass: Identify entries that have section numbers and ensure their levels are correct
        for entry in toc_entries:
            if "section" in entry:
                section_num = entry["section"]
                # Determine level based on depth of numbering
                if re.match(r'\d+\.\d+\.\d+\.\d+', section_num):
                    entry["level"] = 3  # Four-level deep
                elif re.match(r'\d+\.\d+\.\d+', section_num):
                    entry["level"] = 2  # Three-level deep
                elif re.match(r'\d+\.\d+', section_num):
                    entry["level"] = 1  # Two-level deep
                elif re.match(r'\d+\.?', section_num):
                    entry["level"] = 0  # Top level
        
        # Second pass: For entries without explicit levels, infer from surrounding entries
        # This uses a heuristic approach based on the assumption that TOCs are hierarchical
        for i in range(1, len(toc_entries)):
            current = toc_entries[i]
            previous = toc_entries[i-1]
            
            # If the current entry doesn't have a defined level or section number
            if "level" not in current or ("section" not in current and current["level"] == 0):
                # Analyze the entry text to determine if it's likely a sub-entry
                if len(current["text"]) - len(current["text"].lstrip()) > len(previous["text"]) - len(previous["text"].lstrip()):
                    # More indentation than previous entry suggests a sub-level
                    current["level"] = min(3, previous["level"] + 1)
                elif current["text"].startswith(('  ', '\t', '»', '•', '◦', '▪', '‣')):
                    # Entry starts with indentation or bullet characters
                    current["level"] = min(3, previous["level"] + 1)
                
        # Third pass: Ensure no level gaps (don't jump from level 0 to level 2 without a level 1)
        for i in range(1, len(toc_entries)):
            current = toc_entries[i]
            previous = toc_entries[i-1]
            
            if current["level"] > previous["level"] + 1:
                # Adjust to avoid skipping levels
                current["level"] = previous["level"] + 1
    
    def _extract_toc_from_pdf_outline(self, document: Dict) -> Optional[Dict]:
        """Extract table of contents from PDF outline/bookmarks if available.
        
        Args:
            document: Document dictionary with PDF metadata
            
        Returns:
            TOC information dictionary or None if not available
        """
        # Check which key contains the outline information
        outline_data = None
        if "pdf_outline" in document:
            outline_data = document["pdf_outline"]
        elif "pdf_bookmarks" in document:
            outline_data = document["pdf_bookmarks"]
            
        if not outline_data:
            return None
            
        # Process outline data into TOC entries
        toc_entries = []
        
        # Try to extract entries from different outline formats
        try:
            # Convert the outline to TOC entries
            if isinstance(outline_data, list):
                for i, item in enumerate(outline_data):
                    if isinstance(item, dict):
                        # Common PDF outline format with title and page
                        if "title" in item and ("page" in item or "pagenum" in item):
                            page_num = item.get("page", item.get("pagenum", 0))
                            # Convert to int if it's a string
                            if isinstance(page_num, str):
                                try:
                                    page_num = int(page_num)
                                except ValueError:
                                    page_num = 0
                                    
                            level = item.get("level", 0)
                            if isinstance(level, str):
                                try:
                                    level = int(level)
                                except ValueError:
                                    level = 0
                                    
                            entry = {
                                "title": item["title"],
                                "page": page_num,
                                "level": level,
                                "text": item["title"]
                            }
                            toc_entries.append(entry)
                            
                        # PyMuPDF format with title and page
                        elif "title" in item and "page" in item:
                            entry = {
                                "title": item["title"],
                                "page": item["page"],
                                "level": item.get("level", 0),
                                "text": item["title"]
                            }
                            toc_entries.append(entry)
                    elif isinstance(item, (list, tuple)) and len(item) >= 2:
                        # Simple format [title, page_num, level?]
                        title = item[0]
                        page_num = item[1]
                        level = item[2] if len(item) > 2 else 0
                        
                        if isinstance(page_num, str):
                            try:
                                page_num = int(page_num)
                            except ValueError:
                                page_num = 0
                                
                        entry = {
                            "title": title,
                            "page": page_num,
                            "level": level,
                            "text": title
                        }
                        toc_entries.append(entry)
            
            # If we found any valid entries, create a TOC info structure
            if toc_entries:
                # Determine language
                toc_language = self._detect_toc_language(None, toc_entries)
                
                return {
                    "title": "Table of Contents",
                    "entries": toc_entries,
                    "entry_count": len(toc_entries),
                    "language": toc_language,
                    "source": "pdf_outline"
                }
        except Exception as e:
            logger.warning(f"Error extracting TOC from PDF outline: {e}")
            
        return None
    
    def _detect_toc_language(self, toc_title: Optional[str], toc_entries: List[Dict]) -> str:
        """Detect the language of the table of contents.
        
        Args:
            toc_title: The title of the table of contents
            toc_entries: List of TOC entry dictionaries
            
        Returns:
            ISO language code or 'unknown'
        """
        if not toc_title and not toc_entries:
            return "unknown"
            
        # Language signature patterns
        language_patterns = {
            "en": [r"contents", r"index", r"table of contents", r"chapter", r"section", r"page"],
            "it": [r"indice", r"sommario", r"contenuti", r"capitolo", r"sezione", r"pagina"],
            "es": [r"índice", r"tabla de contenidos", r"contenidos", r"capítulo", r"sección", r"página"],
            "fr": [r"table des matières", r"sommaire", r"chapitre", r"section", r"page"],
            "de": [r"inhaltsverzeichnis", r"inhalt", r"kapitel", r"abschnitt", r"seite"],
            "pt": [r"índice", r"sumário", r"conteúdo", r"capítulo", r"seção", r"página"],
            "nl": [r"inhoudsopgave", r"inhoud", r"hoofdstuk", r"sectie", r"pagina"],
            "pl": [r"spis treści", r"indeks", r"rozdział", r"sekcja", r"strona"],
            "sv": [r"innehållsförteckning", r"innehåll", r"kapitel", r"avsnitt", r"sida"],
            "da": [r"indholdsfortegnelse", r"indhold", r"kapitel", r"afsnit", r"side"],
            "el": [r"περιεχόμενα", r"ευρετήριο", r"κεφάλαιο", r"ενότητα", r"σελίδα"],
            "fi": [r"sisällysluettelo", r"sisältö", r"luku", r"kohta", r"sivu"],
            "ro": [r"cuprins", r"conținut", r"capitol", r"secțiune", r"pagină"],
            "hu": [r"tartalomjegyzék", r"tartalom", r"fejezet", r"szakasz", r"oldal"],
            "cs": [r"obsah", r"rejstřík", r"kapitola", r"část", r"strana"]
        }
        
        # Combine all text to analyze
        all_text = toc_title or ""
        for entry in toc_entries[:min(10, len(toc_entries))]:  # Use only first 10 entries
            if "title" in entry:
                all_text += " " + entry["title"]
            elif "text" in entry:
                all_text += " " + entry["text"]
                
        all_text = all_text.lower()
        
        # Count matches for each language
        language_scores = {}
        for lang, patterns in language_patterns.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, all_text, re.IGNORECASE):
                    score += 1
            language_scores[lang] = score
            
        # Find the language with the highest score
        max_score = 0
        detected_language = "unknown"
        
        for lang, score in language_scores.items():
            if score > max_score:
                max_score = score
                detected_language = lang
                
        # If score is too low, return unknown
        if max_score < 2:
            return "unknown"
            
        return detected_language
        
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
    
    def _custom_sentence_tokenize(self, text: str) -> List[str]:
        """Custom sentence tokenization as fallback when NLTK fails.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of sentences
        """
        if not text:
            return []
            
        # Split on common sentence terminators with positive lookbehind
        # This handles periods, question marks, and exclamation points
        import re
        
        # Pattern looks for sentence terminators followed by spaces and capital letters
        # Handles various end-of-sentence punctuations correctly
        pattern = r'(?<=[.!?])\s+(?=[A-Z])'
        sentences = re.split(pattern, text)
        
        # Further process to handle list items, headings, etc.
        processed_sentences = []
        for sentence in sentences:
            # If very long, try to split further at newlines that appear to separate sentences
            if len(sentence) > 200:
                # Split on newlines that are followed by capital letters, numbers at the start of a line, etc.
                newline_pattern = r'\n+(?=[A-Z]|\d+[\.\)]|\*\s|•\s)'
                sub_sentences = re.split(newline_pattern, sentence)
                processed_sentences.extend([s.strip() for s in sub_sentences if s.strip()])
            else:
                processed_sentences.append(sentence.strip())
                
        # Make sure each sentence appears to end with sentence-ending punctuation
        final_sentences = []
        for s in processed_sentences:
            if s and not s[-1] in '.!?;:':
                s += '.'
            if s:
                final_sentences.append(s)
                
        return final_sentences
    
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
        
        # Check for section/chapter headers in multiple languages
        # English
        if re.match(r"^(?:Chapter|CHAPTER|Section|SECTION)", header):
            return 1
        elif re.match(r"^(?:Subsection|SUBSECTION|Part|PART)", header):
            return 2
            
        # Italian
        if re.match(r"^(?:Capitolo|CAPITOLO|Sezione|SEZIONE)", header):
            return 1
        elif re.match(r"^(?:Sottosezione|SOTTOSEZIONE|Parte|PARTE)", header):
            return 2
            
        # Spanish
        if re.match(r"^(?:Capítulo|CAPÍTULO|Sección|SECCIÓN)", header):
            return 1
        elif re.match(r"^(?:Subsección|SUBSECCIÓN|Parte|PARTE)", header):
            return 2
            
        # French
        if re.match(r"^(?:Chapitre|CHAPITRE|Section|SECTION)", header):
            return 1
        elif re.match(r"^(?:Sous-section|SOUS-SECTION|Partie|PARTIE)", header):
            return 2
            
        # German
        if re.match(r"^(?:Kapitel|KAPITEL|Abschnitt|ABSCHNITT)", header):
            return 1
        elif re.match(r"^(?:Unterabschnitt|UNTERABSCHNITT|Teil|TEIL)", header):
            return 2
            
        # Portuguese
        if re.match(r"^(?:Capítulo|CAPÍTULO|Seção|SEÇÃO)", header):
            return 1
        elif re.match(r"^(?:Subseção|SUBSEÇÃO|Parte|PARTE)", header):
            return 2
            
        # Other European languages
        if re.match(r"^(?:Hoofdstuk|Rozdział|Kapittel|Κεφάλαιο|Luku|Capitol|Fejezet|Kapitola)", header):
            return 1
        
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
    
    def _normalize_text(self, text: str) -> str:
        """Normalize and clean text, standardizing quotes and special characters.
        
        Args:
            text: Input text with potentially problematic characters
            
        Returns:
            Normalized text with standardized quotes and special characters
        """
        if not text:
            return ""
            
        # Use the Unicode normalizer utility if available
        if UNICODE_NORMALIZER_AVAILABLE:
            return UnicodeNormalizer.clean_for_display(text)
            
        # Fallback to basic normalization if the utility is not available
        normalized_text = text
        
        # Normalize quotes
        normalized_text = normalized_text.replace('"', '"').replace('"', '"')
        normalized_text = normalized_text.replace(''', "'").replace(''', "'")
        
        # Normalize dashes
        normalized_text = normalized_text.replace('—', '-').replace('–', '-')
        
        # Normalize ellipsis
        normalized_text = normalized_text.replace('…', '...')
        
        # For more comprehensive normalization, use NFKD normalization
        normalized_text = unicodedata.normalize('NFKD', normalized_text)
        
        return normalized_text
    
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
            
        # Normalize text to handle problematic characters
        normalized_text = self._normalize_text(chunk_text)
        
        try:
            # Use NLTK to tokenize sentences with robust fallback
            try:
                # Try NLTK's sent_tokenize (preferred method)
                from nltk.tokenize import sent_tokenize
                try:
                    sentences = sent_tokenize(normalized_text)
                except LookupError as e:
                    # If missing punkt data, try to download it
                    logger.info(f"Downloading NLTK punkt tokenizer due to: {e}")
                    import nltk
                    # Download both punkt and punkt_tab resources
                    nltk.download('punkt', quiet=True)
                    try:
                        nltk.download('punkt_tab', quiet=True)
                    except:
                        logger.warning("Could not download punkt_tab, continuing with punkt only")
                    try:
                        sentences = sent_tokenize(normalized_text)
                    except Exception as e2:
                        logger.warning(f"Still could not use sent_tokenize: {e2}")
                        raise e2  # Will be caught by outer try/except
            except Exception as ex:
                # If any other error occurs, fallback to manual sentence splitting
                logger.warning(f"Using custom sentence tokenization due to NLTK error: {ex}")
                sentences = self._custom_sentence_tokenize(normalized_text)
                
            if not sentences:
                # Fallback if tokenization returns empty list
                sentences = [s.strip() + "." for s in normalized_text.split('.') if s.strip()]
                
            if not sentences:
                return normalized_text[:self.summary_max_length]
                
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
                return self._normalize_text(truncated)
            
            # Final normalization to ensure all problematic characters are handled
            return self._normalize_text(full_summary)
            
        except Exception as e:
            logger.warning(f"Failed to generate summary: {str(e)}", exc_info=True)
            # Fallback to a simple first N chars approach
            fallback_text = chunk_text[:self.summary_max_length] + "..." if len(chunk_text) > self.summary_max_length else chunk_text
            # Apply normalization to the fallback text as well
            return self._normalize_text(fallback_text)
    
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
                    summary = self._generate_chunk_summary(chunk_text)
                    # Ensure the summary is also normalized
                    chunk_dict["summary"] = self._normalize_text(summary)
                
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
                summary = self._generate_chunk_summary(chunk_text)
                # Ensure the summary is also normalized
                chunk_dict["summary"] = self._normalize_text(summary)
            
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
                        first_summary = self._generate_chunk_summary(first_half_text)
                        second_summary = self._generate_chunk_summary(second_half_text)
                        # Ensure the summaries are also normalized
                        first_half["summary"] = self._normalize_text(first_summary)
                        second_half["summary"] = self._normalize_text(second_summary)
                    
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
                    summary = self._generate_chunk_summary(chunk_text_cleaned)
                    # Ensure the summary is also normalized
                    chunk_dict["summary"] = self._normalize_text(summary)
                
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
                        summary = self._generate_chunk_summary(chunk_text_cleaned)
                        # Ensure the summary is also normalized
                        chunk_dict["summary"] = self._normalize_text(summary)
                    
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
                summary = self._generate_chunk_summary(merged_chunk["text"])
                # Ensure the summary is also normalized
                merged_chunk["summary"] = self._normalize_text(summary)
                
            final_chunks.append(merged_chunk)
            
        # Log chunking statistics
        logger.info(f"Hybrid chunking: {len(size_chunks)} size chunks → {len(final_chunks)} final chunks")
        
        # Add chunk IDs
        for i, chunk in enumerate(final_chunks):
            chunk["id"] = f"chunk_{i+1}"
        
        return final_chunks