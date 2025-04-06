"""Module for spelling correction in extracted texts."""

import re
from typing import Dict, List, Optional, Set, Union
import unicodedata

class SpellingCorrector:
    """Corrects spelling errors in text, especially OCR-induced errors."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the spelling corrector.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Configure spelling settings
        spelling_config = self.config.get("spelling", {})
        self.enabled = spelling_config.get("enabled", True)
        self.language = spelling_config.get("language", "auto")
        self.backend = spelling_config.get("backend", "symspellpy")
        self.max_edit_distance = spelling_config.get("max_edit_distance", 2)
        
        # Initialize language detection
        self.language_detector = None
        if self.language == "auto":
            try:
                import langdetect
                self.language_detector = langdetect
            except ImportError:
                print("Warning: langdetect package not found. Using 'en' as default language.")
        
        # Initialize spelling backend
        self._initialize_backend()
        
        # Common OCR substitution errors
        self.ocr_substitutions = {
            # Common OCR errors
            "m": ["rn"],  # 'm' mistaken as 'rn'
            "h": ["li", "l1"],  # 'h' mistaken as 'li' or 'l1'
            "n": ["ri"],  # 'n' mistaken as 'ri'
            "w": ["vv", "1/v"],  # 'w' mistaken as 'vv' or '1/v'
            "b": ["lo", "l0"],  # 'b' mistaken as 'lo' or 'l0'
            "d": ["cl", "ol"],  # 'd' mistaken as 'cl' or 'ol'
            "0": ["O", "o"],  # '0' mistaken as 'O' or 'o'
            "1": ["l", "I"],  # '1' mistaken as 'l' or 'I'
            "5": ["S", "s"],  # '5' mistaken as 'S' or 's'
            "8": ["B"],  # '8' mistaken as 'B'
            "B": ["8", "13"],  # 'B' mistaken as '8' or '13'
            "D": ["0"],  # 'D' mistaken as '0'
            "I": ["1", "l"],  # 'I' mistaken as '1' or 'l'
            "O": ["0", "D"],  # 'O' mistaken as '0' or 'D'
            "S": ["5"],  # 'S' mistaken as '5'
            "Z": ["2"],  # 'Z' mistaken as '2'
        }
        
        # Additional custom substitutions from config
        custom_substitutions = spelling_config.get("ocr_substitutions", {})
        if custom_substitutions:
            for key, values in custom_substitutions.items():
                if key in self.ocr_substitutions:
                    self.ocr_substitutions[key].extend(values)
                else:
                    self.ocr_substitutions[key] = values
    
    def _initialize_backend(self):
        """Initialize the spelling correction backend."""
        self.spellchecker = None
        
        if not self.enabled:
            return
        
        # Try each backend in order until one succeeds
        backends = []
        if self.backend == "symspellpy":
            backends = ["symspellpy", "textblob", "wordfreq"]
        elif self.backend == "textblob":
            backends = ["textblob", "wordfreq", "symspellpy"]
        else:
            backends = ["wordfreq", "symspellpy", "textblob"]
            
        # Try each backend in preferred order
        for backend in backends:
            if backend == "symspellpy":
                success = self._initialize_symspellpy()
                if success:
                    self.backend = "symspellpy"
                    return
            elif backend == "textblob":
                success = self._initialize_textblob()
                if success:
                    self.backend = "textblob"
                    return
            elif backend == "wordfreq":
                success = self._initialize_wordfreq()
                if success:
                    self.backend = "wordfreq"
                    return
                    
        # If we got here, no backend was successful
        print("WARNING: No spelling correction backend could be initialized. Spelling correction disabled.")
    
    def _initialize_symspellpy(self):
        """Initialize SymSpellPy backend."""
        try:
            from symspellpy import SymSpell, Verbosity
            import pkg_resources
            
            sym_spell = SymSpell(max_dictionary_edit_distance=self.max_edit_distance, prefix_length=7)
            
            # Get appropriate dictionary based on language
            lang_code = self._get_language_code()
            dictionary_path = pkg_resources.resource_filename(
                "symspellpy", f"frequency_dictionary_{lang_code}_82_765.txt"
            )
            
            # If specific language not available, fall back to English
            if not pkg_resources.resource_exists("symspellpy", f"frequency_dictionary_{lang_code}_82_765.txt"):
                dictionary_path = pkg_resources.resource_filename(
                    "symspellpy", "frequency_dictionary_en_82_765.txt"
                )
                
            # Load dictionary
            if not sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1):
                print(f"Warning: Failed to load SymSpellPy dictionary for {lang_code}")
                return False
                
            self.spellchecker = sym_spell
            self.verbosity = Verbosity.CLOSEST  # Use the closest match
            print(f"Initialized SymSpellPy with {lang_code} dictionary")
            return True
            
        except ImportError:
            print("Warning: symspellpy package not found. Trying alternative backends.")
            self.spellchecker = None
            return False
        except Exception as e:
            print(f"Error initializing SymSpellPy: {str(e)}. Trying alternative backends.")
            self.spellchecker = None
            return False
    
    def _initialize_textblob(self):
        """Initialize TextBlob backend."""
        try:
            from textblob import TextBlob
            self.spellchecker = TextBlob
            print("Initialized TextBlob spelling correction")
            return True
        except ImportError:
            print("Warning: textblob package not found. Trying alternative backends.")
            self.spellchecker = None
            return False
        except Exception as e:
            print(f"Error initializing TextBlob: {str(e)}. Trying alternative backends.")
            self.spellchecker = None
            return False
    
    def _initialize_wordfreq(self):
        """Initialize wordfreq-based spelling correction as a fallback."""
        try:
            from wordfreq import word_frequency
            
            def correct_word(word):
                # Only suggest corrections for words that are unlikely
                if word_frequency(word, self._get_language_code()) > 1e-6:
                    return word
                    
                # Try some basic corrections
                candidates = self._generate_candidates(word)
                best_candidate = word
                best_freq = word_frequency(word, self._get_language_code())
                
                for candidate in candidates:
                    freq = word_frequency(candidate, self._get_language_code())
                    if freq > best_freq:
                        best_candidate = candidate
                        best_freq = freq
                        
                return best_candidate
            
            self.spellchecker = {
                "correct_word": correct_word
            }
            print("Initialized wordfreq-based spelling correction")
            return True
        except ImportError:
            print("Warning: wordfreq package not found. No spelling correction will be available.")
            self.spellchecker = None
            return False
        except Exception as e:
            print(f"Error initializing wordfreq: {str(e)}. No spelling correction will be available.")
            self.spellchecker = None
            return False
    
    def _get_language_code(self):
        """Get the language code to use for spelling correction."""
        if self.language != "auto":
            return self.language
            
        # Default to English if language detection is not available
        if not self.language_detector:
            return "en"
            
        # Use previously detected language if available
        if hasattr(self, "detected_language") and self.detected_language:
            return self.detected_language
            
        # Default to English
        return "en"
    
    def _detect_language(self, text):
        """Detect the language of the text."""
        if not self.language_detector or self.language != "auto":
            return self._get_language_code()
            
        try:
            # Use a sample of the text for faster detection
            sample = text[:5000]
            self.detected_language = self.language_detector.detect(sample)
            
            # Map to language codes supported by spelling backends
            lang_map = {
                "en": "en",
                "fr": "fr",
                "es": "es",
                "it": "it",
                "de": "de",
                "pt": "pt",
                "nl": "nl",
                "ru": "ru",
                "sv": "sv",
                "pl": "pl",
                "cs": "cs",
                "hu": "hu",
                "da": "da",
                "no": "no",
                "fi": "fi"
            }
            
            # Default to English if language not supported
            if self.detected_language not in lang_map:
                return "en"
                
            return lang_map[self.detected_language]
        except:
            return "en"  # Default to English on error
    
    def _generate_candidates(self, word):
        """Generate spelling candidates for a word based on common OCR errors."""
        candidates = set([word])
        
        # Skip very short words and non-words
        if len(word) < 3 or not re.match(r'^[a-zA-Z]+$', word):
            return candidates
        
        # Generate candidates by replacing potential OCR errors
        for idx, char in enumerate(word):
            if char in self.ocr_substitutions:
                for subst in self.ocr_substitutions[char]:
                    new_word = word[:idx] + subst + word[idx+1:]
                    candidates.add(new_word)
        
        # Generate candidates with common transformations
        candidates.add(word.lower())
        candidates.add(word.capitalize())
        
        # Add versions with diacritics removed
        normalized = ''.join((c for c in unicodedata.normalize('NFD', word) 
                            if unicodedata.category(c) != 'Mn'))
        candidates.add(normalized)
        
        return candidates
    
    def correct_word(self, word):
        """Correct a single word's spelling."""
        if not self.enabled or not self.spellchecker or not word:
            return word
            
        # Skip words that are too short, contain numbers, or special characters
        if len(word) < 3 or not re.match(r'^[a-zA-Z]+$', word):
            return word
        
        try:
            if self.backend == "symspellpy":
                suggestions = self.spellchecker.lookup(word.lower(), self.verbosity, 
                                                 max_edit_distance=self.max_edit_distance)
                if suggestions:
                    return suggestions[0].term
                return word
                
            elif self.backend == "textblob":
                blob = self.spellchecker(word)
                return str(blob.correct())
                
            else:  # wordfreq or custom backend
                return self.spellchecker["correct_word"](word)
                
        except Exception as e:
            print(f"Error correcting word '{word}': {str(e)}")
            return word
    
    def correct_text(self, text):
        """Correct spelling in a complete text."""
        if not self.enabled or not self.spellchecker or not text:
            return text
        
        # Detect language if in auto mode
        if self.language == "auto":
            self._detect_language(text)
        
        # Split text into words and non-words
        tokens = re.findall(r'[a-zA-Z]+|[^a-zA-Z]+', text)
        
        # Correct only words
        corrected_tokens = []
        for token in tokens:
            if re.match(r'^[a-zA-Z]+$', token):
                corrected_tokens.append(self.correct_word(token))
            else:
                corrected_tokens.append(token)
        
        # Join back into text
        return ''.join(corrected_tokens)
    
    def correct_document(self, document: Dict) -> Dict:
        """Correct spelling in a document.
        
        Args:
            document: Document dictionary with text and pages
            
        Returns:
            Document with corrected text
        """
        if not self.enabled or not self.spellchecker:
            print(f"Spelling correction skipped: enabled={self.enabled}, spellchecker={self.spellchecker is not None}")
            return document
            
        result = document.copy()
        
        # Detect language from the full document text
        detected_lang = None
        if self.language == "auto":
            detected_lang = self._detect_language(document["text"])
            print(f"Detected language for spelling correction: {detected_lang}")
        
        # Correct the full text
        print(f"Applying spelling correction to document text ({len(document['text'])} chars)...")
        result["text"] = self.correct_text(document["text"])
        
        # Correct each page
        corrected_pages = []
        for i, page in enumerate(document.get("pages", [])):
            print(f"Applying spelling correction to page {i+1}...")
            page_copy = page.copy()
            page_copy["text"] = self.correct_text(page["text"])
            corrected_pages.append(page_copy)
        
        result["pages"] = corrected_pages
        
        # Add spelling correction metadata
        result["metadata"] = result.get("metadata", {})
        result["metadata"]["spelling_corrected"] = True
        result["metadata"]["spelling_language"] = self._get_language_code()
        result["metadata"]["spelling_backend"] = self.backend
        
        print(f"Spelling correction completed using {self.backend} backend in {self._get_language_code()} language")
        return result