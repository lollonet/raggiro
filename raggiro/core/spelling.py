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
        
        # Synchronize with OCR language if available and auto is selected
        extraction_config = self.config.get("extraction", {})
        ocr_language = extraction_config.get("ocr_language", "")
        
        # Default language setting - auto or from config
        self.language = spelling_config.get("language", "auto")
        
        # Sync with OCR language if:
        # 1. OCR language is set (not empty)
        # 2. OCR language is not 'auto'
        # 3. Spelling language is 'auto' (otherwise user explicitly chose spelling language)
        if ocr_language and ocr_language != "auto" and self.language == "auto":
            # Map Tesseract language codes to spelling language codes
            ocr_to_spell_map = {
                "eng": "en",
                "ita": "it",
                "fra": "fr",
                "deu": "de",
                "spa": "es",
                "por": "pt"
            }
            
            # Extract primary language from Tesseract language string
            # (might be compound like "eng+ita", take first part)
            primary_ocr_lang = ocr_language.split("+")[0]
            
            if primary_ocr_lang in ocr_to_spell_map:
                # Use OCR language for spelling correction
                self.language = ocr_to_spell_map[primary_ocr_lang]
                print(f"Syncing spelling language with OCR: '{primary_ocr_lang}' → '{self.language}'")
        
        self.backend = spelling_config.get("backend", "standard")  # Changed default to 'standard'
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
        if self.backend == "standard":
            backends = ["pyspellchecker", "symspellpy", "textblob", "wordfreq"]
        elif self.backend == "symspellpy":
            backends = ["symspellpy", "pyspellchecker", "textblob", "wordfreq"]
        elif self.backend == "textblob":
            backends = ["textblob", "pyspellchecker", "wordfreq", "symspellpy"]
        else:
            backends = ["wordfreq", "pyspellchecker", "symspellpy", "textblob"]
            
        # Try each backend in preferred order
        for backend in backends:
            if backend == "pyspellchecker":
                success = self._initialize_pyspellchecker()
                if success:
                    self.backend = "standard"  # User-friendly name
                    return
            elif backend == "symspellpy":
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
        
    def _initialize_pyspellchecker(self):
        """Initialize pyspellchecker backend with standard dictionaries."""
        try:
            from spellchecker import SpellChecker
            
            # Get language code for pyspellchecker
            lang_code = self._get_language_code()
            
            # Map 2-letter code to pyspellchecker language name if needed
            lang_map = {
                "en": "en",
                "it": "it",
                "fr": "fr",
                "de": "de",
                "es": "es", 
                "pt": "pt",
                "ru": "ru",
            }
            
            spell_lang = lang_map.get(lang_code, "en")
            
            # Create spellchecker with the specific language
            try:
                spell = SpellChecker(language=spell_lang, distance=self.max_edit_distance)
                print(f"Successfully initialized Standard Spellchecker with {spell_lang} dictionary")
                
                # Store language info
                self.used_dictionary = f"standard-{spell_lang}"
                
                # Create wrapper for the API
                def correct_word(word):
                    # Skip very short words
                    if len(word) < 3:
                        return word
                    
                    # For words with numbers or special chars, skip correction
                    if not word.isalpha():
                        return word
                        
                    # Get the correction
                    return spell.correction(word)
                
                self.spellchecker = {
                    "correct_word": correct_word,
                    "instance": spell,
                }
                return True
                
            except ValueError as e:
                # This happens when the language is not supported
                print(f"Language '{spell_lang}' not supported by pyspellchecker: {str(e)}")
                # Try with English as fallback
                if spell_lang != "en":
                    try:
                        spell = SpellChecker(language="en", distance=self.max_edit_distance)
                        print(f"Falling back to English dictionary for pyspellchecker")
                        
                        self.used_dictionary = "standard-en"
                        
                        def correct_word(word):
                            if len(word) < 3 or not word.isalpha():
                                return word
                            return spell.correction(word)
                        
                        self.spellchecker = {
                            "correct_word": correct_word,
                            "instance": spell,
                        }
                        return True
                    except Exception as fallback_error:
                        print(f"Error initializing fallback English dictionary: {str(fallback_error)}")
                        return False
                return False
            except Exception as e:
                print(f"Error initializing pyspellchecker: {str(e)}")
                return False
                
        except ImportError:
            print("Warning: pyspellchecker package not found. Trying alternative backends.")
            return False
    
    def _initialize_symspellpy(self):
        """Initialize SymSpellPy backend."""
        try:
            from symspellpy import SymSpell, Verbosity
            import pkg_resources
            import os
            
            sym_spell = SymSpell(max_dictionary_edit_distance=self.max_edit_distance, prefix_length=7)
            
            # Get appropriate dictionary based on language
            lang_code = self._get_language_code()
            
            # Define all possible dictionary file patterns we might use
            dictionary_names = [
                f"frequency_dictionary_{lang_code}_82_765.txt",  # Standard name
                f"frequency_dictionary_{lang_code}.txt",         # Generic name
                "frequency_dictionary_en_82_765.txt"             # English fallback
            ]
            
            # Try to load language-specific dictionary first
            dictionary_loaded = False
            used_dictionary = None
            
            for dict_name in dictionary_names:
                try:
                    # Check if the dictionary exists in symspellpy package
                    if pkg_resources.resource_exists("symspellpy", dict_name):
                        dictionary_path = pkg_resources.resource_filename("symspellpy", dict_name)
                        
                        # Try to load the dictionary
                        if sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1):
                            dictionary_loaded = True
                            used_dictionary = dict_name
                            break
                except Exception as dict_err:
                    print(f"Error with dictionary {dict_name}: {str(dict_err)}")
            
            # If standard dictionaries failed, try to look for custom dictionaries
            if not dictionary_loaded:
                # Check for dictionaries in a custom directory
                custom_dict_dir = os.path.join(os.path.dirname(__file__), "..", "data", "dictionaries")
                if not os.path.exists(custom_dict_dir):
                    try:
                        # Try to create the directory
                        os.makedirs(custom_dict_dir, exist_ok=True)
                    except:
                        pass
                
                # Look for custom dictionaries matching the language
                custom_dict_path = None
                for file in os.listdir(custom_dict_dir) if os.path.exists(custom_dict_dir) else []:
                    if file.startswith(f"frequency_dictionary_{lang_code}") and file.endswith(".txt"):
                        custom_dict_path = os.path.join(custom_dict_dir, file)
                        break
                
                # If a custom dictionary was found, try to load it
                if custom_dict_path and os.path.exists(custom_dict_path):
                    try:
                        if sym_spell.load_dictionary(custom_dict_path, term_index=0, count_index=1):
                            dictionary_loaded = True
                            used_dictionary = os.path.basename(custom_dict_path)
                    except Exception as custom_err:
                        print(f"Error loading custom dictionary: {str(custom_err)}")
            
            # If no dictionary was loaded, return False
            if not dictionary_loaded:
                print(f"Warning: Failed to load any dictionary for language '{lang_code}'")
                print("Available dictionaries in symspellpy package:")
                for resource in pkg_resources.resource_listdir("symspellpy", ""):
                    if resource.startswith("frequency_dictionary_"):
                        print(f"  - {resource}")
                return False
                
            self.spellchecker = sym_spell
            self.verbosity = Verbosity.CLOSEST  # Use the closest match
            self.used_dictionary = used_dictionary
            print(f"Initialized SymSpellPy with dictionary: {used_dictionary} for language '{lang_code}'")
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
        # If not auto or detector not available, use configured language
        if not self.language_detector or self.language != "auto":
            return self._get_language_code()
            
        try:
            # Use a sample of the text for faster detection
            # Use a larger sample for better accuracy
            sample = text[:8000]
            
            # Skip if sample is too short
            if len(sample.strip()) < 50:
                print("Text sample too short for reliable language detection")
                return self._get_language_code()
            
            # Use langdetect with more reliable detection
            try:
                # Try to get language probabilities for more confidence
                from langdetect import detect_langs
                lang_probs = detect_langs(sample)
                
                # Log the detected language probabilities
                print(f"Language detection results: {lang_probs}")
                
                # Use the highest probability language
                if lang_probs:
                    highest_prob = lang_probs[0]
                    self.detected_language = highest_prob.lang
                    print(f"Detected language: {self.detected_language} with probability {highest_prob.prob:.2f}")
                else:
                    # Fallback to simple detection
                    self.detected_language = self.language_detector.detect(sample)
                    print(f"Detected language (simple method): {self.detected_language}")
            except:
                # Fallback to simple detection if detect_langs fails
                self.detected_language = self.language_detector.detect(sample)
                print(f"Detected language (fallback method): {self.detected_language}")
            
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
                print(f"Detected language '{self.detected_language}' not supported for spelling correction. Using English.")
                return "en"
                
            detected_code = lang_map[self.detected_language]
            print(f"Using language code '{detected_code}' for spelling correction")
            return detected_code
        except Exception as e:
            print(f"Error in language detection: {str(e)}. Using default language.")
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
        
        # Save the original text for comparison purposes
        result["original_text"] = document["text"]
        
        # Log spelling correction settings
        print("-"*80)
        print(f"SPELLING CORRECTION SETTINGS:")
        print(f"Language: {self.language}")
        print(f"Backend: {self.backend}")
        print(f"Max edit distance: {self.max_edit_distance}")
        
        # Show dictionary details
        if hasattr(self, 'used_dictionary'):
            print(f"Dictionary: {self.used_dictionary}")
            
        # Show specific backend details
        if self.backend == "standard" and hasattr(self.spellchecker, "instance"):
            spell = self.spellchecker["instance"]
            try:
                # Try to print word count to show dictionary size
                dict_size = len(spell._words)
                print(f"Standard dictionary loaded with {dict_size} words")
                
                # Show a sample of dictionary words (first 20)
                sample_words = list(spell._words.keys())[:20]
                print(f"Sample dictionary words: {', '.join(sample_words)}")
            except:
                pass
                
        # Show OCR language synchronization
        extraction_config = self.config.get("extraction", {})
        ocr_language = extraction_config.get("ocr_language", "")
        if ocr_language:
            print(f"OCR language setting: {ocr_language}")
            print(f"Language sync: {'enabled' if self.language != 'auto' else 'using auto detection'}")
            
        print("-"*80)
        
        # Detect language from the full document text
        if self.language == "auto":
            # Only detect if we have significant text
            if len(document["text"].strip()) > 500:
                detected_lang = self._detect_language(document["text"])
                print(f"Detected language for spelling correction: {detected_lang}")
            else:
                print("Document text too short for reliable language detection")
        
        # Correct the full text
        print(f"Applying spelling correction to document text ({len(document['text'])} chars)...")
        
        # Time the correction process
        import time
        start_time = time.time()
        
        # Make sure language is correctly applied before correction
        if hasattr(self, 'detected_language'):
            print(f"Using detected language: {self.detected_language}")
            
        result["text"] = self.correct_text(document["text"])
        
        correction_time = time.time() - start_time
        print(f"Full text corrected in {correction_time:.2f} seconds")
        
        # Correct each page and preserve original text
        corrected_pages = []
        original_pages = []
        
        print(f"Correcting {len(document.get('pages', []))} individual pages...")
        
        for i, page in enumerate(document.get("pages", [])):
            print(f"Applying spelling correction to page {i+1}...")
            
            # Save original page
            original_pages.append(page.copy())
            
            # Create corrected page
            page_copy = page.copy()
            original_text = page["text"]
            
            # If the page text is too short, skip correction
            if len(original_text.strip()) < 20:
                print(f"Page {i+1} has very little text, skipping correction")
                corrected_text = original_text
            else:
                corrected_text = self.correct_text(original_text)
            
            # Save both versions
            page_copy["raw_text"] = original_text
            page_copy["text"] = corrected_text
            
            # Calculate correction statistics
            if len(original_text) > 0:
                diff_count = sum(1 for a, b in zip(original_text, corrected_text) if a != b)
                diff_percentage = (diff_count / len(original_text)) * 100
                print(f"Page {i+1}: {diff_count} characters corrected ({diff_percentage:.2f}%)")
            
            corrected_pages.append(page_copy)
        
        result["pages"] = corrected_pages
        result["original_pages"] = original_pages
        
        # Add spelling correction metadata
        result["metadata"] = result.get("metadata", {})
        result["metadata"]["spelling_corrected"] = True
        result["metadata"]["spelling_language"] = self._get_language_code()
        result["metadata"]["spelling_backend"] = self.backend
        if hasattr(self, 'used_dictionary'):
            result["metadata"]["spelling_dictionary"] = self.used_dictionary
        
        # Calculate overall correction statistics
        if len(document["text"]) > 0 and len(result["text"]) > 0:
            total_diff_count = sum(1 for a, b in zip(document["text"], result["text"]) if a != b)
            total_diff_percentage = (total_diff_count / len(document["text"])) * 100
            result["metadata"]["spelling_corrections"] = total_diff_count
            result["metadata"]["spelling_correction_percentage"] = round(total_diff_percentage, 2)
            print(f"Total: {total_diff_count} characters corrected ({total_diff_percentage:.2f}%)")
        
        print(f"Spelling correction completed using {self.backend} backend in {self._get_language_code()} language")
        if hasattr(self, 'used_dictionary'):
            print(f"Using dictionary: {self.used_dictionary}")
        return result