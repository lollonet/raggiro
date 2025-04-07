"""
Utility module for Unicode character normalization.
Provides consistent handling of special characters across the entire application.
"""

import unicodedata
import re
from typing import Union, Dict, Optional, List


class UnicodeNormalizer:
    """
    Handles normalization of Unicode characters, especially problematic ones
    like various quotes, apostrophes, and special symbols that might display
    incorrectly in some contexts.
    """
    
    # Map of problematic characters to their normalized versions
    CHAR_MAP = {
        # Quotes and apostrophes
        '"': '"',       # Left double quote
        '"': '"',       # Right double quote
        '„': '"',       # Double low-9 quotation mark
        '″': '"',       # Double prime
        '‟': '"',       # Double reversed comma quotation mark
        '«': '"',       # Left-pointing double angle quotation mark
        '»': '"',       # Right-pointing double angle quotation mark
        ''': "'",       # Left single quote
        ''': "'",       # Right single quote
        '‚': "'",       # Single low-9 quotation mark
        '‛': "'",       # Single reversed comma quotation mark
        '′': "'",       # Prime
        '‹': "'",       # Left-pointing single angle quotation mark
        '›': "'",       # Right-pointing single angle quotation mark
        
        # Dashes and hyphens
        '—': '-',       # Em dash
        '–': '-',       # En dash
        '‒': '-',       # Figure dash
        '―': '-',       # Horizontal bar
        
        # Punctuation marks and special characters
        '…': '...',     # Ellipsis
        '•': '*',       # Bullet
        '·': '.',       # Middle dot
        '®': '(R)',     # Registered trademark
        '™': '(TM)',    # Trademark
        '©': '(c)',     # Copyright
        '†': '+',       # Dagger
        '‡': '++',      # Double dagger
        '§': 'Section', # Section sign
        '¶': 'Para',    # Pilcrow sign
        
        # Question and exclamation marks - Enhanced handling
        '¿': '?',       # Inverted question mark
        '？': '?',       # Full-width question mark
        '⁇': '??',      # Double question mark
        '‽': '?!',      # Interrobang
        '⁈': '?!',      # Question exclamation mark
        '⁉': '!?',      # Exclamation question mark
        '¡': '!',       # Inverted exclamation mark
        '！': '!',       # Full-width exclamation mark
        '\u037E': '?',  # Greek question mark
        '\u061F': '?',  # Arabic question mark
        '\u2047': '??', # Double question mark
        '\u2048': '?!', # Question exclamation mark
        '\u2049': '!?', # Exclamation question mark
        '\u3008': '?',  # Left angle bracket as question mark substitute
        '\u3009': '?',  # Right angle bracket as question mark substitute
        '\u2026\u2047': '...?', # Ellipsis followed by double question
        '\u2026\u003F': '...?',   # Ellipsis followed by question mark
        
        # Currency symbols
        '€': 'EUR',    # Euro
        '£': 'GBP',    # Pound sterling
        '¥': 'JPY',    # Yen
        '₹': 'INR',    # Indian Rupee
        '₽': 'RUB',    # Ruble
        '₴': 'UAH',    # Hryvnia
        '₩': 'KRW',    # Won
        
        # Mathematical and technical symbols
        '±': '+/-',    # Plus-minus sign
        '≤': '<=',     # Less than or equal to
        '≥': '>=',     # Greater than or equal to
        '≠': '!=',     # Not equal to
        '∞': 'inf',    # Infinity
        '√': 'sqrt',   # Square root
        '∑': 'sum',    # Summation
        '∏': 'prod',   # Product
        '∂': 'd',      # Partial differential
        '∫': 'int',    # Integral
        '≈': '~=',     # Almost equal to
        '≡': '===',    # Identical to
        '≜': '=def',   # Equal by definition
        '∝': 'prop',   # Proportional to
        
        # Arrows and direction indicators
        '←': '<-',     # Left arrow
        '→': '->',     # Right arrow
        '↑': '^',      # Up arrow
        '↓': 'v',      # Down arrow
        '↔': '<->',    # Left-right arrow
        '⇒': '=>',     # Right double arrow
        '⇐': '<=',     # Left double arrow
        '⇔': '<=>',    # Left-right double arrow
        
        # Special spaces
        '\u00A0': ' ',  # Non-breaking space
        '\u200B': '',   # Zero-width space
        '\u200C': '',   # Zero-width non-joiner
        '\u200D': '',   # Zero-width joiner
        '\u2060': '',   # Word joiner
        '\u2007': ' ',  # Figure space
        '\u2008': ' ',  # Punctuation space
        '\u2009': ' ',  # Thin space
        '\u200A': ' ',  # Hair space
        '\u202F': ' ',  # Narrow no-break space
        '\u205F': ' ',  # Medium mathematical space
        '\u3000': ' ',  # Ideographic space (CJK)
        
        # UTF-8 replacement character and unknown character
        '\uFFFD': '?',  # Replacement character
        '\u2BD1': '?',  # Uncertainty sign
        '\u2370': '?',  # APL Question Mark
        '\u003F': '?',  # Standard question mark (included for consistency)
        '\u0294': '?',  # Glottal stop that can look like a question mark in some fonts
        '\u0241': '?',  # Glottal stop letter that resembles question mark
    }
    
    @classmethod
    def normalize_text(cls, text: str) -> str:
        """
        Normalize Unicode text, replacing problematic characters with ASCII equivalents.
        
        Args:
            text: The text to normalize
            
        Returns:
            The normalized text
        """
        if not text:
            return ""
        
        try:
            # Handle non-string inputs gracefully
            if not isinstance(text, str):
                text = str(text)
                
            # First, pre-normalize with NFC to ensure composed characters
            # (important for consistent replacement)
            text = unicodedata.normalize('NFC', text)
            
            # Special pre-processing for known problematic question mark sequences
            # This helps with handling question marks in various combinations
            text = re.sub(r'[\u2026\u2047]', '...?', text)  # Ellipsis + question mark
            text = re.sub(r'[\u2026]\s*[\?]', '...?', text)  # Ellipsis followed by question mark
            
            # Replace known problematic characters
            normalized_text = text
            for old_char, new_char in cls.CHAR_MAP.items():
                normalized_text = normalized_text.replace(old_char, new_char)
                
            # Additional handling for question marks and other punctuation
            # that might appear as replacement characters
            normalized_text = normalized_text.replace('\uFFFD', '?')  # Explicit replacement character
            
            # Special handling for any remaining question mark-like characters
            # that might not be covered in the CHAR_MAP
            normalized_text = re.sub(r'[\u037E\u061F\u2047\u2048\u2049\u3008\u3009\u0294\u0241]', '?', normalized_text)
            
            # Normalize remaining Unicode characters using NFKD
            # (Compatibility Decomposition - this decomposes characters like é into e + ´)
            normalized_text = unicodedata.normalize('NFKD', normalized_text)
            
            # Remove combining characters to convert accented letters to ASCII equivalents
            # (e.g., convert 'é' to 'e' by removing the combining acute accent)
            normalized_text = ''.join(c for c in normalized_text 
                                    if not unicodedata.combining(c))
            
            # Additional custom handling for question marks within combined characters
            # This ensures consistent representation of question marks
            normalized_text = re.sub(r'[^\x00-\x7F]+[\?]', '?', normalized_text)
            normalized_text = re.sub(r'[\?][^\x00-\x7F]+', '?', normalized_text)
            
            # Final safety check: replace any remaining non-ASCII with '?'
            normalized_text = ''.join(c if ord(c) < 128 else '?' for c in normalized_text)
            
            return normalized_text
        except Exception as e:
            # Failsafe: if any errors occur, try a simpler approach
            try:
                # Even in the failsafe, try to handle question marks properly
                text = re.sub(r'[^\x00-\x7F]+[\?]', '?', text)
                text = re.sub(r'[\?][^\x00-\x7F]+', '?', text)
                return ''.join(c if ord(c) < 128 else '?' for c in text)
            except:
                return str(text)
    
    @classmethod
    def clean_for_display(cls, text: str) -> str:
        """
        Simplified normalization for display purposes that preserves accented characters
        but normalizes problematic quotes, dashes, etc.
        
        Args:
            text: The text to normalize for display
            
        Returns:
            Display-friendly normalized text
        """
        if not text:
            return ""
        
        try:
            # Handle non-string inputs gracefully
            if not isinstance(text, str):
                text = str(text)
                
            # First, pre-normalize with NFC to ensure composed characters
            text = unicodedata.normalize('NFC', text)
            
            # Special pre-processing for problematic question mark sequences
            text = re.sub(r'[\u2026\u2047]', '...?', text)  # Ellipsis + question mark
            text = re.sub(r'[\u2026]\s*[\?]', '...?', text)  # Ellipsis followed by question mark
                
            # Replace known problematic characters for display
            normalized_text = text
            for old_char, new_char in cls.CHAR_MAP.items():
                normalized_text = normalized_text.replace(old_char, new_char)
            
            # Special handling for question marks - ensure proper display
            # These are specific conversions for display contexts
            normalized_text = normalized_text.replace('\uFFFD', '?')  # Replacement character
            
            # Specific handling for any remaining question mark-like characters
            normalized_text = re.sub(r'[\u037E\u061F\u2047\u2048\u2049\u3008\u3009\u0294\u0241]', '?', normalized_text)
            
            # Ensure question marks appear correctly with adjacent characters
            normalized_text = re.sub(r'[^\x00-\x7F]+[\?]', '?', normalized_text)
            normalized_text = re.sub(r'[\?][^\x00-\x7F]+', '?', normalized_text)
            
            # For display, we generally want to keep accented characters
            # but normalize any remaining problematic Unicode
            
            # Remove any zero-width characters and control characters that may affect display
            normalized_text = re.sub(r'[\u200B-\u200F\u2060-\u206F]', '', normalized_text)
            
            return normalized_text
        except Exception as e:
            # Failsafe: if normalization fails, return original or best effort
            try:
                # Even in failsafe, ensure question marks display correctly
                text = re.sub(r'[\uFFFD\u037E\u061F\u2047\u2048\u2049]', '?', text)
                return text
            except:
                return str(text)
    
    @classmethod
    def normalize_filename(cls, filename: str) -> str:
        """
        Normalize a filename to ensure it's safe for file systems.
        
        Args:
            filename: The filename to normalize
            
        Returns:
            A safe, normalized filename
        """
        if not filename:
            return "unnamed"
            
        # Normalize Unicode first using our standard method
        normalized = cls.normalize_text(filename)
        
        # Replace spaces with underscores
        normalized = normalized.replace(' ', '_')
        
        # Remove any characters that are invalid in filenames
        normalized = re.sub(r'[\\/:*?"<>|]', '', normalized)
        
        # Ensure it's not empty after cleaning
        if not normalized:
            return "unnamed"
            
        return normalized
    
    @classmethod
    def is_ascii(cls, text: str) -> bool:
        """
        Check if the text is pure ASCII.
        
        Args:
            text: The text to check
            
        Returns:
            True if the text is ASCII, False otherwise
        """
        try:
            return all(ord(c) < 128 for c in text)
        except:
            return False
    
    @classmethod
    def detect_problematic_chars(cls, text: str) -> List[str]:
        """
        Detect problematic characters in the text.
        
        Args:
            text: The text to analyze
            
        Returns:
            List of problematic characters found with category information
        """
        if not text:
            return []
        
        # Prepare result structure
        problematic = []
        
        # Pre-normalize with NFC for consistent detection
        if isinstance(text, str):
            text = unicodedata.normalize('NFC', text)
        else:
            try:
                text = str(text)
                text = unicodedata.normalize('NFC', text)
            except:
                return [('<non-string-input>', 'conversion-error')]
        
        # Special check for potential question mark issues
        question_mark_pattern = r'[\u037E\u061F\u2047\u2048\u2049\u3008\u3009\u0294\u0241\uFFFD\u2BD1\u2370]'
        question_mark_matches = re.findall(question_mark_pattern, text)
        for char in question_mark_matches:
            if char not in problematic:
                problematic.append(char)
        
        # Check for known problematic chars
        for char in cls.CHAR_MAP.keys():
            if char in text and char not in problematic:
                problematic.append(char)
        
        # Check for other non-ASCII characters
        # Add exception for common accented characters that are usually fine
        acceptable_accented = 'àèéìòùÀÈÉÌÒÙáéíóúÁÉÍÓÚäëïöüÄËÏÖÜâêîôûÂÊÎÔÛçÇñÑ'
        for char in text:
            if ord(char) > 127 and char not in problematic and char not in acceptable_accented:
                # Get Unicode category for better diagnosis
                try:
                    category = unicodedata.category(char)
                    name = unicodedata.name(char, 'UNKNOWN')
                    if 'QUESTION' in name or 'MARK' in name:
                        # Prioritize question mark-related characters
                        problematic.insert(0, char)
                    else:
                        problematic.append(char)
                except:
                    problematic.append(char)
        
        return problematic