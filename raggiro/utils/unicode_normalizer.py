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
        '\u201C': '"',       # Left double quote
        '\u201D': '"',       # Right double quote
        '\u201E': '"',       # Double low-9 quotation mark
        '\u2033': '"',       # Double prime
        '\u201F': '"',       # Double reversed comma quotation mark
        '\u00AB': '"',       # Left-pointing double angle quotation mark
        '\u00BB': '"',       # Right-pointing double angle quotation mark
        '\u2018': "'",       # Left single quote
        '\u2019': "'",       # Right single quote
        '\u201A': "'",       # Single low-9 quotation mark
        '\u201B': "'",       # Single reversed comma quotation mark
        '\u2032': "'",       # Prime
        '\u2039': "'",       # Left-pointing single angle quotation mark
        '\u203A': "'",       # Right-pointing single angle quotation mark
        
        # Dashes and hyphens
        '\u2014': '-',       # Em dash
        '\u2013': '-',       # En dash
        '\u2012': '-',       # Figure dash
        '\u2015': '-',       # Horizontal bar
        
        # Punctuation marks and special characters
        '\u2026': '...',     # Ellipsis
        '\u2022': '*',       # Bullet
        '\u00B7': '.',       # Middle dot
        '\u00AE': '(R)',     # Registered trademark
        '\u2122': '(TM)',    # Trademark
        '\u00A9': '(c)',     # Copyright
        '\u2020': '+',       # Dagger
        '\u2021': '++',      # Double dagger
        '\u00A7': 'Section', # Section sign
        '\u00B6': 'Para',    # Pilcrow sign
        
        # Question and exclamation marks - Enhanced handling
        '\u00BF': '?',       # Inverted question mark
        '\uFF1F': '?',       # Full-width question mark
        '\u2047': '??',      # Double question mark
        '\u203D': '?!',      # Interrobang
        '\u2048': '?!',      # Question exclamation mark
        '\u2049': '!?',      # Exclamation question mark
        '\u00A1': '!',       # Inverted exclamation mark
        '\uFF01': '!',       # Full-width exclamation mark
        '\u037E': '?',       # Greek question mark
        '\u061F': '?',       # Arabic question mark
        '\u2370': '?',       # APL Question Mark
        '\u0294': '?',       # Glottal stop that can look like a question mark in some fonts
        '\u0241': '?',       # Glottal stop letter that resembles question mark
        '\u003F': '?',       # Standard question mark (included for consistency)
        
        # Currency symbols
        '\u20AC': 'EUR',    # Euro
        '\u00A3': 'GBP',    # Pound sterling
        '\u00A5': 'JPY',    # Yen
        '\u20B9': 'INR',    # Indian Rupee
        '\u20BD': 'RUB',    # Ruble
        '\u20B4': 'UAH',    # Hryvnia
        '\u20A9': 'KRW',    # Won
        
        # Mathematical and technical symbols
        '\u00B1': '+/-',    # Plus-minus sign
        '\u2264': '<=',     # Less than or equal to
        '\u2265': '>=',     # Greater than or equal to
        '\u2260': '!=',     # Not equal to
        '\u221E': 'inf',    # Infinity
        '\u221A': 'sqrt',   # Square root
        '\u2211': 'sum',    # Summation
        '\u220F': 'prod',   # Product
        '\u2202': 'd',      # Partial differential
        '\u222B': 'int',    # Integral
        '\u2248': '~=',     # Almost equal to
        '\u2261': '===',    # Identical to
        '\u225C': '=def',   # Equal by definition
        '\u221D': 'prop',   # Proportional to
        
        # Arrows and direction indicators
        '\u2190': '<-',     # Left arrow
        '\u2192': '->',     # Right arrow
        '\u2191': '^',      # Up arrow
        '\u2193': 'v',      # Down arrow
        '\u2194': '<->',    # Left-right arrow
        '\u21D2': '=>',     # Right double arrow
        '\u21D0': '<=',     # Left double arrow
        '\u21D4': '<=>',    # Left-right double arrow
        
        # Special spaces
        '\u00A0': ' ',      # Non-breaking space
        '\u200B': '',       # Zero-width space
        '\u200C': '',       # Zero-width non-joiner
        '\u200D': '',       # Zero-width joiner
        '\u2060': '',       # Word joiner
        '\u2007': ' ',      # Figure space
        '\u2008': ' ',      # Punctuation space
        '\u2009': ' ',      # Thin space
        '\u200A': ' ',      # Hair space
        '\u202F': ' ',      # Narrow no-break space
        '\u205F': ' ',      # Medium mathematical space
        '\u3000': ' ',      # Ideographic space (CJK)
        
        # UTF-8 replacement character and unknown character
        '\uFFFD': '?',      # Replacement character
        '\u2BD1': '?',      # Uncertainty sign
    }
    
    # Additional Unicode blocks to check for question mark sequences
    QUESTION_MARK_PATTERNS = [
        # Ellipsis + question mark combinations
        ['\u2026', '\u003F'],  # HORIZONTAL ELLIPSIS + QUESTION MARK
        ['\u2026', '\u003F', '\u003F'],  # HORIZONTAL ELLIPSIS + QUESTION MARK + QUESTION MARK
        
        # Other problematic combinations
        ['\uFFFD', '\u003F'],  # REPLACEMENT CHARACTER + QUESTION MARK
        ['\u003F', '\uFFFD'],  # QUESTION MARK + REPLACEMENT CHARACTER
    ]
    
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
            # Replace known problematic character combinations
            normalized_text = text
            for pattern in cls.QUESTION_MARK_PATTERNS:
                pattern_str = ''.join(pattern)
                replacement = '?' * len([c for c in pattern if c in ['\u003F', '\uFFFD', '\u037E', '\u061F']])
                if not replacement:
                    replacement = '?'  # Default to single question mark
                normalized_text = normalized_text.replace(pattern_str, replacement)
            
            # Replace ellipsis followed by question mark with the standard sequence
            normalized_text = re.sub(r'\u2026\s*\?', '...?', normalized_text)
            
            # Replace known problematic characters
            for old_char, new_char in cls.CHAR_MAP.items():
                normalized_text = normalized_text.replace(old_char, new_char)
                
            # Additional handling for question marks and other punctuation
            # that might appear as replacement characters
            normalized_text = normalized_text.replace('\uFFFD', '?')  # Explicit replacement character
            
            # Special handling for any remaining question mark-like characters
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
            for pattern in cls.QUESTION_MARK_PATTERNS:
                pattern_str = ''.join(pattern)
                replacement = '?' * len([c for c in pattern if c in ['\u003F', '\uFFFD', '\u037E', '\u061F']])
                if not replacement:
                    replacement = '?'  # Default to single question mark
                text = text.replace(pattern_str, replacement)
                
            # Also handle ellipsis + question mark combinations
            text = re.sub(r'\u2026\s*\?', '...?', text)
            
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
        
    @classmethod
    def escape_unicode(cls, text: str) -> str:
        """
        Escape all non-ASCII characters as Python Unicode escape sequences.
        This is useful for debugging Unicode issues.
        
        Args:
            text: The text to escape
            
        Returns:
            Text with all non-ASCII chars as escape sequences
        """
        if not text:
            return ""
        
        result = ""
        for char in text:
            if ord(char) < 128:
                result += char
            else:
                # Convert to Python unicode escape sequence
                result += f"\\u{ord(char):04x}"
                
        return result
        
    @classmethod
    def unescape_unicode(cls, text: str) -> str:
        """
        Unescape Unicode escape sequences like \\uXXXX back to characters.
        
        Args:
            text: The text with escape sequences
            
        Returns:
            Text with escape sequences converted to characters
        """
        if not text:
            return ""
            
        try:
            # This handles \uXXXX escape sequences
            return text.encode('utf-8').decode('unicode_escape')
        except Exception as e:
            # If there's any error, return the original text
            return text