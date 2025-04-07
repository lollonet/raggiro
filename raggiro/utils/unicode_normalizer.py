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
        
        # Other problematic characters
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
        
        # Currency symbols
        '€': 'EUR',    # Euro
        '£': 'GBP',    # Pound sterling
        '¥': 'JPY',    # Yen
        '₹': 'INR',    # Indian Rupee
        
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
        
        # Arrows and direction indicators
        '←': '<-',     # Left arrow
        '→': '->',     # Right arrow
        '↑': '^',      # Up arrow
        '↓': 'v',      # Down arrow
        '↔': '<->',    # Left-right arrow
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
            
        # Replace known problematic characters
        normalized_text = text
        for old_char, new_char in cls.CHAR_MAP.items():
            normalized_text = normalized_text.replace(old_char, new_char)
            
        # Normalize remaining Unicode characters using NFKD
        # (Compatibility Decomposition - this decomposes characters like é into e + ´)
        normalized_text = unicodedata.normalize('NFKD', normalized_text)
        
        # Remove combining characters to convert accented letters to ASCII equivalents
        # (e.g., convert 'é' to 'e' by removing the combining acute accent)
        normalized_text = ''.join(c for c in normalized_text 
                                if not unicodedata.combining(c))
        
        return normalized_text
    
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
            
        # Only replace the definitely problematic characters for display
        normalized_text = text
        for old_char, new_char in cls.CHAR_MAP.items():
            normalized_text = normalized_text.replace(old_char, new_char)
            
        return normalized_text
    
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
            List of problematic characters found
        """
        if not text:
            return []
            
        # Check for known problematic chars
        problematic = []
        for char in cls.CHAR_MAP.keys():
            if char in text:
                problematic.append(char)
        
        # Check for other non-ASCII characters
        for char in text:
            if ord(char) > 127 and char not in problematic and char not in 'àèéìòùÀÈÉÌÒÙ':
                problematic.append(char)
                
        return problematic