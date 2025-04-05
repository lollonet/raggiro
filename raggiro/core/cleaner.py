"""Module for text cleaning and normalization."""

import re
from typing import Dict, List, Optional, Set, Union

class Cleaner:
    """Cleans and normalizes extracted text."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the cleaner.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Configure cleaner settings
        cleaning_config = self.config.get("cleaning", {})
        self.remove_headers_footers = cleaning_config.get("remove_headers_footers", True)
        self.normalize_whitespace = cleaning_config.get("normalize_whitespace", True)
        self.remove_special_chars = cleaning_config.get("remove_special_chars", True)
        self.min_line_length = cleaning_config.get("min_line_length", 3)
        
        # Common patterns to remove
        self.header_footer_patterns = [
            r"^\s*Page\s+\d+\s+(of|\/)\s+\d+\s*$",
            r"^\s*\d+\s*$",  # Page numbers only
            r"^[\-\=\_]{10,}$",  # Separator lines
            r"^\s*Confidential\s*$",
            r"^\s*DRAFT\s*$",
            r"^\s*Copyright\s+©.*$",
            r"^\s*All\s+rights\s+reserved\s*$",
            r"^\s*Printed\s+on:.*$",
            r"^\s*Generated\s+on:.*$",
            r"^\s*Created\s+on:.*$",
            r"^\s*Modified\s+on:.*$",
            r"^\s*Date:\s+.*$",
            r"^\s*Time:\s+.*$",
        ]
        
        # Add custom patterns from config
        custom_patterns = cleaning_config.get("header_footer_patterns", [])
        if custom_patterns:
            self.header_footer_patterns.extend(custom_patterns)
        
        # Compile regex patterns
        self.header_footer_regex = [re.compile(pattern, re.IGNORECASE) for pattern in self.header_footer_patterns]
        
        # Special characters to remove or replace
        self.special_chars_replacements = {
            # Control characters
            "\x00": "",  # Null character
            "\x01": "",  # Start of Heading
            "\x02": "",  # Start of Text
            "\x03": "",  # End of Text
            "\x04": "",  # End of Transmission
            "\x05": "",  # Enquiry
            "\x06": "",  # Acknowledge
            "\x07": "",  # Bell
            "\x08": "",  # Backspace
            "\x0B": "",  # Vertical Tab
            "\x0C": "",  # Form Feed
            "\x0E": "",  # Shift Out
            "\x0F": "",  # Shift In
            "\x10": "",  # Data Link Escape
            "\x11": "",  # Device Control 1
            "\x12": "",  # Device Control 2
            "\x13": "",  # Device Control 3
            "\x14": "",  # Device Control 4
            "\x15": "",  # Negative Acknowledge
            "\x16": "",  # Synchronous Idle
            "\x17": "",  # End of Transmission Block
            "\x18": "",  # Cancel
            "\x19": "",  # End of Medium
            "\x1A": "",  # Substitute
            "\x1B": "",  # Escape
            "\x1C": "",  # File Separator
            "\x1D": "",  # Group Separator
            "\x1E": "",  # Record Separator
            "\x1F": "",  # Unit Separator
            "\x7F": "",  # Delete
            
            # Normalize some whitespace/line break characters
            "\r": "\n",  # Carriage Return to Line Feed
            "\t": " ",   # Tab to Space
            "\f": "\n",  # Form Feed to Line Feed
            "\v": "\n",  # Vertical Tab to Line Feed
            
            # Common problematic characters
            "�": "",      # Replacement character
            "\\uf0b7": "•",  # Bullet
            "\\uf0a7": "•",  # Bullet
            "\\u2022": "•",  # Bullet
            "\\u2023": "•",  # Triangular Bullet
            "\\u25e6": "◦",  # White Bullet
            "\\u2043": "⁃",  # Hyphen Bullet
        }
    
    def clean(self, text: str) -> str:
        """Clean and normalize text.
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Replace special characters
        if self.remove_special_chars:
            for char, replacement in self.special_chars_replacements.items():
                text = text.replace(char, replacement)
        
        # Split text into lines for processing
        lines = text.split("\n")
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            
            # Skip lines that are too short
            if len(line) < self.min_line_length:
                continue
                
            # Remove headers and footers
            if self.remove_headers_footers:
                if any(pattern.match(line) for pattern in self.header_footer_regex):
                    continue
            
            cleaned_lines.append(line)
        
        # Join lines back together
        cleaned_text = "\n".join(cleaned_lines)
        
        # Normalize whitespace
        if self.normalize_whitespace:
            # Remove duplicate spaces
            cleaned_text = re.sub(r"\s+", " ", cleaned_text)
            
            # Remove duplicate line breaks
            cleaned_text = re.sub(r"\n\s*\n+", "\n\n", cleaned_text)
            
            # Trim leading/trailing whitespace
            cleaned_text = cleaned_text.strip()
        
        return cleaned_text
    
    def clean_document(self, document: Dict) -> Dict:
        """Clean all text in a document.
        
        Args:
            document: Document dictionary with text and pages
            
        Returns:
            Document with cleaned text
        """
        result = document.copy()
        
        # Clean the full text
        result["text"] = self.clean(document["text"])
        
        # Clean each page
        cleaned_pages = []
        for page in document.get("pages", []):
            page_copy = page.copy()
            page_copy["text"] = self.clean(page["text"])
            cleaned_pages.append(page_copy)
        
        result["pages"] = cleaned_pages
        
        return result
    
    def remove_duplicates(self, text: str) -> str:
        """Remove duplicate paragraphs from text.
        
        Args:
            text: Text to process
            
        Returns:
            Text with duplicate paragraphs removed
        """
        if not text:
            return ""
        
        paragraphs = text.split("\n\n")
        unique_paragraphs = []
        seen = set()
        
        for paragraph in paragraphs:
            # Skip very short paragraphs
            if len(paragraph) < 15:
                unique_paragraphs.append(paragraph)
                continue
                
            # Normalize for comparison
            normalized = re.sub(r"\s+", " ", paragraph.strip().lower())
            
            if normalized not in seen:
                seen.add(normalized)
                unique_paragraphs.append(paragraph)
        
        return "\n\n".join(unique_paragraphs)
    
    def clean_specific_patterns(self, text: str, patterns: List[str], replacements: List[str]) -> str:
        """Replace specific patterns in text.
        
        Args:
            text: Text to process
            patterns: List of regex patterns
            replacements: List of replacements (must match patterns length)
            
        Returns:
            Text with patterns replaced
        """
        if not text or not patterns:
            return text
            
        if len(patterns) != len(replacements):
            raise ValueError("Patterns and replacements must have the same length")
            
        for pattern, replacement in zip(patterns, replacements):
            text = re.sub(pattern, replacement, text)
            
        return text