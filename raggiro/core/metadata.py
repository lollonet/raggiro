"""Module for metadata extraction and enrichment."""

import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import dateparser
import langdetect
from bs4 import BeautifulSoup

class MetadataExtractor:
    """Extracts and enriches metadata from documents."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the metadata extractor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Date extraction patterns
        self.date_patterns = [
            r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2},?\s+\d{4}\b",
            r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b",
            r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b",
            r"\bDate:?\s*\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}\b",
            r"\bDate:?\s*[A-Za-z]+\s+\d{1,2},?\s+\d{4}\b",
        ]
        
        # Date patterns as compiled regex
        self.date_regex = [re.compile(pattern, re.IGNORECASE) for pattern in self.date_patterns]
        
        # Title extraction patterns
        self.title_patterns = [
            r"^(?:TITLE|Title|SUBJECT|Subject):\s*(.*?)$",
            r"^(?:REPORT|Report)\s+(?:TITLE|Title):\s*(.*?)$",
            r"^(?:DOCUMENT|Document)\s+(?:TITLE|Title):\s*(.*?)$",
            # Book title patterns
            r"^\s*([A-Z][A-Za-z0-9\s'\":]+)(?:\n\s*|\s+)(?:by|di|BY|DI)\s+([A-Z][A-Za-z\s.]+)",
            # Standalone large title at beginning of document (often books)
            r"^\s*([A-Z][A-Z0-9\s'\":]{10,50})\s*$",
        ]
        
        # Title patterns as compiled regex
        self.title_regex = [re.compile(pattern, re.MULTILINE) for pattern in self.title_patterns]
        
        # Author extraction patterns
        self.author_patterns = [
            r"^(?:AUTHOR|Author|BY|By|PREPARED BY|Prepared by):\s*(.*?)$",
            r"^(?:AUTHORS|Authors):\s*(.*?)$",
            r"^(?:WRITTEN BY|Written by):\s*(.*?)$",
            # Common book author formats
            r"(?:by|di|BY|DI)\s+([A-Z][A-Za-z\s.]+)",
            r"(?:^|\n)\s*([A-Z][A-Za-z\s.-]+)(?=\n)",
        ]
        
        # Author patterns as compiled regex
        self.author_regex = [re.compile(pattern, re.MULTILINE) for pattern in self.author_patterns]
        
        # Common topic/category keywords with their categories
        self.topic_keywords = {
            "technical": ["technical", "specification", "manual", "guide", "documentation", "user guide", "reference"],
            "legal": ["legal", "contract", "agreement", "terms", "conditions", "policy", "regulation", "law", "compliance"],
            "financial": ["financial", "budget", "cost", "revenue", "expense", "profit", "loss", "balance", "account", "finance", "banking"],
            "academic": ["research", "study", "analysis", "thesis", "dissertation", "journal", "paper", "academic", "scientific"],
            "business": ["business", "marketing", "sales", "customer", "product", "service", "strategy", "management", "operation"],
            "report": ["report", "summary", "overview", "analysis", "review", "assessment", "evaluation"],
            "presentation": ["presentation", "slide", "deck", "overview", "briefing"],
            "book": ["book", "novel", "chapter", "author", "publisher", "publication", "copyright", "edition", "preface", "foreword"],
            "music": ["music", "musician", "jazz", "piano", "improvisation", "performance", "mastery", "practice", "technique", "composition"],
        }
        
        # Publisher extraction patterns
        self.publisher_patterns = [
            r"(?:Published by|Publisher|PUBLISHER):\s*(.*?)(?:\n|$)",
            r"(?:Copyright|©)\s+[\d]{4}[^\n]*?by\s+(.*?)(?:\n|$)",
            r"((?:\w+\s+)(?:Press|Publications|Publishing|Books|Media))",
        ]
        
        # Publisher patterns as compiled regex
        self.publisher_regex = [re.compile(pattern, re.IGNORECASE | re.MULTILINE) for pattern in self.publisher_patterns]
    
    def extract(self, document: Dict, file_metadata: Dict) -> Dict:
        """Extract and enrich metadata from a document.
        
        Args:
            document: Document dictionary with text and extractions
            file_metadata: Basic file metadata from file_handler
            
        Returns:
            Enriched metadata dictionary
        """
        result = {}
        
        # Start with existing metadata
        result.update(document.get("metadata", {}))
        
        # Add file metadata
        result["file"] = file_metadata
        
        # Extract text content to work with
        text = document["text"]
        first_chunk = text[:10000]  # First 10000 chars for metadata extraction (increased from 5000)
        
        # Add file-based metadata
        result["filename"] = file_metadata.get("filename", "")
        result["file_path"] = file_metadata.get("path", "")
        result["file_type"] = file_metadata.get("extension", "").lstrip(".")
        
        # Extract other metadata
        extracted_title = self._extract_title(first_chunk, document)
        if extracted_title:
            result["title"] = extracted_title
            
        extracted_author = self._extract_author(first_chunk, document)
        if extracted_author:
            result["author"] = extracted_author
            
        extracted_date = self._extract_date(first_chunk, document)
        if extracted_date:
            result["date"] = extracted_date
            
        detected_language = self._detect_language(text)
        if detected_language:
            result["language"] = detected_language
            
        # Extract publisher (new)
        extracted_publisher = self._extract_publisher(first_chunk)
        if extracted_publisher:
            result["publisher"] = extracted_publisher
            
        detected_topics = self._detect_topic(text, result)
        if detected_topics:
            result["topics"] = detected_topics
        
        # Add content statistics
        result["word_count"] = len(text.split())
        result["char_count"] = len(text)
        
        if "pages" in document:
            result["page_count"] = len(document["pages"])
        
        if "chunks" in document:
            result["chunk_count"] = len(document["chunks"])
        
        return result
    
    def _extract_title(self, text: str, document: Dict) -> Optional[str]:
        """Extract document title from text or metadata.
        
        Args:
            text: Text to extract from
            document: Document dictionary with metadata
            
        Returns:
            Extracted title or None
        """
        # First check existing metadata
        existing_metadata = document.get("metadata", {})
        if existing_metadata.get("title"):
            return existing_metadata["title"]
        
        # Check for ALL CAPS title at the beginning (common in books)
        first_lines = text.split("\n")[:10]  # First 10 lines
        for line in first_lines:
            line = line.strip()
            # Look for all caps title of reasonable length
            if line.isupper() and 10 <= len(line) <= 100 and not any(word in line.lower() for word in ["page", "copyright", "confidential"]):
                return line
                
        # Look for book title with author format: "TITLE by AUTHOR"
        title_author_match = re.search(r"^\s*([A-Z][A-Za-z0-9\s'\":]+)(?:\n\s*|\s+)(?:by|di|BY|DI)\s+([A-Z][A-Za-z\s.]+)", text, re.MULTILINE)
        if title_author_match:
            title = title_author_match.group(1).strip()
            if len(title) > 5 and len(title) < 200:
                return title
        
        # Then try pattern extraction
        for pattern in self.title_regex:
            matches = pattern.findall(text)
            if matches:
                title = matches[0].strip()
                if len(title) > 5 and len(title) < 200:  # Reasonable title length
                    return title
        
        # If no patterns match, try filename-based title for books
        filename = document.get("metadata", {}).get("file", {}).get("filename", "")
        if filename and "." in filename:
            # Remove file extension and replace underscores/hyphens with spaces
            potential_title = re.sub(r'\.[^.]+$', '', filename)
            potential_title = re.sub(r'[_-]', ' ', potential_title)
            # Capitalize properly
            potential_title = " ".join(word.capitalize() for word in potential_title.split())
            if len(potential_title) > 5:
                return potential_title$', '', filename)
            potential_title = re.sub(r'[_-]', ' ', potential_title)
            # Capitalize properly
            potential_title = " ".join(word.capitalize() for word in potential_title.split())
            if len(potential_title) > 5:
                return potential_title
        
        # If no patterns match, try the first non-empty significant line
        lines = text.split("\n")
        for line in lines[:15]:  # Check only first 15 lines
            line = line.strip()
            if line and len(line) > 5 and len(line) < 200:
                if not any(re.search(r"page|copyright|confidential|draft|author|date", line, re.IGNORECASE)):
                    return line
        
        # If html, try to get title from the h1 or header tags
        if document.get("metadata", {}).get("document_type") == "html":
            try:
                soup = BeautifulSoup(text, "html.parser")
                h1 = soup.find("h1")
                if h1:
                    return h1.get_text().strip()
                
                header = soup.find("header")
                if header:
                    h_tags = header.find_all(["h1", "h2", "h3"])
                    if h_tags:
                        return h_tags[0].get_text().strip()
            except:
                pass
        
        return None
    
    def _extract_author(self, text: str, document: Dict) -> Optional[str]:
        """Extract document author from text or metadata.
        
        Args:
            text: Text to extract from
            document: Document dictionary with metadata
            
        Returns:
            Extracted author or None
        """
        # First check existing metadata
        existing_metadata = document.get("metadata", {})
        if existing_metadata.get("author"):
            return existing_metadata["author"]
        
        # Then try pattern extraction
        for pattern in self.author_regex:
            matches = pattern.findall(text)
            if matches:
                author = matches[0].strip()
                if len(author) > 2 and len(author) < 100:  # Reasonable author name length
                    return author
        
        # Special case for common book format "TITLE by AUTHOR"
        title_author_match = re.search(r"^\s*([A-Z][A-Za-z0-9\s'\":]+)(?:\n\s*|\s+)(?:by|di|BY|DI)\s+([A-Z][A-Za-z\s.]+)", text, re.MULTILINE)
        if title_author_match:
            author = title_author_match.group(2).strip()
            if len(author) > 2 and len(author) < 100:
                return author
        
        # Try to find an author line
        lines = text.split("\n")
        for line in lines[:30]:  # Check first 30 lines (increased from 20)
            line = line.strip().lower()
            if line.startswith("author:") or line.startswith("by:") or line.startswith("prepared by:"):
                parts = line.split(":", 1)
                if len(parts) > 1:
                    author = parts[1].strip()
                    if len(author) > 2:
                        return author.title()  # Convert to title case
            
            # Check for standalone author name after title
            if line.startswith("by ") and len(line) > 3:
                author = line[3:].strip()
                if len(author) > 2:
                    return author.title()
        
        return None
        
    def _extract_publisher(self, text: str) -> Optional[str]:
        """Extract publisher information from text.
        
        Args:
            text: Text to extract from
            
        Returns:
            Extracted publisher or None
        """
        # Try pattern extraction
        for pattern in self.publisher_regex:
            matches = pattern.findall(text)
            if matches:
                publisher = matches[0].strip()
                if len(publisher) > 2 and len(publisher) < 100:  # Reasonable publisher length
                    return publisher
        
        # Look for copyright lines which often contain publisher info
        copyright_match = re.search(r"(?:©|Copyright)\s+(?:\d{4})?[^\n]*?((?:\w+\s+){1,4}(?:Press|Publications|Publishing|Books|Media|Group))", text, re.IGNORECASE)
        if copyright_match:
            publisher = copyright_match.group(1).strip()
            if len(publisher) > 2:
                return publisher
                
        # Try to find publisher in first few lines
        lines = text.split("\n")
        for line in lines[:50]:  # Check first 50 lines
            line = line.strip()
            # Look for publishing house patterns
            if re.search(r"(?:Press|Publications|Publishing|Books|Media|Group)$", line, re.IGNORECASE):
                if 2 < len(line) < 100:
                    return line
                    
            # Check explicit publisher declarations
            if re.match(r"(?:Published by|Publisher)[:\s]", line, re.IGNORECASE):
                publisher = re.sub(r"(?:Published by|Publisher)[:\s]", "", line, flags=re.IGNORECASE).strip()
                if len(publisher) > 2:
                    return publisher
        
        return None
    
    def _extract_date(self, text: str, document: Dict) -> Optional[str]:
        """Extract document date from text or metadata.
        
        Args:
            text: Text to extract from
            document: Document dictionary with metadata
            
        Returns:
            Extracted date (ISO format) or None
        """
        # First check existing metadata
        existing_metadata = document.get("metadata", {})
        
        # Check common metadata fields
        date_fields = ["created", "creation_date", "creationDate", "modified", "modification_date", "modDate"]
        for field in date_fields:
            if field in existing_metadata and existing_metadata[field]:
                # Try to parse the date string
                try:
                    parsed_date = dateparser.parse(existing_metadata[field])
                    if parsed_date:
                        return parsed_date.isoformat().split("T")[0]  # YYYY-MM-DD
                except:
                    pass
        
        # Try to extract date from text using regex patterns
        for pattern in self.date_regex:
            matches = pattern.findall(text)
            if matches:
                # Try to parse each match
                for match in matches:
                    try:
                        parsed_date = dateparser.parse(match)
                        if parsed_date:
                            # Check if date is reasonable (not future, not too old)
                            now = datetime.now()
                            if parsed_date.year > 1900 and parsed_date <= now:
                                return parsed_date.isoformat().split("T")[0]  # YYYY-MM-DD
                    except:
                        continue
        
        # Look for date lines
        lines = text.split("\n")
        for line in lines[:20]:  # Check first 20 lines
            line = line.strip().lower()
            if "date:" in line:
                try:
                    # Extract the date part
                    date_part = line.split("date:", 1)[1].strip()
                    parsed_date = dateparser.parse(date_part)
                    if parsed_date:
                        return parsed_date.isoformat().split("T")[0]  # YYYY-MM-DD
                except:
                    pass
        
        # If we have file metadata, use modification date as a fallback
        if "file" in existing_metadata:
            file_metadata = existing_metadata["file"]
            if "modified_at" in file_metadata:
                try:
                    modified_at = datetime.fromtimestamp(file_metadata["modified_at"])
                    return modified_at.isoformat().split("T")[0]  # YYYY-MM-DD
                except:
                    pass
        
        return None
    
    def _detect_language(self, text: str) -> Optional[str]:
        """Detect the language of the text.
        
        Args:
            text: Text to analyze
            
        Returns:
            ISO language code or None
        """
        if not text or len(text.strip()) < 50:
            return None
            
        try:
            # Use a sample of the text for faster detection
            sample = text[:5000]
            lang = langdetect.detect(sample)
            return lang
        except:
            return None
    
    def _detect_topic(self, text: str, metadata: Dict) -> List[str]:
        """Detect topics/categories of the document.
        
        Args:
            text: Text to analyze
            metadata: Existing metadata
            
        Returns:
            List of detected topics
        """
        topics = []
        
        # First look at the title
        title = metadata.get("title", "").lower()
        
        for topic, keywords in self.topic_keywords.items():
            for keyword in keywords:
                if keyword in title:
                    topics.append(topic)
                    break
        
        # Then look at the first part of the document
        sample = text[:10000].lower()
        
        for topic, keywords in self.topic_keywords.items():
            if topic not in topics:  # Only check topics we haven't already identified
                for keyword in keywords:
                    if keyword in sample and sample.count(keyword) > 2:
                        topics.append(topic)
                        break
        
        return topics