"""
Module containing lists of words that should be skipped during spelling correction.
These include technical terms, brand names, and foreign language terms that might
be incorrectly "corrected" by the spelling checker.
"""

# Full list of terms that should not be spell-checked or corrected
TECHNICAL_TERMS = {
    # Termini musicali e artistici
    "jazz", "blues", "rock", "swing", "folk", "bebop", "soul", "funk", "hip", "hop", "rap", 
    "reggae", "punk", "metal", "techno", "disco", "pop", "band", "jam", "groove", 
    "piano", "bass", "drum", "sax", "beat", "chord", "note", "scale", "tempo", "key",
    "major", "minor", "pitch", "rhythm", "tune", "staff", "score", "music", "solo", "duo",
    "trio", "quartet", "quintet", "sextet", "backing", "lead", "sharp", "flat", 
    
    # Tecnologia
    "software", "hardware", "app", "byte", "web", "cloud", "server", "router", "database", 
    "python", "java", "html", "css", "xml", "json", "api", "http", "https", "url", "domain",
    "code", "script", "laptop", "desktop", "mobile", "tablet", "smartphone", "email", "wifi",
    
    # Termini inglesi comuni
    "meeting", "team", "business", "manager", "leader", "design", "report", "staff", "board",
    "file", "test", "call", "link", "post", "blog", "week", "weekend", "day", "night", "time",
    "road", "street", "mail", "house", "store", "shop", "office", "back", "front", "side",
    "best", "good", "bad", "top", "down", "open", "close", "start", "stop", "free", "login",
    
    # Marchi e piattaforme
    "google", "microsoft", "apple", "amazon", "facebook", "twitter", "instagram", "linkedin",
    "youtube", "spotify", "netflix", "zoom", "teams", "whatsapp", "telegram", "signal", "slack", 
    "gmail", "outlook",
    
    # Termini fotografici e informatici
    "jpeg", "jpg", "png", "gif", "raw", "hdr", "iso", "frame", "pixel", "resolution", "format",
    "pdf", "doc", "docx", "ppt", "xlsx", "zip", "rar", "backup", "sync", "update", "install"
}

# Add capitalized and uppercase versions of all terms
SKIP_WORDS = set()
for word in TECHNICAL_TERMS:
    SKIP_WORDS.add(word)
    SKIP_WORDS.add(word.capitalize())
    SKIP_WORDS.add(word.upper())