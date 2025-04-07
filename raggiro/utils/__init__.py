"""Utility functions and helpers."""
import functools
import logging
import traceback
from typing import Dict, Any, Callable

# Set up logger
logger = logging.getLogger("raggiro.utils")

def create_base_result(extraction_method=None, success=False):
    """Create a standard base result dictionary.
    
    Args:
        extraction_method: The method used for extraction
        success: Whether the operation was successful
        
    Returns:
        Dictionary with standard result structure
    """
    return {
        "text": "",
        "pages": [],
        "metadata": {},
        "extraction_method": extraction_method,
        "has_text_layer": False,
        "success": success,
        "error": None,
    }

def log_processing_phase(logger, document, status, component, phase, phase_number, total_phases, processing_time_ms, error=None):
    """Log a processing phase with standardized format.
    
    Args:
        logger: The logger instance to use
        document: Document being processed
        status: 'success' or 'failure'
        component: Component name (e.g., 'extractor')
        phase: Phase name (e.g., 'Text extraction')
        phase_number: Current phase number
        total_phases: Total number of phases
        processing_time_ms: Processing time in milliseconds
        error: Optional error message for failures
    """
    if not logger:
        return
        
    if status == "failure" and error:
        logger.log_file_processing(
            document, status,
            component=component,
            phase=phase,
            phase_number=phase_number,
            total_phases=total_phases,
            processing_time_ms=processing_time_ms,
            error=error
        )
    else:
        logger.log_file_processing(
            document, status,
            component=component,
            phase=phase,
            phase_number=phase_number,
            total_phases=total_phases,
            processing_time_ms=processing_time_ms
        )
    
def safe_extraction(result_field: str = "error"):
    """Decorator for safely handling exceptions in extraction functions.
    
    Args:
        result_field: Field name in the result dictionary to store error message
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {str(e)}", exc_info=True)
                # Get default result from the first arg's base_result_method if available
                if args and hasattr(args[0], 'base_result_method'):
                    result = args[0].base_result_method()
                else:
                    result = create_base_result()
                    
                result[result_field] = str(e)
                return result
        return wrapper
    return decorator

def with_memory_cleanup(gc_collect: bool = True):
    """Decorator for functions that need explicit memory cleanup.
    
    Args:
        gc_collect: Whether to call garbage collection after function execution
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            finally:
                # Delete the first argument if it's likely a large object
                if args and hasattr(args[0], "__dict__") and not isinstance(args[0], type):
                    for key in list(args[0].__dict__.keys()):
                        if key.startswith("_temp"):
                            delattr(args[0], key)
                
                # Run garbage collection if specified
                if gc_collect:
                    import gc
                    gc.collect()
        return wrapper
    return decorator
    
def chunk_summary_generator(text, max_length=150):
    """Generate a summary for text chunks.
    This is a simple utility to create consistent summaries for text chunks.
    
    Args:
        text: The text to summarize
        max_length: Maximum summary length
        
    Returns:
        A simple summary based on the first few sentences
    """
    if not text:
        return ""
        
    # Get first few sentences up to max_length
    sentences = text.split('.')
    summary = ""
    for sentence in sentences:
        if len(summary) + len(sentence) + 1 <= max_length:
            summary += sentence.strip() + "."
        else:
            break
            
    # Ensure proper ending
    if not summary.endswith('.'):
        summary += '.'
        
    return summary.strip()