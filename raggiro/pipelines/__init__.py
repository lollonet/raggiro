"""Pipeline implementations for different document types."""

from typing import Dict, Optional, Type, Union

from raggiro.pipelines.technical_pipeline import TechnicalPipeline

# Pipeline registry for different document categories
PIPELINE_REGISTRY = {
    "technical": TechnicalPipeline,
}

def get_pipeline_for_category(category: str, config: Optional[Dict] = None):
    """Get the appropriate pipeline class for a document category.
    
    Args:
        category: Document category
        config: Configuration dictionary
        
    Returns:
        Pipeline instance for the category or None if not found
    """
    pipeline_class = PIPELINE_REGISTRY.get(category)
    if pipeline_class:
        return pipeline_class(config)
    return None