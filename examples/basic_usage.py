"""Example of using Raggiro for document processing and RAG."""

import os
import sys
from pathlib import Path

# Add the parent directory to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from raggiro.processor import DocumentProcessor
from raggiro.rag.indexer import VectorIndexer
from raggiro.rag.pipeline import RagPipeline
from raggiro.utils.config import load_config

def process_documents(input_path, output_path):
    """Process documents from input_path to output_path."""
    print(f"Processing documents from {input_path} to {output_path}")
    
    # Load configuration
    config = load_config()
    
    # Create processor
    processor = DocumentProcessor(config)
    
    # Process documents
    if Path(input_path).is_file():
        result = processor.process_file(input_path, output_path)
        
        if result["success"]:
            print(f"Successfully processed {input_path}")
        else:
            print(f"Failed to process {input_path}: {result.get('error', 'Unknown error')}")
    else:
        result = processor.process_directory(input_path, output_path, recursive=True)
        
        if result["success"]:
            summary = result["summary"]
            print(f"Processed {summary['total_files']} files")
            print(f"Successfully processed: {summary['successful_files']} files ({summary['success_rate']}%)")
            print(f"Failed: {summary['failed_files']} files")
        else:
            print(f"Failed to process directory: {result.get('error', 'Unknown error')}")
    
    return result

def index_documents(input_path, index_path):
    """Index processed documents from input_path to index_path."""
    print(f"Indexing documents from {input_path} to {index_path}")
    
    # Load configuration
    config = load_config()
    
    # Create indexer
    indexer = VectorIndexer(config)
    
    # Index documents
    result = indexer.index_directory(input_path)
    
    if result["success"]:
        summary = result["summary"]
        print(f"Indexed {summary['total_files']} files")
        print(f"Successfully indexed: {summary['successful_files']} files ({summary['success_rate']}%)")
        print(f"Failed: {summary['failed_files']} files")
        print(f"Total chunks indexed: {summary['total_chunks_indexed']}")
        
        # Save the index
        save_result = indexer.save_index(index_path)
        
        if save_result["success"]:
            print(f"Saved index to {save_result['index_path']}")
            print(f"Saved lookup to {save_result['lookup_path']}")
        else:
            print(f"Failed to save index: {save_result.get('error', 'Unknown error')}")
    else:
        print(f"Failed to index documents: {result.get('error', 'Unknown error')}")
    
    return result

def query_rag(index_path, query):
    """Query the RAG pipeline."""
    print(f"Querying RAG pipeline: {query}")
    
    # Load configuration
    config = load_config()
    
    # Create pipeline
    pipeline = RagPipeline(config)
    
    # Load index
    load_result = pipeline.retriever.load_index(index_path)
    
    if not load_result["success"]:
        print(f"Failed to load index: {load_result.get('error', 'Unknown error')}")
        return None
    
    # Query the pipeline
    result = pipeline.query(query)
    
    if result["success"]:
        print("\nQuery:", result["original_query"])
        
        if "rewritten_query" in result:
            print("Rewritten Query:", result["rewritten_query"])
        
        print("\nResponse:", result["response"])
        print(f"\nUsed {result['chunks_used']} chunks for this response.")
    else:
        print(f"Failed to query RAG pipeline: {result.get('error', 'Unknown error')}")
    
    return result

def main():
    """Main function."""
    # Create example directories
    input_dir = "examples/documents"
    output_dir = "examples/output"
    index_dir = "examples/index"
    
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(index_dir, exist_ok=True)
    
    # Check if example files exist
    if not list(Path(input_dir).glob("*")):
        print(f"No example files found in {input_dir}")
        print("Please add some document files (PDF, DOCX, etc.) to the input directory")
        return
    
    # Process and index documents
    process_result = process_documents(input_dir, output_dir)
    
    if process_result["success"]:
        index_result = index_documents(output_dir, index_dir)
        
        if index_result["success"]:
            # Example queries
            queries = [
                "What is the main topic of these documents?",
                "Summarize the key points from the documents.",
                "What recommendations are made in these documents?",
            ]
            
            for query in queries:
                query_rag(index_dir, query)
                print("\n" + "-" * 80 + "\n")

if __name__ == "__main__":
    main()