#!/usr/bin/env python3
"""
Test per la valutazione del chunking semantico
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Aggiungi il percorso principale al Python path
current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent.parent
sys.path.insert(0, str(root_dir))

from raggiro.processor import DocumentProcessor
from raggiro.rag.indexer import VectorIndexer
from raggiro.rag.pipeline import RagPipeline
from raggiro.utils.config import load_config
import toml

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Test del chunking semantico')
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Path al documento da elaborare (PDF, DOCX, ecc.)')
    parser.add_argument('--output', '-o', type=str, 
                        default=str(Path.cwd() / 'test_output'),
                        help='Directory di output per i documenti elaborati')
    parser.add_argument('--index', type=str, 
                        default=str(Path.cwd() / 'test_index'),
                        help='Directory per l\'indice vettoriale')
    parser.add_argument('--queries', '-q', type=str, nargs='+',
                        help='Query da testare (opzionale)')
    parser.add_argument('--top-k', type=int, default=3,
                        help='Numero di chunk da recuperare per query (default: 3)')
    return parser.parse_args()

def main():
    """Main function."""
    # Parse arguments
    args = parse_arguments()
    
    # Create output directories
    output_dir = Path(args.output)
    index_dir = Path(args.index)
    output_dir.mkdir(parents=True, exist_ok=True)
    index_dir.mkdir(parents=True, exist_ok=True)
    
    # Load configuration with proper path
    config_path = root_dir / "config" / "config.toml"
    print(f"Loading config from: {config_path}")
    
    # Skip TOML direct loading due to variable interpolation issues
    print("TOML direct loading skipped - TOML file uses variable interpolation")
    
    # Create a hardcoded configuration instead
    config = {
        "llm": {
            "provider": "ollama",
            "ollama_base_url": "http://192.168.63.204:11434",
            "ollama_timeout": 30
        },
        "rewriting": {
            "enabled": True,
            "llm_type": "ollama",
            "ollama_model": "llama3",
            "temperature": 0.1,
            "max_tokens": 200,
            "ollama_base_url": "http://192.168.63.204:11434"
        },
        "generation": {
            "llm_type": "ollama",
            "ollama_model": "mistral",
            "temperature": 0.7,
            "max_tokens": 1000,
            "ollama_base_url": "http://192.168.63.204:11434"
        },
        "segmentation": {
            "semantic_chunking": True,
            "chunking_strategy": "hybrid"
        }
    }
    print(f"Using hardcoded config with Ollama URL: {config['llm']['ollama_base_url']}")
    print(f"Strategia di chunking attuale: {config.get('segmentation', {}).get('chunking_strategy', 'size')}")
    
    # Process the document
    print(f"\n=== Elaborazione documento: {args.input} ===")
    processor = DocumentProcessor(config)
    
    # Check if input is a JSON file (already processed document)
    input_path = Path(args.input)
    if input_path.suffix.lower() == '.json':
        try:
            # Try to load the JSON directly
            print(f"Detected JSON input file, trying to load directly: {input_path}")
            with open(input_path, 'r', encoding='utf-8') as f:
                document_data = json.load(f)
            
            # Check if it's a processed document with expected structure
            if isinstance(document_data, dict) and 'chunks' in document_data and 'metadata' in document_data:
                print(f"Successfully loaded pre-processed document from JSON")
                process_result = {"success": True, "document": document_data}
                
                # Save the processed JSON to the output directory for indexing
                output_json_path = output_dir / f"{input_path.stem}_processed.json"
                print(f"Saving processed document to {output_json_path} for indexing")
                with open(output_json_path, 'w', encoding='utf-8') as f:
                    json.dump(document_data, f, ensure_ascii=False, indent=2)
            else:
                # Process it as a normal file
                print(f"JSON doesn't contain a valid document structure, processing as regular file")
                process_result = processor.process_file(args.input, output_dir)
        except Exception as e:
            print(f"Error loading JSON: {e}, processing as regular file")
            process_result = processor.process_file(args.input, output_dir)
    else:
        # Process as normal file
        process_result = processor.process_file(args.input, output_dir)
    
    if not process_result["success"]:
        print(f"Errore nell'elaborazione del documento: {process_result.get('error', 'Errore sconosciuto')}")
        sys.exit(1)
    
    # Extract document info
    document = process_result["document"]
    doc_title = document['metadata'].get('title', Path(args.input).name)
    
    print(f"Documento elaborato: {doc_title}")
    print(f"Numero di chunk creati: {len(document['chunks'])}")
    print(f"Metadati estratti: {', '.join(filter(None, [document['metadata'].get(k) for k in ['title', 'author', 'date', 'language']]))}")
    
    # Show chunk information
    print("\n--- Informazioni sui chunk ---")
    semantic_chunks = 0
    total_segments = 0
    
    for i, chunk in enumerate(document['chunks']):
        is_semantic = chunk.get('semantic', False)
        if is_semantic:
            semantic_chunks += 1
        
        segments = chunk.get('segments', [])
        total_segments += len(segments)
        segment_types = {}
        for seg in segments:
            seg_type = seg.get('type', 'unknown')
            segment_types[seg_type] = segment_types.get(seg_type, 0) + 1
        
        segment_type_str = ", ".join(f"{k}: {v}" for k, v in segment_types.items())
        chunk_type = "semantico" if is_semantic else "dimensionale"
        
        print(f"Chunk {i+1}: {len(chunk['text'])} caratteri, {len(segments)} segmenti ({segment_type_str}), tipo: {chunk_type}")
    
    if semantic_chunks > 0:
        print(f"\nChunk semantici: {semantic_chunks} ({semantic_chunks/len(document['chunks'])*100:.1f}%)")
    print(f"Segmenti totali: {total_segments} (media: {total_segments/len(document['chunks']):.1f} per chunk)")
    
    # Index the document
    print("\n=== Indicizzazione del documento ===")
    indexer = VectorIndexer(config)
    index_result = indexer.index_directory(output_dir)
    
    if not index_result["success"]:
        print(f"Errore nell'indicizzazione: {index_result.get('error', 'Errore sconosciuto')}")
        sys.exit(1)
    
    save_result = indexer.save_index(index_dir)
    print(f"Indice salvato in {index_dir}")
    
    # Test with queries
    default_queries = [
        "Qual è l'argomento principale di questo documento?",
        "Riassumi i punti chiave di questo documento.",
        "Quali raccomandazioni vengono fatte in questo documento?",
    ]
    
    queries = args.queries if args.queries else default_queries
    
    print("\n=== Test di query RAG ===")
    
    # Initialize RAG pipeline
    # Check config being passed to the pipeline
    print("=== Config diagnostics ===")
    print(f"Config type: {type(config)}")
    llm_config = config.get('llm', {})
    print(f"LLM config: {llm_config}")
    rewriting_config = config.get('rewriting', {})
    print(f"Rewriting config: {rewriting_config}")
    generation_config = config.get('generation', {})
    print(f"Generation config: {generation_config}")
    
    # Initialize the pipeline with the forced config
    pipeline = RagPipeline(config)
    load_result = pipeline.retriever.load_index(index_dir)
    
    if not load_result["success"]:
        print(f"Errore nel caricamento dell'indice: {load_result.get('error', 'Errore sconosciuto')}")
        sys.exit(1)
    
    # File for saving results
    results_file = output_dir / "rag_results.json"
    results = []
    
    for query in queries:
        print("\n" + "=" * 80)
        print(f"QUERY: {query}")
        
        # Query RAG pipeline
        response = pipeline.query(query, top_k=args.top_k)
        
        if response["success"]:
            if "rewritten_query" in response:
                print(f"Query riscritta: {response['rewritten_query']}")
            
            print(f"\nRISPOSTA ({response['chunks_used']} chunks usati):")
            print(response["response"])
            
            # Get used chunks
            used_chunks = []
            for i, step in enumerate(response.get("steps", [])):
                if step["step"] == "retrieve":
                    retrieved_chunks = step["result"].get("chunks", [])
                    for chunk in retrieved_chunks[:args.top_k]:
                        used_chunks.append({
                            "id": chunk.get("id", f"chunk_{i}"),
                            "similarity": chunk.get("similarity", 0),
                            "text_preview": chunk.get("text", "")[:100] + "..."
                        })
            
            # Save result
            results.append({
                "query": query,
                "rewritten_query": response.get("rewritten_query"),
                "response": response["response"],
                "chunks_used": response["chunks_used"],
                "used_chunks": used_chunks,
                "success": True
            })
        else:
            print(f"\nERRORE: {response.get('error', 'Errore sconosciuto')}")
            results.append({
                "query": query,
                "success": False,
                "error": response.get("error", "Errore sconosciuto")
            })
    
    # Save results
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\nRisultati salvati in: {results_file}")
    print("\n=== Test completato ===")

if __name__ == "__main__":
    main()