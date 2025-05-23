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
root_dir = current_dir.parent
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
    # Add Ollama configuration options
    parser.add_argument('--ollama-url', type=str, default='http://ollama:11434',
                        help='URL del server Ollama')
    parser.add_argument('--rewriting-model', type=str, default='llama3',
                        help='Nome del modello Ollama per il rewriting delle query')
    parser.add_argument('--generation-model', type=str, default='mistral',
                        help='Nome del modello Ollama per la generazione delle risposte')
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
    
    # Try loading configuration, but handle interpolation errors and override with command line args
    try:
        # First try to load with utility function
        config = load_config(str(config_path))
        # Override config with command line args
        if "llm" not in config:
            config["llm"] = {}
        config["llm"]["provider"] = "ollama"
        config["llm"]["ollama_base_url"] = args.ollama_url
        
        if "rewriting" not in config:
            config["rewriting"] = {}
        config["rewriting"]["llm_type"] = "ollama"
        config["rewriting"]["ollama_model"] = args.rewriting_model
        config["rewriting"]["ollama_base_url"] = args.ollama_url
        
        if "generation" not in config:
            config["generation"] = {}
        config["generation"]["llm_type"] = "ollama"
        config["generation"]["ollama_model"] = args.generation_model
        config["generation"]["ollama_base_url"] = args.ollama_url
    except Exception as e:
        print(f"Error loading config directly: {str(e)}")
        # Fall back to config with command line args
        config = {
            "llm": {
                "provider": "ollama",
                "ollama_base_url": args.ollama_url,
                "ollama_timeout": 30
            },
            "rewriting": {
                "enabled": True,
                "llm_type": "ollama",
                "ollama_model": args.rewriting_model,
                "temperature": 0.1,
                "max_tokens": 200,
                "ollama_base_url": args.ollama_url
            },
            "generation": {
                "llm_type": "ollama",
                "ollama_model": args.generation_model,
                "temperature": 0.7,
                "max_tokens": 1000,
                "ollama_base_url": args.ollama_url
            },
            "segmentation": {
                "semantic_chunking": True,
                "chunking_strategy": "hybrid"
            }
        }
    # Print configuration details for verification
    print(f"Using Ollama URL: {config.get('llm', {}).get('ollama_base_url', 'Not set')}")
    print(f"Using rewriting model: {config.get('rewriting', {}).get('ollama_model', 'Not set')}")
    print(f"Using generation model: {config.get('generation', {}).get('ollama_model', 'Not set')}")
    print(f"Strategia di chunking attuale: {config.get('segmentation', {}).get('chunking_strategy', 'size')}")
    
    # Process the document
    print(f"\n=== Elaborazione documento: {args.input} ===")
    processor = DocumentProcessor(config)
    
    process_result = processor.process_file(args.input, output_dir)
    
    if not process_result["success"]:
        print(f"Errore nell'elaborazione del documento: {process_result.get('error', 'Errore sconosciuto')}")
        sys.exit(1)
    
    # Extract document info
    document = process_result["document"]
    doc_title = document['metadata'].get('title', Path(args.input).name)
    
    print(f"Documento elaborato: {doc_title}")
    print(f"Numero di chunks: {len(document['chunks'])}")
    
    # Analyze chunks
    print("\n=== Analisi dei chunks ===")
    total_segments = 0
    semantic_chunks = 0
    
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
    print("=== Config diagnostics ===")
    print(f"Config type: {type(config)}")
    
    # Verify and log LLM configurations
    llm_config = config.get('llm', {})
    rewriting_config = config.get('rewriting', {})
    generation_config = config.get('generation', {})
    
    # Log and confirm Ollama URLs are correctly set
    ollama_base_url = llm_config.get('ollama_base_url', 'Not set')
    rewriting_ollama_url = rewriting_config.get('ollama_base_url', 'Inherits from LLM')
    generation_ollama_url = generation_config.get('ollama_base_url', 'Inherits from LLM')
    
    print(f"Main LLM Ollama URL: {ollama_base_url}")
    print(f"Rewriting LLM Ollama URL: {rewriting_ollama_url}")
    print(f"Generation LLM Ollama URL: {generation_ollama_url}")
    
    print(f"Rewriting model: {rewriting_config.get('ollama_model', 'Not set')}")
    print(f"Generation model: {generation_config.get('ollama_model', 'Not set')}")
    
    # Log retrieval configuration
    retrieval_config = config.get('retrieval', {})
    print(f"Retrieval top_k: {retrieval_config.get('top_k', 'Not set')}")
    print(f"Using command line top_k: {args.top_k}")
    
    # Make sure all components use the same Ollama URL
    # This ensures consistent behavior across all components
    if llm_config.get('provider') == 'ollama':
        # Create a validated config with consistent URLs
        validated_config = config.copy()
        
        # Make sure rewriting and generation use the same URL as the main LLM config
        if 'rewriting' not in validated_config:
            validated_config['rewriting'] = {}
        if 'generation' not in validated_config:
            validated_config['generation'] = {}
            
        validated_config['rewriting']['ollama_base_url'] = ollama_base_url
        validated_config['generation']['ollama_base_url'] = ollama_base_url
        
        print(f"Using validated Ollama URL: {ollama_base_url} for all components")
        
        # Initialize the pipeline with the validated config
        pipeline = RagPipeline(validated_config)
    else:
        # Initialize with original config for non-Ollama providers
        pipeline = RagPipeline(config)
        
    # Load the index
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
        
        try:
            # Get response
            response = pipeline.query(query, top_k=args.top_k)
            
            if response["success"]:
                print(f"RISPOSTA:\n{response['response']}")
                
                # Add query result to results list
                results.append({
                    "query": query,
                    "response": response["response"],
                    "chunks_used": response.get("chunks_used", 0),
                    "success": True
                })
                
                # Check if there was query rewriting
                if "rewritten_query" in response:
                    print(f"\nQuery riscritta: {response['rewritten_query']}")
            else:
                print(f"\nERRORE: {response.get('error', 'Errore sconosciuto')}")
                
                # Add error response to results list
                results.append({
                    "query": query,
                    "error": response.get("error", "Errore sconosciuto"),
                    "success": False
                })
        except Exception as e:
            print(f"\nERRORE: {str(e)}")
            
            # Add exception to results list
            results.append({
                "query": query,
                "error": str(e),
                "success": False
            })
    
    # Save results
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\nRisultati salvati in: {results_file}")
    
    print("\n=== Test completato ===")

if __name__ == "__main__":
    main()