#!/usr/bin/env python3
"""
Test della pipeline RAG (Retrieval-Augmented Generation) usando la configurazione TOML.
"""

import json
import os
import sys
import argparse
from pathlib import Path
import tempfile
import time

# Add the project directory to PYTHONPATH
current_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(current_dir))

from raggiro.utils.config import load_config
from raggiro.rag.indexer import VectorIndexer
from raggiro.rag.pipeline import RagPipeline
from raggiro.rag.retriever import VectorRetriever

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Test della pipeline RAG con documenti elaborati')
    parser.add_argument('--input', '-i', type=str, 
                      default=str(Path.cwd() / 'test_output' / 'sample_report.json'),
                      help='Path al file JSON del documento elaborato')
    parser.add_argument('--output', '-o', type=str, 
                      default=str(Path.cwd() / 'test_output'),
                      help='Directory di output per i risultati')
    parser.add_argument('--index', type=str, 
                      default=None,
                      help='Directory per l\'indice vettoriale (se non specificata, verrà creata una temporanea)')
    parser.add_argument('--queries', '-q', type=str, nargs='+',
                      help='Query da testare (opzionale)')
    parser.add_argument('--top-k', type=int, default=3,
                      help='Numero di chunk da recuperare per query (default: 3)')
    return parser.parse_args()

def main():
    """Main function."""
    # Parse arguments
    args = parse_arguments()
    
    # Configurazioni
    output_dir = Path(args.output)
    input_json = Path(args.input)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if we need a temporary index directory
    if args.index:
        index_dir = Path(args.index)
        index_dir.mkdir(parents=True, exist_ok=True)
        temp_index_dir = None
    else:
        # Create a temporary directory for the index
        temp_index_dir = tempfile.mkdtemp(prefix="raggiro_index_")
        index_dir = Path(temp_index_dir)
    
    # Load configuration
    config_path = current_dir / "config" / "config.toml"
    print(f"Loading config from: {config_path}")

    try:
        config = load_config(str(config_path))
        
        # Print diagnostics about config
        print(f"Using Ollama URL: {config.get('llm', {}).get('ollama_base_url', 'Not set')}")
        print(f"LLM provider: {config.get('llm', {}).get('provider', 'Not set')}")
        print(f"Rewriting model: {config.get('rewriting', {}).get('ollama_model', 'Not set')}")
        print(f"Generation model: {config.get('generation', {}).get('ollama_model', 'Not set')}")
    except Exception as e:
        print(f"Error loading config: {str(e)}")
        print("Using hardcoded configuration with correct Ollama URL")
        config = {
            "llm": {
                "provider": "ollama",
                "ollama_base_url": "http://ollama:11434",
                "ollama_timeout": 30
            },
            "rewriting": {
                "enabled": True,
                "llm_type": "ollama",
                "ollama_model": "llama3",
                "temperature": 0.1,
                "max_tokens": 200,
                "ollama_base_url": "http://ollama:11434"
            },
            "generation": {
                "llm_type": "ollama",
                "ollama_model": "mistral",
                "temperature": 0.7,
                "max_tokens": 1000,
                "ollama_base_url": "http://ollama:11434"
            }
        }
        print(f"Using Ollama URL: {config['llm']['ollama_base_url']}")

    print("=== Test della pipeline RAG ===")
    print(f"File di input: {input_json}")

    # Carica il documento JSON
    try:
        with open(input_json, 'r', encoding='utf-8') as f:
            document = json.load(f)
        print(f"Documento caricato: {document['metadata'].get('title', 'Documento senza titolo')}")
        print(f"Contiene {len(document['chunks'])} chunks")
    except Exception as e:
        print(f"Errore durante il caricamento del documento: {str(e)}")
        sys.exit(1)

    # Save document to output dir for indexing
    document_json_path = output_dir / f"{input_json.stem}_processed.json"
    with open(document_json_path, 'w', encoding='utf-8') as f:
        json.dump(document, f, ensure_ascii=False, indent=2)
    
    # Initialize the indexer and index the document
    print(f"\n=== Indicizzazione del documento ===")
    indexer = VectorIndexer(config)
    index_result = indexer.index_directory(output_dir)
    
    if not index_result["success"]:
        print(f"Errore nell'indicizzazione: {index_result.get('error', 'Errore sconosciuto')}")
        sys.exit(1)
    
    # Save the index
    save_result = indexer.save_index(index_dir)
    if save_result["success"]:
        print(f"Indice salvato in {index_dir}")
    else:
        print(f"Errore nel salvataggio dell'indice: {save_result.get('error', 'Errore sconosciuto')}")
        sys.exit(1)
    
    # Initialize the RAG pipeline
    print("\n=== Inizializzazione pipeline RAG ===")
    pipeline = RagPipeline(config)
    
    # Load the index
    load_result = pipeline.retriever.load_index(index_dir)
    
    if not load_result["success"]:
        print(f"Errore nel caricamento dell'indice: {load_result.get('error', 'Errore sconosciuto')}")
        sys.exit(1)
    
    # Test con alcune query diverse
    default_queries = [
        "Quali sono i principali trend in AI nel 2023?",
        "Come viene utilizzata l'AI in ambito sanitario?",
        "Quali sfide devono affrontare le organizzazioni nell'implementazione dell'AI?",
        "Cosa sono i Large Language Models e quali sono i loro sviluppi recenti?"
    ]

    # Usa le query fornite dall'utente o quelle predefinite
    queries = args.queries if args.queries else default_queries

    print("\n=== Esecuzione di query RAG ===")
    
    # File per salvare i risultati
    results_file = output_dir / "rag_results.json"
    results = []

    for query in queries:
        print("\n" + "=" * 80)
        print(f"QUERY: {query}")
        
        # Query the RAG pipeline
        start_time = time.time()
        response = pipeline.query(query, top_k=args.top_k)
        query_time = time.time() - start_time
        
        if response["success"]:
            # Show rewritten query if available
            if "rewritten_query" in response:
                print(f"\nQuery riscritta: {response['rewritten_query']}")
            
            # Show response
            print(f"\nRISPOSTA ({response['chunks_used']} chunks usati, {query_time:.2f}s):")
            print(response["response"])
            
            # Extract chunk information if available
            used_chunks = []
            for step in response.get("steps", []):
                if step["step"] == "retrieve":
                    retrieved_chunks = step["result"].get("chunks", [])
                    for chunk in retrieved_chunks[:args.top_k]:
                        used_chunks.append({
                            "id": chunk.get("id", "unknown"),
                            "similarity": chunk.get("similarity", 0),
                            "text_preview": chunk.get("text", "")[:100] + "..."
                        })
            
            # Save result
            results.append({
                "query": query,
                "rewritten_query": response.get("rewritten_query"),
                "response": response["response"],
                "chunks_used": response.get("chunks_used", 0),
                "query_time": query_time,
                "used_chunks": used_chunks,
                "success": True
            })
        else:
            print(f"\nERRORE: {response.get('error', 'Errore sconosciuto')}")
            
            results.append({
                "query": query,
                "success": False,
                "error": response.get("error", "Errore sconosciuto"),
                "query_time": query_time
            })

    # Salva i risultati in formato JSON
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nRisultati salvati in: {results_file}")

    # Clean up temporary index directory if we created one
    if temp_index_dir:
        import shutil
        try:
            shutil.rmtree(temp_index_dir)
            print(f"Directory temporanea dell'indice rimossa: {temp_index_dir}")
        except Exception as e:
            print(f"Errore nella rimozione della directory temporanea: {e}")

    print("\n=== Test RAG completato ===")

if __name__ == "__main__":
    main()