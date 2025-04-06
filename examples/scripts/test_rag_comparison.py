#!/usr/bin/env python3
"""
Test per confrontare diverse strategie di chunking sulla stessa query RAG
"""

import os
import sys
import json
import argparse
from pathlib import Path
import tempfile
import shutil
import time

# Aggiungi il percorso principale al Python path
current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent.parent
sys.path.insert(0, str(root_dir))

from raggiro.processor import DocumentProcessor
from raggiro.rag.indexer import VectorIndexer
from raggiro.rag.pipeline import RagPipeline
from raggiro.utils.config import load_config

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Confronto tra strategie di chunking per RAG')
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Path al documento da elaborare (PDF, DOCX, ecc.)')
    parser.add_argument('--output', '-o', type=str, 
                        default=str(Path.cwd() / 'test_comparison'),
                        help='Directory di output per i risultati')
    parser.add_argument('--queries', '-q', type=str, nargs='+',
                        help='Query da testare (opzionale)')
    parser.add_argument('--strategies', '-s', type=str, nargs='+',
                        default=['size', 'semantic', 'hybrid'],
                        help='Strategie di chunking da confrontare')
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

def process_with_strategy(input_file, output_dir, strategy, config):
    """Process a document with a specific chunking strategy."""
    # Create temp directories
    temp_dir = Path(tempfile.mkdtemp(prefix="raggiro_"))
    temp_output = temp_dir / "output"
    temp_index = temp_dir / "index"
    temp_output.mkdir(parents=True, exist_ok=True)
    temp_index.mkdir(parents=True, exist_ok=True)
    
    try:
        # Modify config to use specified strategy
        config_copy = config.copy()
        if "segmentation" not in config_copy:
            config_copy["segmentation"] = {}
        
        config_copy["segmentation"]["semantic_chunking"] = strategy != "size"
        config_copy["segmentation"]["chunking_strategy"] = strategy
        
        # Process document
        start_time = time.time()
        processor = DocumentProcessor(config_copy)
        process_result = processor.process_file(input_file, temp_output)
        process_time = time.time() - start_time
        
        if not process_result["success"]:
            return {
                "strategy": strategy,
                "success": False,
                "error": process_result.get("error", "Unknown error"),
                "process_time": process_time
            }
        
        document = process_result["document"]
        
        # Index document
        indexer = VectorIndexer(config_copy)
        index_result = indexer.index_directory(temp_output)
        
        if not index_result["success"]:
            return {
                "strategy": strategy,
                "success": False,
                "error": index_result.get("error", "Failed to index document"),
                "process_time": process_time
            }
        
        indexer.save_index(temp_index)
        
        # Return information about the processing
        chunk_info = [{
            "id": chunk.get("id", f"chunk_{i}"),
            "length": len(chunk["text"]),
            "segments": len(chunk.get("segments", [])),
            "semantic": chunk.get("semantic", False)
        } for i, chunk in enumerate(document.get("chunks", []))]
        
        return {
            "strategy": strategy,
            "success": True,
            "document": {
                "title": document["metadata"].get("title", Path(input_file).name),
                "chunks": len(document.get("chunks", [])),
                "semantic_chunks": sum(1 for c in document.get("chunks", []) if c.get("semantic", False)),
                "chunk_info": chunk_info
            },
            "temp_dirs": {
                "output": str(temp_output),
                "index": str(temp_index)
            },
            "process_time": process_time
        }
    except Exception as e:
        return {
            "strategy": strategy,
            "success": False,
            "error": str(e),
            "process_time": 0
        }

def test_queries(queries, strategy_results, config, top_k=3):
    """Test queries against each processing strategy."""
    query_results = []
    
    for query in queries:
        query_result = {
            "query": query,
            "strategies": []
        }
        
        for strategy_result in strategy_results:
            if not strategy_result["success"]:
                query_result["strategies"].append({
                    "strategy": strategy_result["strategy"],
                    "success": False,
                    "error": strategy_result.get("error", "Unknown error")
                })
                continue
            
            # Initialize RAG pipeline for this strategy
            strategy_config = config.copy()
            strategy_config["segmentation"] = strategy_config.get("segmentation", {}).copy()
            strategy_config["segmentation"]["semantic_chunking"] = strategy_result["strategy"] != "size"
            strategy_config["segmentation"]["chunking_strategy"] = strategy_result["strategy"]
            
            pipeline = RagPipeline(strategy_config)
            
            # Load index
            index_dir = strategy_result["temp_dirs"]["index"]
            load_result = pipeline.retriever.load_index(index_dir)
            
            if not load_result["success"]:
                query_result["strategies"].append({
                    "strategy": strategy_result["strategy"],
                    "success": False,
                    "error": "Failed to load index"
                })
                continue
            
            # Query the pipeline
            start_time = time.time()
            response = pipeline.query(query, top_k=top_k)
            query_time = time.time() - start_time
            
            if response["success"]:
                # Extract used chunks info
                used_chunks = []
                for i, step in enumerate(response.get("steps", [])):
                    if step["step"] == "retrieve":
                        retrieved_chunks = step["result"].get("chunks", [])
                        for chunk in retrieved_chunks[:top_k]:
                            used_chunks.append({
                                "id": chunk.get("id", f"chunk_{i}"),
                                "similarity": chunk.get("similarity", 0),
                                "text_preview": chunk.get("text", "")[:100] + "..."
                            })
                
                query_result["strategies"].append({
                    "strategy": strategy_result["strategy"],
                    "success": True,
                    "response": response["response"],
                    "rewritten_query": response.get("rewritten_query"),
                    "chunks_used": response.get("chunks_used", 0),
                    "used_chunks": used_chunks,
                    "query_time": query_time
                })
            else:
                query_result["strategies"].append({
                    "strategy": strategy_result["strategy"],
                    "success": False,
                    "error": response.get("error", "Unknown error"),
                    "query_time": query_time
                })
        
        query_results.append(query_result)
    
    return query_results

def main():
    """Main function."""
    # Parse arguments
    args = parse_arguments()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load configuration with proper path
    config_path = root_dir / "config" / "config.toml"
    print(f"Loading config from: {config_path}")
    
    # Try loading configuration, but handle interpolation errors and override with command line args
    try:
        # Load base config
        config = load_config(str(config_path))
        # Override with command line arguments
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
        print(f"Error loading config: {str(e)}")
        print("Using hardcoded configuration with command line Ollama settings")
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
    
    # Print configuration diagnostics
    print(f"Using Ollama URL: {config.get('llm', {}).get('ollama_base_url', 'Not set')}")
    print(f"Using rewriting model: {config.get('rewriting', {}).get('ollama_model', 'Not set')}")
    print(f"Using generation model: {config.get('generation', {}).get('ollama_model', 'Not set')}")
    
    # Process with each strategy
    print(f"=== Confronto strategie di chunking per: {args.input} ===")
    print(f"Strategie: {', '.join(args.strategies)}")
    
    strategy_results = []
    for strategy in args.strategies:
        print(f"\nElaborazione con strategia: {strategy}")
        result = process_with_strategy(args.input, output_dir, strategy, config)
        
        if result["success"]:
            print(f"✅ Successo: {result['document']['chunks']} chunks creati ({result['process_time']:.2f}s)")
            if strategy != "size":
                print(f"  Chunk semantici: {result['document']['semantic_chunks']} ({result['document']['semantic_chunks']/result['document']['chunks']*100:.1f}%)")
        else:
            print(f"❌ Errore: {result.get('error', 'Errore sconosciuto')}")
            
        strategy_results.append(result)
    
    # Set up queries
    default_queries = [
        "Qual è l'argomento principale di questo documento?",
        "Riassumi i punti chiave di questo documento.",
        "Quali raccomandazioni vengono fatte in questo documento?",
    ]
    
    queries = args.queries if args.queries else default_queries
    
    # Test queries
    print("\n=== Test query RAG ===")
    query_results = test_queries(queries, strategy_results, config, args.top_k)
    
    # Save results
    comparison_results = {
        "input_file": args.input,
        "strategies": args.strategies,
        "top_k": args.top_k,
        "strategy_results": [
            {k: v for k, v in r.items() if k != "temp_dirs"} 
            for r in strategy_results
        ],
        "query_results": query_results
    }
    
    results_file = output_dir / "comparison_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(comparison_results, f, ensure_ascii=False, indent=2)
    
    # Create a summary markdown file
    summary_file = output_dir / "comparison_summary.md"
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(f"# Confronto Strategie di Chunking RAG\n\n")
        f.write(f"File di input: `{args.input}`\n\n")
        f.write(f"## Statistiche Chunking\n\n")
        
        f.write("| Strategia | Chunks | Chunks semantici | Tempo (s) |\n")
        f.write("|-----------|--------|------------------|----------|\n")
        
        for result in strategy_results:
            if result["success"]:
                semantic_chunks = result["document"].get("semantic_chunks", 0)
                total_chunks = result["document"].get("chunks", 0)
                semantic_percent = f"{semantic_chunks/total_chunks*100:.1f}%" if total_chunks > 0 else "N/A"
                f.write(f"| {result['strategy']} | {total_chunks} | {semantic_chunks} ({semantic_percent}) | {result['process_time']:.2f} |\n")
            else:
                f.write(f"| {result['strategy']} | Errore | Errore | {result['process_time']:.2f} |\n")
        
        f.write("\n## Risultati Query\n\n")
        
        for query_result in query_results:
            f.write(f"### Query: {query_result['query']}\n\n")
            
            for strategy_query in query_result["strategies"]:
                f.write(f"#### Strategia: {strategy_query['strategy']}\n\n")
                
                if strategy_query["success"]:
                    rewritten = strategy_query.get("rewritten_query")
                    if rewritten:
                        f.write(f"Query riscritta: {rewritten}\n\n")
                    
                    f.write(f"Risposta ({strategy_query.get('chunks_used', 0)} chunks, {strategy_query.get('query_time', 0):.2f}s):\n\n")
                    f.write(f"{strategy_query['response']}\n\n")
                else:
                    f.write(f"❌ Errore: {strategy_query.get('error', 'Errore sconosciuto')}\n\n")
    
    print(f"\nRisultati salvati in: {results_file}")
    print(f"Riepilogo salvato in: {summary_file}")
    
    # Clean up temp directories
    for result in strategy_results:
        if result["success"] and "temp_dirs" in result:
            try:
                temp_output = result["temp_dirs"]["output"]
                temp_index = result["temp_dirs"]["index"]
                temp_parent = Path(temp_output).parent
                
                shutil.rmtree(temp_parent, ignore_errors=True)
            except Exception as e:
                print(f"Warning: Failed to clean up temp directories: {e}")
    
    print("\n=== Confronto completato ===")

if __name__ == "__main__":
    main()