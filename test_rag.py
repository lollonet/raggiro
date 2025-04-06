#!/usr/bin/env python3
"""
Test semplificato della pipeline RAG (Retrieval-Augmented Generation)
"""

import json
import os
import sys
import argparse
from pathlib import Path
import re

# Add the project directory to PYTHONPATH
current_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(current_dir))

from raggiro.utils.config import load_config

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Test della pipeline RAG con documenti elaborati')
    parser.add_argument('--input', '-i', type=str, 
                      default=str(Path.cwd() / 'test_output' / 'sample_report.json'),
                      help='Path al file JSON del documento elaborato')
    parser.add_argument('--output', '-o', type=str, 
                      default=str(Path.cwd() / 'test_output'),
                      help='Directory di output per i risultati')
    parser.add_argument('--queries', '-q', type=str, nargs='+',
                      help='Query da testare (opzionale)')
    return parser.parse_args()

# Parsing degli argomenti
args = parse_arguments()

# Configurazioni
TEST_DIR = Path(args.output)
INPUT_JSON = Path(args.input)

# Load configuration
config_path = current_dir / "config" / "config.toml"
print(f"Loading config from: {config_path}")
config = load_config(str(config_path))
print(f"Using Ollama URL: {config.get('llm', {}).get('ollama_base_url', 'Not set')}")

print("=== Test della pipeline RAG ===")
print(f"File di input: {INPUT_JSON}")

# Funzione per simulare il retrieval basato su similarità semantica
def simulate_retrieval(document, query, top_k=3):
    """Simulazione semplificata del retrieval di chunks rilevanti."""
    print(f"\nQuery: {query}")
    print(f"Cerco i {top_k} chunks più rilevanti...")
    
    # In un caso reale, useremmo embedding vettoriali per la similarità semantica
    # Qui usiamo una semplice ricerca per parole chiave
    query_terms = query.lower().split()
    
    # Calcola punteggi basati su semplice conteggio di termini
    chunk_scores = []
    for i, chunk in enumerate(document["chunks"]):
        score = 0
        text = chunk["text"].lower()
        for term in query_terms:
            score += text.count(term)
        chunk_scores.append((i, score))
    
    # Ordina per punteggio e prendi i top k
    top_chunks = sorted(chunk_scores, key=lambda x: x[1], reverse=True)[:top_k]
    
    results = []
    for i, score in top_chunks:
        if score > 0:  # Solo chunk con punteggio positivo
            chunk = document["chunks"][i]
            results.append({
                "id": chunk["id"],
                "text": chunk["text"],
                "score": score
            })
    
    print(f"Trovati {len(results)} chunks rilevanti")
    return results

# Funzione per simulare la generazione di una risposta
def simulate_response_generation(query, chunks):
    """Simulazione della generazione di risposta basata sui chunks recuperati."""
    print("\nGenerazione della risposta...")
    
    if not chunks:
        return "Non ho trovato informazioni rilevanti per rispondere alla tua domanda."
    
    # Prepara la risposta basata sui chunks recuperati
    # In un caso reale, qui useremmo un LLM
    response = f"Basandomi sulle informazioni disponibili, ecco la risposta alla tua domanda: '{query}'\n\n"
    
    # Estrai frasi rilevanti dai chunks
    relevant_sentences = []
    for chunk in chunks:
        text = chunk["text"]
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        for sentence in sentences:
            for term in query.lower().split():
                if term in sentence.lower() and len(sentence) > 20:
                    relevant_sentences.append((sentence, chunk["id"]))
                    break
    
    # Costruisci la risposta
    if relevant_sentences:
        response += "La risposta è:\n\n"
        for i, (sentence, chunk_id) in enumerate(relevant_sentences[:5]):
            response += f"{i+1}. {sentence} [Fonte: {chunk_id}]\n\n"
    else:
        # Se non troviamo frasi specifiche, usa i chunk interi
        response += "Ecco le informazioni più rilevanti:\n\n"
        for i, chunk in enumerate(chunks):
            excerpt = chunk["text"][:200] + "..." if len(chunk["text"]) > 200 else chunk["text"]
            response += f"Fonte {i+1} [{chunk['id']}]: {excerpt}\n\n"
    
    return response

# Carica il documento JSON
try:
    with open(INPUT_JSON, 'r') as f:
        document = json.load(f)
    print(f"Documento caricato: {document['metadata'].get('title', 'Documento senza titolo')}")
    print(f"Contiene {len(document['chunks'])} chunks")
except Exception as e:
    print(f"Errore durante il caricamento del documento: {str(e)}")
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

print("\n=== Simulazione di query RAG ===")

# Assicurati che la directory di output esista
TEST_DIR.mkdir(parents=True, exist_ok=True)

# File per salvare i risultati
results_file = TEST_DIR / "rag_results.json"
results = []

for query in queries:
    print("\n" + "=" * 80)
    print(f"QUERY: {query}")
    
    # 1. Retrieval - trova i chunks rilevanti
    retrieved_chunks = simulate_retrieval(document, query, top_k=2)
    
    # 2. Generazione - crea una risposta basata sui chunks
    if retrieved_chunks:
        response = simulate_response_generation(query, retrieved_chunks)
        print("\nRISPOSTA:")
        print(response)
        
        # Salva il risultato
        results.append({
            "query": query,
            "chunks": [{"id": c["id"], "score": c["score"]} for c in retrieved_chunks],
            "response": response
        })
    else:
        print("\nNessun risultato rilevante trovato per questa query.")
        results.append({
            "query": query,
            "chunks": [],
            "response": "Nessun risultato rilevante trovato."
        })

# Salva i risultati in formato JSON
with open(results_file, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\nRisultati salvati in: {results_file}")

print("\n=== Test RAG completato ===")