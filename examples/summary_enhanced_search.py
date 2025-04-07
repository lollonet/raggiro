#!/usr/bin/env python
"""
Example showcasing the three summary-enhanced search strategies in Raggiro.

This example demonstrates:
1. Dual embeddings (text + summary)
2. Summary-based filtering and relevance boosting 
3. Enhanced response templates incorporating summaries
"""

import os
import sys
import json
from pathlib import Path

# Add parent directory to path for direct running
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from raggiro.core.segmenter import TextSegmenter
from raggiro.rag.pipeline import RagPipeline

def main():
    """Run example showcasing summary-enhanced search."""
    print("\n🔍 Raggiro: Summary-Enhanced Search Example")
    print("=" * 60)
    
    # Create sample document with summaries
    print("\n1️⃣ Creating sample chunks with summaries...")
    chunks = create_sample_chunks_with_summaries()
    print(f"   ✓ Created {len(chunks)} chunks with summaries")
    
    # Save chunks to a file for indexing
    output_dir = Path("./tmp/summary_example")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "sample_document.json"
    
    document = {
        "metadata": {
            "title": "Summary-Enhanced Search Example",
            "file": {
                "path": str(output_path),
                "hash": "sample123",
                "name": "sample_document.json"
            }
        },
        "chunks": chunks
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(document, f, ensure_ascii=False, indent=2)
    print(f"   ✓ Saved document to {output_path}")
    
    # Configure the RAG pipeline with summary enhancement
    config = {
        "indexing": {
            "embedding_model": "all-MiniLM-L6-v2",
            "vector_db": "faiss",
            "use_dual_embeddings": True,  # Strategy 1: Use dual embeddings
            "summary_weight": 0.3,  # Weight for summary vs full text
        },
        "retrieval": {
            "embedding_model": "all-MiniLM-L6-v2",
            "vector_db": "faiss",
            "top_k": 3,
            "use_summary_filtering": True,  # Strategy 2: Summary-based filtering
            "summary_relevance_threshold": 0.3,
            "summary_boost_factor": 0.2,
        },
        "generation": {
            "llm_type": "ollama",
            "ollama_model": "mistral",
            # Default prompt template already enhanced with summary instructions
            # Strategy 3: Enhanced prompt template using summaries
        }
    }
    
    # Create and configure the pipeline
    print("\n2️⃣ Setting up RAG pipeline with summary enhancement...")
    pipeline = RagPipeline(config)
    
    # Index the document
    print("\n3️⃣ Indexing document with dual embeddings (text + summary)...")
    index_result = pipeline.indexer.index_document(output_path)
    if index_result["success"]:
        print(f"   ✓ Successfully indexed {index_result['chunks_indexed']} chunks")
    else:
        print(f"   ✗ Failed to index document: {index_result.get('error')}")
        return
    
    # Try different queries to demonstrate summary-based search
    test_queries = [
        "Quali sono i vantaggi del machine learning?",
        "Come si può migliorare la sicurezza informatica?",
        "Cosa distingue un database SQL da NoSQL?"
    ]
    
    # Process each query with the summary-enhanced pipeline
    print("\n4️⃣ Testing queries with summary-enhanced retrieval:")
    
    for query in test_queries:
        print(f"\n📝 Query: \"{query}\"")
        
        # Run the query through the pipeline
        result = pipeline.query(query)
        
        if not result["success"]:
            print(f"   ✗ Query failed: {result.get('error')}")
            continue
        
        # Print retrieved chunks with summaries and scores
        print("\n   📊 Retrieved chunks:")
        for i, chunk in enumerate(result.get("chunks", [])[:3]):
            print(f"   Chunk {i+1} (score: {chunk['score']:.4f}):")
            
            # Print summary first
            if "summary" in chunk:
                print(f"   Summary: {chunk['summary'][:150]}...")
            
            # Print relevance boost if available
            if "score_boost" in chunk:
                print(f"   Boost: +{chunk['score_boost']:.4f} (from summary relevance: {chunk.get('summary_relevance', 0):.4f})")
                
            print(f"   Text: {chunk['text'][:100]}...\n")
        
        # Print the generated response
        print("\n   🤖 Generated response:")
        print(f"   {result['response'][:500]}...")
        print("\n" + "-" * 60)

def create_sample_chunks_with_summaries():
    """Create sample chunks with summaries for demonstration."""
    
    # Sample texts with their corresponding manually-created summaries
    sample_texts = [
        {
            "text": """Il machine learning è un ramo dell'intelligenza artificiale che si concentra 
            sullo sviluppo di algoritmi e modelli statistici che permettono ai computer di migliorare le 
            loro prestazioni su un compito specifico attraverso l'esperienza, senza essere esplicitamente 
            programmati. Questo approccio è particolarmente utile quando la programmazione di regole 
            esplicite sarebbe troppo complessa o impraticabile. Le applicazioni del machine learning 
            includono il riconoscimento vocale, la visione artificiale, i motori di raccomandazione, 
            i sistemi di traduzione automatica e molti altri campi. I principali vantaggi del machine 
            learning includono: 1) Automazione di compiti complessi, 2) Capacità di scoprire pattern nascosti, 
            3) Adattabilità a nuovi dati, 4) Scalabilità per grandi volumi di informazioni.""",
            "summary": "Il machine learning è un ramo dell'IA che permette ai computer di migliorare tramite esperienza. I principali vantaggi sono automazione di compiti complessi, scoperta di pattern nascosti, adattabilità e scalabilità."
        },
        {
            "text": """La sicurezza informatica è un campo in continua evoluzione che si concentra sulla 
            protezione di sistemi, reti e dati da accessi non autorizzati, attacchi e danni. Con l'aumento 
            della digitalizzazione, la sicurezza informatica è diventata fondamentale per organizzazioni 
            di ogni dimensione. Le strategie per migliorare la sicurezza informatica includono: 
            implementazione dell'autenticazione a più fattori, formazione regolare del personale sui rischi 
            di sicurezza, mantenimento di software e sistemi aggiornati, crittografia dei dati sensibili, 
            backup regolari, monitoraggio continuo delle attività di rete, implementazione di firewall 
            e sistemi di rilevamento delle intrusioni, sviluppo e applicazione di politiche di sicurezza 
            rigorose, e valutazioni periodiche della vulnerabilità e test di penetrazione.""",
            "summary": "La sicurezza informatica protegge sistemi e dati da accessi non autorizzati. Per migliorarla: usare autenticazione multi-fattore, formare il personale, mantenere sistemi aggiornati, crittografare dati e implementare monitoraggio continuo."
        },
        {
            "text": """I database SQL (Structured Query Language) e NoSQL rappresentano due approcci 
            fondamentalmente diversi all'archiviazione e alla gestione dei dati. I database SQL sono 
            relazionali, utilizzano tabelle con schemi predefiniti e relazioni ben strutturate tra entità. 
            Sono ottimizzati per operazioni ACID (Atomicità, Coerenza, Isolamento, Durabilità) e 
            garantiscono l'integrità referenziale. D'altra parte, i database NoSQL sono non relazionali, 
            offrono schemi flessibili e sono progettati per gestire grandi volumi di dati distribuiti. 
            Esistono vari tipi di database NoSQL, tra cui orientati ai documenti (MongoDB), chiave-valore 
            (Redis), a colonne (Cassandra) e a grafo (Neo4j). La scelta tra SQL e NoSQL dipende dalle 
            esigenze specifiche del progetto, come la struttura dei dati, i requisiti di scala, la 
            coerenza necessaria e il tipo di query previste.""",
            "summary": "I database SQL sono relazionali con schemi predefiniti, ottimizzati per operazioni ACID. I NoSQL sono non relazionali con schemi flessibili, progettati per gestire grandi volumi di dati distribuiti in vari formati come documenti, chiave-valore, colonne o grafi."
        }
    ]
    
    # Create chunks with metadata and summaries
    chunks = []
    for i, item in enumerate(sample_texts):
        chunks.append({
            "id": f"chunk_{i+1}",
            "text": item["text"],
            "summary": item["summary"],
            "segments": [{"type": "header", "text": f"Sezione {i+1}"}]
        })
    
    return chunks

if __name__ == "__main__":
    main()