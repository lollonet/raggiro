# Pipeline RAG

Raggiro fornisce una pipeline RAG (Retrieval-Augmented Generation) completa, che integra il recupero di informazioni da documenti con la generazione di risposte utilizzando modelli linguistici.

## Componenti della pipeline RAG

La pipeline RAG include:

- **VectorIndexer**: Crea e gestisce indici vettoriali
- **VectorRetriever**: Recupera chunk rilevanti dall'indice
- **QueryRewriter**: Migliora le query per un recupero migliore
- **ResponseGenerator**: Genera risposte dai chunk recuperati
- **RagPipeline**: Orchestra l'intero flusso di lavoro RAG

## Supporto multilingua

Raggiro fornisce un supporto completo per documenti in diverse lingue, rilevando e preservando la lingua originale in tutta la pipeline:

```python
# La pipeline rileva automaticamente la lingua del documento
result = pipeline.query("Qual è il tema principale del documento?")  # Query in italiano

# La risposta sarà nella stessa lingua della query
print(result["response"])  # Risposta in italiano

# È possibile specificare esplicitamente la lingua del documento
result = pipeline.query("What is the main topic?", document_language="it")  # Forza risposte in italiano
```

Il sistema:
1. Rileva la lingua della query
2. Verifica la coerenza con la lingua del documento (se specificata)
3. Genera risposte nella lingua appropriata

## Esempio: Utilizzo della pipeline RAG

```python
from raggiro.rag.pipeline import RagPipeline
from raggiro.utils.config import load_config

# Carica la configurazione
config = load_config()

# Inizializza la pipeline RAG
pipeline = RagPipeline(config)

# Carica un indice creato in precedenza
pipeline.retriever.load_index("directory_indice")

# Interroga la pipeline
result = pipeline.query("Quali sono i vantaggi principali di questo prodotto?")

# Visualizza la risposta
print(result["response"])

# Ottieni informazioni sul processo di query
if "rewritten_query" in result:
    print(f"Query originale: {result['original_query']}")
    print(f"Query riscritta: {result['rewritten_query']}")

print(f"Chunk utilizzati: {result['chunks_used']}")
```

## Ottimizzazione del recupero

Raggiro supporta diverse strategie di chunking per ottimizzare il recupero delle informazioni:

### Chunking basato sulla dimensione

Suddivide il documento in chunk di dimensione fissa, ideale per documenti ben strutturati.

```toml
[segmentation]
chunking_strategy = "size"
max_chunk_size = 500  # Dimensione in caratteri per ogni chunk
chunk_overlap = 100   # Sovrapposizione tra chunk
```

### Chunking semantico

Suddivide il documento in base al significato semantico, migliorando il raggruppamento delle informazioni correlate.

```toml
[segmentation]
chunking_strategy = "semantic"
semantic_similarity_threshold = 0.65  # Soglia per la similarità semantica
```

### Chunking ibrido

Combina chunking basato su dimensione e semantica per un equilibrio ottimale.

```toml
[segmentation]
chunking_strategy = "hybrid"
max_chunk_size = 500
chunk_overlap = 100
semantic_similarity_threshold = 0.65
```

## Riscrittura delle query

La riscrittura delle query è una funzionalità chiave che migliora la qualità del recupero:

```python
# Esempio di riscrittura di query
original_query = "Vantaggi del prodotto"
rewritten_query = pipeline.rewriter.rewrite(original_query)
print(f"Query originale: {original_query}")
print(f"Query riscritta: {rewritten_query}")
```

La riscrittura delle query:
1. Espande query brevi o ambigue
2. Aggiunge sinonimi e termini correlati
3. Migliora la specificità della query
4. Preserva la lingua originale della query

## Generazione di risposte

Il componente di generazione di risposte integra i chunk recuperati in risposte coerenti:

```python
# Configurazione avanzata del generatore
from raggiro.rag.generator import ResponseGenerator

generator = ResponseGenerator(config)

# Genera una risposta da un elenco di chunk
chunks = retriever.retrieve(query, top_k=5)
response = generator.generate(query, chunks)

print(response)
```

Caratteristiche di generazione:
- Citazione automatica delle fonti
- Mantenimento della coerenza del linguaggio
- Gestione di informazioni contraddittorie
- Calibrazione della lunghezza della risposta