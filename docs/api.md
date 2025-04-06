# Riferimento API

Puoi utilizzare Raggiro programmaticamente nel tuo codice Python.

## API di base

### Elaborazione dei documenti

```python
from raggiro.processor import DocumentProcessor
from raggiro.utils.config import load_config

# Carica la configurazione
config = load_config()

# Elabora un documento
processor = DocumentProcessor(config)
result = processor.process_file("document.pdf", "output_dir")

# Verifica il successo e accedi ai risultati
if result["success"]:
    print(f"Documento elaborato: {result['file_path']}")
    print(f"Metadata: {result['metadata']}")
    print(f"Dimensione del testo: {result['text_size']} caratteri")
    print(f"Segmenti: {result['segment_count']}")
    print(f"Chunks: {result['chunk_count']}")
else:
    print(f"Errore: {result.get('error', 'Errore sconosciuto')}")
```

### Elaborazione di directory

```python
# Elabora una directory di documenti
process_result = processor.process_directory("input_dir", "output_dir", recursive=True)

if process_result["success"]:
    print(f"Elaborati {process_result['summary']['total_files']} documenti")
    print(f"Tasso di successo: {process_result['summary']['success_rate']}%")
```

## API RAG

### Creazione di indici vettoriali

```python
from raggiro.rag.indexer import VectorIndexer

# Inizializza l'indicizzatore vettoriale
indexer = VectorIndexer(config)

# Indicizza una directory di documenti elaborati
index_result = indexer.index_directory("processed_docs")

if index_result["success"]:
    print(f"Indicizzati {index_result['summary']['total_chunks_indexed']} chunks")
    
    # Salva l'indice
    indexer.save_index("index_dir")
    print(f"Indice salvato in {index_dir}")
```

### Interrogazione della pipeline RAG

```python
from raggiro.rag.pipeline import RagPipeline

# Inizializza la pipeline RAG
pipeline = RagPipeline(config)

# Carica un indice esistente
pipeline.retriever.load_index("index_dir")

# Interroga la pipeline
result = pipeline.query("Qual è il concetto principale di questo documento?")

if result["success"]:
    print("\nRisposta:")
    print(result["response"])
    print(f"\nUtilizzati {result['chunks_used']} chunks per questa risposta")
    
    # Accedi ai dettagli del processo di query
    if "rewritten_query" in result:
        print(f"Query originale: {result['original_query']}")
        print(f"Query riscritta: {result['rewritten_query']}")
        
    # Accedi ai chunks recuperati
    if "retrieved_chunks" in result:
        print("\nChunks recuperati:")
        for i, chunk in enumerate(result["retrieved_chunks"]):
            print(f"Chunk {i+1}: {chunk['text'][:100]}... (score: {chunk['score']:.2f})")
else:
    print(f"Errore: {result.get('error', 'Errore sconosciuto')}")
```

## Componenti API

### Estrazione del testo

```python
from raggiro.core.extractor import TextExtractor

# Inizializza l'estrattore di testo
extractor = TextExtractor(config)

# Estrai testo da un file PDF
result = extractor.extract("document.pdf")

if result["success"]:
    print(f"Testo estratto: {len(result['text'])} caratteri")
    print(f"Metodo di estrazione: {result['extraction_method']}")
    print(f"Tempo di estrazione: {result['extraction_time_ms']}ms")
```

### Segmentazione del testo

```python
from raggiro.core.segmenter import TextSegmenter

# Inizializza il segmentatore di testo
segmenter = TextSegmenter(config)

# Segmenta il testo
result = segmenter.segment(text)

if result["success"]:
    print(f"Segmenti creati: {len(result['segments'])}")
    print(f"Distribuzione tipo segmenti: {result['segment_types']}")
    
    # Accedi ai segmenti
    for i, segment in enumerate(result["segments"]):
        print(f"Segmento {i+1} ({segment['type']}): {segment['text'][:50]}...")
```

### Esempi di API completi

```python
import os
from pathlib import Path
from raggiro.processor import DocumentProcessor
from raggiro.rag.indexer import VectorIndexer
from raggiro.rag.pipeline import RagPipeline
from raggiro.utils.config import load_config

# Directory
input_dir = "documents"
output_dir = "processed"
index_dir = "index"

# Crea directory
os.makedirs(output_dir, exist_ok=True)
os.makedirs(index_dir, exist_ok=True)

# Carica configurazione
config = load_config()

# 1. Elabora documenti
print("Elaborazione documenti...")
processor = DocumentProcessor(config)
process_result = processor.process_directory(input_dir, output_dir, recursive=True)

if process_result["success"]:
    print(f"Elaborati {process_result['summary']['total_files']} documenti")
    print(f"Tasso di successo: {process_result['summary']['success_rate']}%")
    
    # 2. Indicizza i documenti elaborati
    print("\nIndicizzazione documenti...")
    indexer = VectorIndexer(config)
    index_result = indexer.index_directory(output_dir)
    
    if index_result["success"]:
        print(f"Indicizzati {index_result['summary']['total_chunks_indexed']} chunks")
        
        # Salva l'indice
        indexer.save_index(index_dir)
        print(f"Indice salvato in {index_dir}")
        
        # 3. Configura la pipeline RAG per le query
        print("\nConfigurazione pipeline RAG...")
        pipeline = RagPipeline(config)
        pipeline.retriever.load_index(index_dir)
        
        # 4. Interroga la pipeline
        while True:
            query = input("\nInserisci la tua query (o 'quit' per uscire): ")
            if query.lower() in ['quit', 'exit', 'q']:
                break
                
            print("\nElaborazione query...")
            result = pipeline.query(query)
            
            if result["success"]:
                print("\nRisposta:")
                print(result["response"])
                print(f"\nUtilizzati {result['chunks_used']} chunks per questa risposta")
            else:
                print(f"Errore: {result.get('error', 'Errore sconosciuto')}")
    else:
        print(f"Indicizzazione fallita: {index_result.get('error', 'Errore sconosciuto')}")
else:
    print(f"Elaborazione fallita: {process_result.get('error', 'Errore sconosciuto')}")
```