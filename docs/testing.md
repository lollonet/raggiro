# Testing e valutazione

Raggiro include strumenti completi per il testing e la valutazione del tuo sistema RAG, accessibili sia tramite interfacce a riga di comando che tramite la GUI.

## Running dei test di valutazione

### Testing da riga di comando

```bash
# Esegui valutazioni promptfoo
raggiro test-rag --prompt-set test_prompts/kenny_werner.yaml --output test_results

# Sovrascrivi le impostazioni di Ollama durante l'esecuzione dei test
raggiro test-rag --prompt-set test_prompts/scrum_guide.yaml --output test_results --ollama-url http://localhost:11434

# Testa il chunking semantico con un documento specifico
./run_semantic_chunking_test.sh documento.pdf --output test_output

# Confronta diverse strategie di chunking
./run_test_comparison.sh documento.pdf --strategies size semantic hybrid
```

### Testing GUI con Streamlit

L'interfaccia Streamlit offre un ambiente user-friendly per il testing RAG:

1. **Scheda Test RAG**:
   - Seleziona documenti elaborati da testare
   - Scegli tra prompt di test predefiniti o creane di personalizzati
   - Esegui test con tracciamento del progresso in tempo reale
   - Visualizza i risultati immediatamente dopo il completamento del test
   - Rilevamento automatico dell'installazione di PromptFoo con messaggi utili
   - Esecuzione completa del test con log e indicatori di progresso

2. **Scheda Visualizza risultati**:
   - Sfoglia la cronologia dei test su diverse esecuzioni
   - Confronta i risultati da diverse strategie di chunking
   - Visualizza metriche e statistiche dettagliate
   - Visualizza graficamente le prestazioni dei test

## Testing del chunking semantico

La funzionalità di chunking semantico può essere testata e analizzata utilizzando gli script di test inclusi.

Per facilità d'uso, sono forniti script helper:

```bash
# Usa lo script helper (consigliato)
./scripts/testing/run_semantic_chunking_test.sh document.pdf --output test_output

# Testa con query personalizzate e impostazioni Ollama
./scripts/testing/run_semantic_chunking_test.sh document.pdf --queries "Qual è l'argomento principale?" "Riassumi i punti chiave" --ollama-url http://localhost:11434 --rewriting-model llama3 --generation-model mistral

# Specifica il numero di chunk da recuperare per ogni query
./scripts/testing/run_semantic_chunking_test.sh document.pdf --top-k 5
```

## Confronto tra strategie di chunking

Puoi confrontare diverse strategie di chunking per trovare l'approccio più efficace per i tuoi documenti. Usa lo script helper fornito:

```bash
# Confronta tutte le strategie disponibili usando lo script helper (consigliato)
./scripts/testing/run_test_comparison.sh documento.pdf

# Confronta solo strategie specifiche con impostazioni Ollama personalizzate
./scripts/testing/run_test_comparison.sh documento.pdf --strategies size hybrid --ollama-url http://localhost:11434 --rewriting-model llama3 --generation-model mistral

# Testa con query specifiche e directory di output
./scripts/testing/run_test_comparison.sh documento.pdf --queries "Qual è l'argomento principale?" --output miei_risultati_test
```

## Configurazioni di test personalizzate

Puoi creare configurazioni di test personalizzate per i tuoi documenti specifici:

```yaml
# Esempio di configurazione personalizzata (my_prompts.yaml)
prompts:
  - "Qual è la tesi principale di questo documento?"
  - "Quali prove supportano l'argomento principale?"
  - "Riassumi le conclusioni in 3 punti."

variants:
  - name: "chunking_semantico"
    description: "Test con chunking semantico"
    config:
      chunking_strategy: "semantic"
      
  - name: "chunking_dimensione"
    description: "Test con chunking basato sulla dimensione"
    config:
      chunking_strategy: "size"

tests:
  - description: "Estrazione informazioni di base"
    assert:
      - type: "contains-any"
        value: ["argomento", "documento", "informazione"]
```

Per eseguire test con configurazioni personalizzate:

```bash
# Test con configurazione personalizzata
python -m raggiro.testing.promptfoo_runner path/to/custom_prompts.yaml test_output
```

## Esempio di risultati di valutazione

```json
{
  "summary": {
    "pass": 18,
    "fail": 2,
    "total": 20,
    "pass_rate": 90.00
  },
  "prompts": [
    {
      "prompt": "Qual è l'argomento principale di questo documento?",
      "results": [
        {
          "pass": true,
          "score": 0.92,
          "response": "L'argomento principale di questo documento è l'intelligenza artificiale generativa e il suo impatto sul mondo del lavoro entro il 2025. Il documento si concentra in particolare sui Large Language Models, l'IA multimodale e le applicazioni dell'IA nel contesto lavorativo. [Fonte: Sezione introduttiva, paragrafo 1]"
        }
      ]
    }
  ]
}
```

## Testing programmatico

Puoi utilizzare anche gli strumenti di testing programmaticamente:

```python
from raggiro.testing.promptfoo_runner import run_tests

# Esegui test con set di prompt personalizzato e directory di output
results = run_tests(
    prompt_file="test_prompts/my_custom_prompts.yaml", 
    output_dir="test_output",
    index_dir="directory_indice"  # Percorso opzionale all'indice vettoriale
)

# Verifica se PromptFoo è installato e i test sono riusciti
if not results["success"]:
    error = results.get("error", "Errore sconosciuto")
    if "promptfoo not installed" in error:
        print("PromptFoo non è installato. Installa con: npm install -g promptfoo")
        print("Poi assicurati di avviarlo con comando 'uv run' se in ambiente virtuale uv")
    else:
        print(f"Test fallito: {error}")
else:
    print(f"Test eseguiti: {results['tests_run']}")
    print(f"Risultati salvati in: {results.get('output_file')}")
```

## Log di elaborazione e statistiche

Durante l'elaborazione dei documenti, Raggiro genera file di log dettagliati:

```
output_dir/
├── logs/
│   ├── raggiro_20250405_123045.log        # Log di elaborazione principale
│   ├── processed_files_20250405_123045.csv # CSV di tutti i file elaborati
│   └── processing_summary_20250405_123045.json # Statistiche di elaborazione
```

### Statistiche di elaborazione con analisi dei metadati

Raggiro genera un riepilogo JSON con statistiche dettagliate sull'esecuzione dell'elaborazione, inclusa la qualità dell'estrazione dei metadati:

```json
{
  "start_time": "2025-04-05T12:30:45.123456",
  "end_time": "2025-04-05T12:35:28.789012",
  "total_files": 24,
  "successful_files": 22,
  "failed_files": 2,
  "success_rate": 91.67,
  "file_types": {
    ".pdf": 15,
    ".docx": 5,
    ".txt": 2,
    ".pptx": 1,
    ".xlsx": 1
  },
  "extraction_methods": {
    "pdf": 10,
    "pdf_ocr": 5,
    "docx": 5,
    "text": 2
  },
  "languages": {
    "en": 18,
    "es": 2,
    "fr": 2
  },
  "topics": {
    "financial": 8,
    "technical": 6,
    "legal": 4,
    "report": 4
  },
  "metadata_completeness": {
    "title": 95.45,
    "author": 81.82,
    "date": 86.36,
    "language": 100.00,
    "topics": 77.27
  },
  "errors": {
    "Unsupported file type: .pptx": 1,
    "Failed to extract text: Document is password protected": 1
  }
}
```

### Metriche della pipeline RAG

Quando utilizzi la pipeline RAG, puoi raccogliere metriche sulle prestazioni delle query:

```python
# Interroga la pipeline RAG e raccogli metriche
result = pipeline.query("Quali sono i vantaggi principali?", collect_metrics=True)

# Accedi alle metriche
print(f"Tempo di elaborazione query: {result['metrics']['query_time_ms']}ms")
print(f"Chunk recuperati: {result['metrics']['chunks_retrieved']}")
print(f"Similarità del chunk migliore: {result['metrics']['top_similarity']:.2f}")
```