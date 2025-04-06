# Riferimento CLI

Raggiro fornisce un'interfaccia a riga di comando completa con diversi comandi per l'elaborazione dei documenti, l'accesso alla GUI e il testing RAG.

## Comando principale

```bash
# Mostra aiuto e comandi disponibili
raggiro --help

# Controlla la versione
raggiro --version
```

## Comando di elaborazione documenti

Il comando `process` è la funzionalità principale per l'elaborazione dei documenti:

```bash
# Mostra aiuto per il comando process
raggiro process --help

# Elabora un singolo file
raggiro process document.pdf --output output_dir

# Elabora una directory di documenti
raggiro process documents/ --output output_dir

# Elabora ricorsivamente (predefinito) o non ricorsivamente
raggiro process documents/ --output output_dir --recursive
raggiro process documents/ --output output_dir --no-recursive

# Abilita OCR per documenti scansionati (predefinito: abilitato)
raggiro process document.pdf --output output_dir --ocr
raggiro process document.pdf --output output_dir --no-ocr

# Specifica i formati di output (predefinito: markdown e json)
raggiro process document.pdf --output output_dir --format markdown --format json

# Esegui in modalità dry-run (nessun file scritto)
raggiro process document.pdf --output output_dir --dry-run

# Imposta il livello di logging (predefinito: info)
raggiro process documents/ --output output_dir --log-level debug

# Usa un file di configurazione personalizzato
raggiro process documents/ --output output_dir --config my_config.toml
```

## Interfacce GUI

Raggiro include sia interfacce GUI basate sul web (Streamlit) che basate sul terminale (Textual):

```bash
# Mostra aiuto per il comando GUI
raggiro gui --help

# Avvia la GUI Streamlit (basata sul web)
raggiro gui
# Oppure esegui direttamente con Streamlit per una migliore integrazione del browser
streamlit run $(which raggiro)

# Avvia la GUI Textual (basata sul terminale)
raggiro gui --tui
```

## Comandi di testing RAG

Il comando `test-rag` consente il testing automatizzato della tua pipeline RAG, utilizzando le impostazioni dal tuo file di configurazione TOML:

```bash
# Mostra aiuto per il comando test-rag
raggiro test-rag --help

# Esegui test con una configurazione promptfoo
raggiro test-rag --prompt-set tests/prompts.yaml

# Specifica la directory di output per i risultati del test
raggiro test-rag --prompt-set tests/prompts.yaml --output test_results

# Sovrascrivi l'URL Ollama dalla riga di comando (sovrascrive la configurazione TOML)
raggiro test-rag --prompt-set tests/prompts.yaml --ollama-url http://localhost:11434

# Specifica modelli diversi per la riscrittura e la generazione (sovrascrive la configurazione TOML)
raggiro test-rag --prompt-set tests/prompts.yaml --rewriting-model llama3 --generation-model mistral
```

## Script di supporto

Raggiro include script helper per semplificare i test e altre operazioni comuni:

```bash
# Verifica l'installazione di PromptFoo
./verify_promptfoo.sh

# Installa PromptFoo (preferibilmente a livello utente)
./install_promptfoo.sh

# Esegui test del chunking semantico
./run_semantic_chunking_test.sh documento.pdf --output test_output

# Confronta le strategie di chunking
./run_test_comparison.sh documento.pdf --strategies size semantic hybrid
```