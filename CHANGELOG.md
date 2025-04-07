# Changelog Raggiro

Questo documento riassume le principali modifiche funzionali apportate al progetto Raggiro, con focus su OCR, elaborazione del testo, supporto multilingua e miglioramenti delle prestazioni.

## Elaborazione e OCR dei Documenti

- **Correzione problema di scope delle variabili nell'elaborazione OCR**: Risolto il problema di accesso a variabili eliminate che causava errori durante l'elaborazione delle pagine (`pix` variable).
- **Elaborazione multi-pagina migliorata**: Implementato un sistema che assicura l'estrazione di tutte le pagine nei documenti PDF, risolvendo il problema dell'estrazione limitata a una singola pagina.
- **OCR forzato per tutti i PDF**: Migliorato il sistema per applicare OCR a tutte le pagine PDF indipendentemente dalla presenza di un layer di testo, garantendo uniformità nei risultati.
- **Ottimizzazione per documenti voluminosi**: Aggiunto supporto specifico per documenti di grandi dimensioni con elaborazione a batch per gestire efficacemente la memoria.
- **Metriche di conteggio caratteri OCR**: Implementate metriche dettagliate per monitorare e visualizzare il numero di caratteri estratti per pagina.

## Supporto Multilingua e Correzione del Testo

- **Rilevamento automatico della lingua**: Implementato sistema di rilevamento automatico della lingua dei documenti per ottimizzare l'OCR e la correzione ortografica.
- **Sincronizzazione OCR-Spelling**: Allineamento tra la lingua utilizzata per l'OCR e quella per la correzione ortografica per risultati più coerenti.
- **Supporto migliorato per l'italiano**: Ottimizzata la gestione dei caratteri accentati e apostrofi italiani nelle operazioni di correzione del testo.
- **Correzione errori di sintassi nelle regex**: Risolti problemi di sintassi nelle espressioni regolari utilizzate per identificare e correggere il testo.
- **Dizionario italiano standard**: Integrato dizionario italiano completo per una correzione ortografica più accurata.

## Prestazioni e Ottimizzazioni

- **Performance della correzione ortografica**: Miglioramento significativo delle prestazioni nella correzione ortografica per documenti di grandi dimensioni.
- **Ottimizzazione della memoria**: Implementata gestione ottimizzata della memoria durante l'elaborazione OCR per prevenire problemi con file di grandi dimensioni.
- **Elaborazione a batch**: Introdotta elaborazione OCR a batch per bilanciare l'uso della memoria e migliorare la stabilità del sistema.
- **Rilevamento robusto della root del repository**: Implementato un sistema più affidabile per identificare la root del repository, migliorando la portabilità.

## Interfaccia e Visualizzazione

- **Vista di confronto documenti**: Corretta la visualizzazione del confronto tra testo originale e testo corretto, assicurando che entrambi i testi siano disponibili.
- **Generazione PDF migliorata**: Aggiunta capacità di generazione PDF per documenti OCR con gestione robusta degli errori.
- **Tab UI per OCR e Correzione**: Aggiunta una nuova sezione nell'interfaccia dedicata alle funzionalità OCR e correzione ortografica.
- **Chunking migliorato per documenti OCR**: Ottimizzato il processo di chunking specificamente per documenti elaborati con OCR.

## Documentazione e Configurazione

- **Documentazione spaCy completa**: Aggiunta documentazione dettagliata per l'integrazione con spaCy e i modelli linguistici.
- **Aggiornamento README**: Integrate nel README informazioni aggiornate sulle funzionalità OCR e correzione ortografica.
- **Dipendenze complete**: Aggiunte tutte le dipendenze necessarie, incluso supporto specifico per Windows con python-magic-bin.
- **Configurazione per supporto multilingua**: Aggiornato il file config.toml per supportare elaborazione multilingua e risolvere problemi di configurazione.

## Sistema RAG (Retrieval-Augmented Generation)

- **Supporto multilingua RAG**: Migliorato il sistema RAG per rispettare la lingua del documento e della query.
- **Impostazioni Ollama aggiornate**: Modificate le impostazioni di Ollama secondo requisiti specifici.
- **Riorganizzazione degli script**: Riorganizzati gli script in cartelle dedicate e aggiunta correzione ortografica OCR.