# Contesto di sviluppo - Branch docling-fork

Questo documento mantiene il contesto di sviluppo per il branch `docling-fork`, dedicato all'implementazione dell'integrazione con DocLing.

## Panoramica del branch

`docling-fork` è un branch di sviluppo sperimentale creato dalla base stabile pre-uv, con l'obiettivo di riprogettare Raggiro per usare DocLing come tecnologia centrale. Questo rappresenta una decisione strategica di lungo termine (3-6 mesi) per semplificare l'architettura e migliorare le capacità del sistema.

## Caratteristiche e obiettivi di DocLing

[DocLing](https://github.com/docling-project/docling) è una libreria avanzata per l'elaborazione documentale che unifica varie funzionalità attualmente fornite da componenti separati:

- **Parsing multiformat**: Supporto nativo per PDF, DOCX, XLSX, HTML e immagini
- **Comprensione avanzata PDF**: Analisi di layout, ordine di lettura e struttura tabelle
- **OCR integrato**: Capacità OCR superiori a Tesseract
- **Struttura semantica**: Comprensione della struttura documentale
- **Integrazione RAG nativa**: Connessione diretta con framework RAG

## Piano di implementazione

### Fase 1: Setup e prototipazione (in corso)
- Creare un ambiente di base con DocLing
- Prototipare le funzionalità principali
- Valutare la compatibilità con i requisiti di Raggiro

### Fase 2: Refactoring dei componenti core
- Sostituire l'estrattore di testo esistente con DocLing
- Sostituire il segmenter con l'API di struttura DocLing
- Migrare la correzione ortografica al preprocessing DocLing

### Fase 3: Integrazione con RAG
- Aggiornare pipeline RAG per usare documenti DocLing
- Implementare retriever avanzato con DocLing
- Sviluppare nuove capacità di ricerca semantica

### Fase 4: Interfaccia utente e documentazione
- Aggiornare l'interfaccia Streamlit
- Creare visualizzazioni avanzate della struttura documentale
- Aggiornare la documentazione con le nuove funzionalità

### Fase 5: Testing e finalizzazione
- Eseguire test comparativi rispetto alla versione originale
- Ottimizzare le prestazioni
- Preparare per il rilascio

## Vantaggi attesi della migrazione

- **Riduzione codebase**: -30-40% righe di codice grazie all'API unificata
- **Miglioramento prestazioni**: +50-70% in accuratezza su documenti complessi
- **Tempo di sviluppo futuro**: -40% per nuove funzionalità
- **Manutenzione**: -60% problemi legati a dipendenze multiple

## Stato attuale - 9 Aprile 2025

- Branch creato dalla base stabile pre-uv
- Documento di pianificazione dettagliata creato (`docs/future_considerations/migration_alternatives.md`)
- Implementazione non ancora iniziata

## Prossimi passaggi

1. **Setup iniziale**:
   - Installare DocLing e creare ambiente di test
   - Creare il primo prototipo di elaborazione documentale
   - Verificare la compatibilità con i documenti di test esistenti

2. **Implementazioni prioritarie**:
   - Sostituzione dell'estrattore di testo esistente
   - Analisi della struttura documentale di base
   - Integrazione OCR iniziale

3. **Confronti e benchmark**:
   - Creare test di confronto con l'implementazione attuale
   - Misurare qualità dell'estrazione testo
   - Valutare preservazione struttura documentale

---

## Sessione del 9 Aprile 2025

### Progressi
- Creato branch dedicato a DocLing
- Definito piano di migrazione dettagliato
- Creato file di contesto specifico per il branch

### Decisioni tecniche
- Approccio "clean-room" partendo dalla base stabile
- Focus iniziale sulle funzionalità di estrazione documentale
- Pianificata implementazione graduale in 5 fasi

### File principali da modificare
- `raggiro/core/document_processor.py` (da creare)
- `raggiro/core/extractor.py` (da sostituire)
- `raggiro/core/segmenter.py` (da sostituire)
- `raggiro/rag/pipeline.py` (da adattare)

### Prossimi passaggi da discutere
- Decisione su quando iniziare l'implementazione concreta
- Prioritizzazione delle funzionalità DocLing da implementare per prime
- Definizione di metriche di successo per la valutazione delle prestazioni