# Istruzioni per Claude

Questo documento contiene istruzioni personalizzate per ottimizzare le sessioni di Claude Code in questo progetto. Claude dovrebbe leggere queste istruzioni all'inizio di ogni sessione.

## Contesto del Progetto Raggiro

Raggiro è un sistema RAG (Retrieval-Augmented Generation) in Python con focus su:
- Elaborazione di documenti multilingua (principalmente italiano/inglese)
- OCR e correzione ortografica di documenti scansionati
- Chunking semantico e indicizzazione di documenti
- Interfaccia Streamlit per interazione utente

## Convenzioni di Codice

Quando lavori sul codice di questo progetto, segui queste linee guida:

1. **Stile di codice**: Segui PEP 8 per il codice Python. Utilizza docstring complete con formato Google-style.
2. **Lingue dei commenti**: Usa l'italiano per i commenti inline e l'inglese per la documentazione delle funzioni.
3. **Logging**: Utilizza sempre il sistema di logging del progetto anziché print statements.
4. **Gestione errori**: Implementa gestione degli errori robusta con messaggi dettagliati.
5. **Type hinting**: Usa i type hints di Python per tutte le funzioni e metodi.
6. **Test**: Considera l'implementazione di test per ogni nuova funzionalità.

## Convenzioni di Documentazione

Per la documentazione:

1. **Lingua**: Usa l'italiano per documentazione destinata agli utenti.
2. **Formato**: Utilizza Markdown per la documentazione.
3. **Esempi**: Includi esempi di codice per illustrare l'utilizzo delle funzionalità.
4. **Struttura**: Organizza la documentazione con intestazioni chiare e numerazione logica.

## Procedure di Qualità

Prima di finalizzare qualsiasi modifica:

1. **Linting**: Esegui `ruff check .` per verificare la qualità del codice.
2. **Test**: Esegui `pytest` per assicurarsi che tutti i test passino.
3. **Documentazione**: Assicurati che la documentazione sia aggiornata con le nuove modifiche.
4. **Commit**: Usa messaggi di commit descrittivi che spiegano il "perché" delle modifiche.

## Dipendenze Specifiche

Il progetto dipende da:
- OCR: `pytesseract`, `Pillow`, `PyMuPDF` (fitz)
- NLP: `spacy`, `langdetect`, `sentence-transformers`
- File: `python-magic`, `python-docx`, `openpyxl`
- UI: `streamlit`, `textual`
- Integrazione LLM: `openai`, `langchain`

## Comandi Utili

- Avvio interfaccia Streamlit: `python run_streamlit.sh`
- Test RAG: `python test_rag.py`
- Test completi: `cd test_prompts && ./run_all_tests.sh`

## Caratteristiche in Corso di Sviluppo

Aree attualmente in sviluppo:
- Miglioramento OCR multilingua
- Ottimizzazione della correzione ortografica
- Visualizzazione del confronto testo originale vs corretto
- Integrazione BERT per embedding semantici avanzati

## Problemi Noti

- L'elaborazione OCR può essere lenta su documenti di grandi dimensioni
- La correzione ortografica in italiano richiede attenzione particolare con caratteri accentati
- Il formato PDF di confronto può presentare problemi con caratteri non standard