# Contesto di sviluppo Raggiro per Claude

Questo documento mantiene il contesto di sviluppo attuale e le decisioni prese per aiutare Claude a riprendere il lavoro in sessioni future.

## Panoramica del progetto

Raggiro è un sistema RAG (Retrieval-Augmented Generation) in Python con focus su:
- Elaborazione di documenti multilingua (principalmente italiano/inglese)
- OCR e correzione ortografica di documenti scansionati
- Chunking semantico e indicizzazione di documenti
- Interfaccia Streamlit per interazione utente

## Stato attuale - 9 Aprile 2025

### Struttura dei branch

1. **`main`**: Contiene tentativi di migrazione a `uv` e correzioni di compatibilità
2. **`stable-pre-uv`**: Versione stabile che usa pip/virtualenv (pre-uv) - BRANCH DI LAVORO PRINCIPALE
3. **`docling-fork`**: Branch per la futura implementazione di DocLing
4. **`backup-uv-changes`**: Backup della versione con tentativi di migrazione a uv

### Problemi critici

1. **Incompatibilità di spaCy con uv**:
   - I modelli linguistici spaCy non sono disponibili su PyPI
   - Causano errori durante l'installazione con `uv`
   - Richiedono installazione manuale separata

2. **Soluzioni implementate e abbandonate**:
   - Tentativo di modificare pyproject.toml per rimuovere dipendenze dirette dai modelli
   - Miglioramento della resilienza di segmenter.py con meccanismi di fallback
   - Aggiornamento della documentazione di installazione

3. **Decisione attuale**:
   - Tornare alla versione pre-uv (branch `stable-pre-uv`)
   - Consolidare questa versione stabile
   - Pianificare la migrazione da spaCy a soluzioni alternative

### Alternative di migrazione

Sono state identificate due possibili strade da seguire:

1. **Migrazione a Stanza (breve termine - 2-4 settimane)**:
   - Libreria NLP di Stanford che offre funzionalità simili a spaCy
   - Gestione modelli integrata (download automatico)
   - Compatibilità completa con gestori di pacchetti

2. **Fork per DocLing (lungo termine - 2-3 mesi)**:
   - Libreria documentale completa che sostituirebbe diverse dipendenze attuali
   - Unificherebbe parsing PDF, OCR, e analisi strutturale
   - Architettura più moderna e integrata con framework RAG

La documentazione dettagliata di queste alternative è disponibile in `docs/future_considerations/migration_alternatives.md`.

## Prossimi passaggi

1. **Priorità immediata**:
   - Verificare e consolidare il branch `stable-pre-uv`
   - Assicurarsi che l'installazione con pip/virtualenv funzioni correttamente
   - Risolvere eventuali problemi di compatibilità rimasti

2. **Breve termine**:
   - Esplorare in dettaglio l'alternativa Stanza
   - Creare un prototipo di implementazione
   - Valutare prestazioni e qualità dell'analisi rispetto a spaCy

3. **Medio termine**:
   - Iniziare lo sviluppo sperimentale del fork DocLing
   - Testare l'elaborazione documentale su campioni rappresentativi

---

## Sessione del 9 Aprile 2025

### Progressi
- Identificato problema con spaCy e uv
- Creato branch stabile pre-uv
- Documentato alternative di migrazione 
- Organizzato repository con branch dedicati

### Decisioni tecniche
- Abbandonato temporaneamente `uv` per tornare a pip/virtualenv
- Deciso di esplorare Stanza come soluzione a breve termine
- Pianificato fork DocLing come soluzione a lungo termine

### Stato attuale
- Branch attivo consigliato: `stable-pre-uv`
- File principali modificati:
  - docs/future_considerations/migration_alternatives.md (nuovo)
  - CLAUDE_CONTEXT.md (nuovo)

### Prossimi passaggi da discutere
- Implementare fix immediati per rendere la versione stable-pre-uv pienamente funzionale
- Decidere se procedere subito con l'esplorazione di Stanza
- Valutare se creare un prototipo DocLing di base