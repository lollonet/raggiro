# Piano di sviluppo per completare i flussi di Raggiro

Questo documento descrive i flussi mancanti o incompleti nel progetto Raggiro e propone una roadmap per completarli.

## 1. Pipeline specializzate per diverse categorie di documenti

Attualmente è implementata solo la `TechnicalPipeline`. Implementare:

- **LegalPipeline**:
  - Ottimizzata per terminologia giuridica
  - Chunking basato su sezioni e articoli di legge
  - Riconoscimento citazioni legali e riferimenti normativi
  - OCR con pesi ottimizzati per formattazione documenti legali

- **AcademicPipeline**:
  - Riconoscimento di citazioni e riferimenti bibliografici
  - Handling speciale per equazioni e formule scientifiche
  - Supporto metadati specifici (abstract, keywords, citation)
  - Ottimizzazione per paper accademici e tesi

- **BusinessPipeline**:
  - Focus su estrazione KPI e dati finanziari
  - Riconoscimento e preservazione struttura tabellare
  - Supporto per estrazione dati da grafici e diagrammi
  - Ottimizzazione per report aziendali e presentazioni

- **NarrativePipeline**:
  - Chunking basato su capitoli/sezioni/paragrafi narrativi
  - Riconoscimento dialoghi e strutture narrative
  - Conservazione flusso narrativo nei chunk
  - Ottimizzazione per opere letterarie e contenuti creativi

## 2. Miglioramento OCR multilingua

Estendere il sistema OCR attuale con:

- **Rilevamento automatico lingua del documento**:
  - Pre-analisi del documento per determinare lingua principale
  - Supporto documenti multilingua con rilevamento di blocchi in lingue diverse
  - Switching automatico tesseract per lingua ottimale

- **Pre/post-processing specifico per italiano**:
  - Ottimizzazione handling caratteri accentati
  - Correzione automatica pattern comuni italiano (à → a', e' → è)
  - Dizionari specializzati per domini tecnici in italiano

- **Integrazione modelli OCR specializzati**:
  - Supporto per caratteri speciali in documenti tecnici
  - Ottimizzazione OCR per tabelle e diagrammi
  - Miglioramento riconoscimento layout per preservare struttura

## 3. Sistema di valutazione RAG

Implementare framework di valutazione per misurare qualità delle risposte:

- **Metriche automatiche**:
  - Integrazione in `rag/pipeline.py` di metriche standard (ROUGE, BLEU)
  - Calcolo rilevanza semantica rispetto alla query
  - Valutazione coerenza risposta e qualità dei riferimenti

- **Interfaccia feedback utente**:
  - Aggiungere alla GUI Streamlit feedback qualitativo
  - Sistema di rating per pertinenza e utilità risposte
  - Segnalazione risposta errate o fuorvianti

- **Sistema miglioramento continuo**:
  - Logging query problematiche per analisi
  - Ottimizzazione parametri retriever basata su feedback
  - Fine-tuning delle pipeline in base al dominio

## 4. Completamento GUI

Aggiungere alla GUI Streamlit:

- **Tab visualizzazione classificazione documento**:
  - Dashboard interattiva con risultato classificazione
  - Visualizzazione confidence scores e features rilevanti
  - Opzione override manuale categoria

- **Vista comparativa testo originale/corretto**:
  - Interfaccia diff per vedere correzioni applicate
  - Statistiche su errori corretti e tipologie
  - Opzione per accettare/rifiutare correzioni specifiche

- **Selezione manuale pipeline di elaborazione**:
  - Override classificazione automatica
  - Configurazione parametri specifici per pipeline
  - Comparazione risultati tra diverse pipeline

- **Dashboard statistiche elaborazione**:
  - Metriche prestazionali su tempi elaborazione
  - Statistiche qualità chunking e indicizzazione
  - Visualizzazione avanzata distribuzione chunk e copertura testo

## 5. Gestione persistente indici

Sviluppare sistema di gestione indici:

- **Versioning degli indici**:
  - Metadata su versione, data creazione, documenti fonte
  - Supporto rollback a versioni precedenti
  - Merge di indici da diverse fonti

- **Metadati avanzati**:
  - Informazioni su configurazione e pipeline utilizzata
  - Statistiche copertura e qualità embedding
  - Tracciamento origine documenti e chunks

- **Ottimizzazione spazio e prestazioni**:
  - Compressione indici per ridurre footprint
  - Pruning vettori meno rilevanti
  - Indicizzazione incrementale per aggiornamenti efficienti

## 6. Integrazione modelli avanzati di embedding

Estendere supporto oltre l'attuale "all-MiniLM-L6-v2":

- **Embedding multilingua avanzati**:
  - Integrazione modelli ottimizzati per italiano
  - Supporto modelli multilingua più potenti
  - Handling specifico per documenti tecnici/legali

- **Embedding ibridi**:
  - Combinazione testo + metadati per ricerca arricchita
  - Embedding specializzati per contenuti tecnici
  - Supporto per query complex e multi-hop

- **Ottimizzazione per quantizzazione**:
  - Supporto ONNX per inference accelerata
  - Quantizzazione modelli per ridurre memoria
  - Bilanciamento qualità/performance

## Piano di implementazione delle pipeline specializzate

Per implementare le pipeline specializzate, seguire questi step:

1. **Refactoring dell'architettura esistente**:
   - Creare una classe base astratta `BasePipeline` che definisca l'interfaccia comune
   - Estrarre logica comune dalla `TechnicalPipeline` esistente
   - Definire metodi astratti per comportamenti specifici delle categorie
   - Aggiungere supporto per configurazioni TOML specifiche per categoria

2. **Implementazione LegalPipeline**:
   - Studiare la struttura di documenti legali rappresentativi
   - Adattare algoritmi di chunking per riconoscere struttura articoli/sezioni
   - Implementare riconoscimento citazioni legali
   - Creare configurazione TOML specifica per documenti legali

3. **Factory pattern per la selezione della pipeline**:
   - Implementare una factory class che selezioni automaticamente la pipeline più appropriata
   - Integrare con il sistema di classificazione esistente
   - Aggiungere override manuale nella GUI

4. **Testing e validazione**:
   - Creare suite di test con documenti legali rappresentativi
   - Misurare qualità del chunking e dell'elaborazione
   - Confrontare risultati con la pipeline generica

## Priorità implementazione

1. **Alta priorità**:
   - Refactoring e creazione BasePipeline
   - Implementazione LegalPipeline
   - Miglioramento OCR multilingua (focus italiano)
   - Completamento GUI per classificazione documento

2. **Media priorità**:
   - Implementazione AcademicPipeline e BusinessPipeline
   - Sistema valutazione RAG
   - Gestione persistente indici
   - Vista comparativa testo originale/corretto

3. **Futura implementazione**:
   - Implementazione NarrativePipeline
   - Integrazione modelli avanzati embedding
   - Dashboard statistiche avanzate
   - Sistema miglioramento continuo basato su feedback