# Contesto di sviluppo - Branch main

Questo documento mantiene il contesto di sviluppo per il branch `main`, che contiene i tentativi di migrazione a `uv` e le relative correzioni di compatibilità.

## Panoramica del branch

Il branch `main` rappresenta lo stato più recente del tentativo di modernizzare Raggiro adottando `uv` come gestore di pacchetti e ambienti virtuali al posto di pip/virtualenv. A causa di incompatibilità con i modelli spaCy, questa linea di sviluppo è attualmente in pausa e non è il branch principale di lavoro.

## Problemi con uv e spaCy

Il tentativo di migrazione ha incontrato i seguenti problemi:

1. **Incompatibilità fondamentale**:
   - I modelli linguistici spaCy (come `it_core_news_sm`, `en_core_web_sm`, ecc.) non sono pubblicati su PyPI
   - `uv` non può installarli direttamente come dipendenze in `pyproject.toml`
   - Questo causa errori del tipo "No solution found when resolving dependencies"

2. **Soluzioni tentate**:
   - Rimozione dei modelli spaCy come dipendenze dirette in `pyproject.toml`
   - Aggiornamento di `segmenter.py` per gestire più intelligentemente i modelli mancanti
   - Miglioramento della documentazione sull'installazione manuale dei modelli
   - Modifiche allo script `setup_dev_env.sh` per un'installazione più resiliente

3. **Stato attuale**:
   - La migrazione è stata sospesa a favore del consolidamento della versione stabile (branch `stable-pre-uv`)
   - Il branch `main` contiene le ultime modifiche di compatibilità che potrebbero essere utili in futuro
   - È stato pianificato un percorso di migrazione alternativo (Stanza o DocLing)

## Modifiche principali

Le principali modifiche in questo branch includono:

1. **Aggiornamento pyproject.toml**:
   - Rimozione dei modelli spaCy come dipendenze dirette
   - Aggiornamento a dipendenze più recenti
   - Configurazione per `uv`

2. **Miglioramento segmenter.py**:
   - Implementazione di un sistema di fallback per modelli mancanti
   - Logging migliorato per problemi di caricamento modelli
   - Tentativo automatico di caricare modelli alternativi

3. **Script di installazione**:
   - Aggiornamento per usare `uv` invece di `pip`
   - Script Python per l'installazione più robusta dei modelli spaCy
   - Gestione degli errori migliorata

4. **Documentazione**:
   - Aggiornamento della documentazione di installazione
   - Sezione troubleshooting per problemi comuni
   - Documentazione sulle alternative di migrazione

## Stato attuale - 9 Aprile 2025

- Branch con tentativi di migrazione a `uv` attualmente in pausa
- Problemi di compatibilità identificati ma non completamente risolti
- Documentazione aggiornata sulle alternative di migrazione

## Decisione strategica

La decisione strategica è di:

1. Mantenere questo branch come riferimento per i tentativi con `uv`
2. Focalizzare lo sviluppo attivo sul branch `stable-pre-uv` (versione stabile con pip)
3. Pianificare una migrazione più ambiziosa verso Stanza o DocLing (branch `docling-fork`)

## Prossimi passaggi

Anche se questo branch non è attivamente sviluppato, potrebbero essere necessarie le seguenti azioni:

1. **Possibile ripresa in futuro**:
   - Se `uv` implementa un supporto migliore per modelli spaCy
   - Se spaCy pubblica i modelli su PyPI
   - Se viene trovata una soluzione alternativa

2. **Cherry-picking di migliorie**:
   - Alcune correzioni in questo branch potrebbero essere utili anche in `stable-pre-uv`
   - Parti del codice di gestione fallback potrebbero essere riutilizzate

3. **Documentazione**:
   - Mantenere aggiornata la documentazione sulle lezioni apprese
   - Aggiornare il file di contesto quando vengono prese nuove decisioni

---

## Sessione del 9 Aprile 2025

### Progressi
- Creato branch backup-uv-changes per salvare lo stato attuale
- Creato documento di contesto specifico per questo branch
- Identificato chiaramente i problemi con la migrazione uv

### Decisioni tecniche
- Sospensione dello sviluppo attivo su questo branch
- Mantenimento come riferimento per futuri tentativi con uv
- Focus su approcci alternativi (Stanza/DocLing)

### File principali modificati
- `pyproject.toml` (rimosse dipendenze modelli spaCy)
- `raggiro/core/segmenter.py` (migliorata resilienza)
- `docs/installation.md` (aggiornata documentazione)
- `scripts/installation/setup_dev_env.sh` (migliorato script)

### Prossimi passaggi da discutere
- Decisione finale sull'abbandono completo di uv
- Valutazione periodica della situazione uv + spaCy
- Eventuale reimplementazione futura con uv quando i problemi saranno risolti