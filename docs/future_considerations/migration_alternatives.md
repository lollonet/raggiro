# Alternative di Migrazione per Raggiro

Questo documento esplora le possibili evoluzioni architetturali di Raggiro, con particolare attenzione alle alternative per sostituire spaCy e migliorare la compatibilità con gestori di pacchetti moderni come `uv`.

## Indice
1. [Problemi attuali](#problemi-attuali)
2. [Alternativa 1: Migrazione a Stanza](#alternativa-1-migrazione-a-stanza)
3. [Alternativa 2: Fork per DocLing](#alternativa-2-fork-per-docling)
4. [Analisi comparativa](#analisi-comparativa)
5. [Raccomandazioni](#raccomandazioni)

## Problemi attuali

Raggiro attualmente si basa su spaCy per funzionalità NLP critiche, ma presenta alcuni problemi:

1. **Incompatibilità con gestori pacchetti moderni**:
   - I modelli linguistici spaCy non sono disponibili su PyPI
   - Richiedono installazione manuale tramite comandi separati
   - Causano errori con `uv` e altri gestori di dipendenze

2. **Gestione complessa dei modelli**:
   - Necessità di codice difensivo per gestire modelli mancanti
   - Fallback in cascata per assicurare funzionalità minima
   - Esperienza utente complicata durante setup

3. **Architettura frammentata**:
   - Dipendenze multiple per diverse funzionalità (PyMuPDF, spaCy, pytesseract)
   - Manutenzione più complessa e interdipendenze fragili

## Alternativa 1: Migrazione a Stanza

Stanza è una libreria NLP sviluppata da Stanford che offre funzionalità simili a spaCy ma con gestione modelli integrata e compatibilità completa con PyPI.

### Piano di migrazione in fasi

#### Fase 1: Prototipo e Valutazione (2-3 giorni)
1. **Creazione di un modulo wrapper**: `raggiro/core/nlp_engine.py` che fornisca un'interfaccia astratta per funzionalità NLP
2. **Implementazione con Stanza**: Creare due implementazioni dell'interfaccia:
   - `StanzaEngine` (nuova implementazione)
   - `SpacyEngine` (adattatore esistente)
3. **Test comparativi** su un set di documenti rappresentativi

#### Fase 2: Refactoring di segmenter.py (3-4 giorni)
1. **Modificare `segmenter.py`** per usare l'interfaccia astratta anziché spaCy direttamente
2. **Isolamento della dipendenza** in un unico punto del codice
3. **Aggiunta configurazione** per selezionare l'engine NLP preferito

#### Fase 3: Aggiornamento Dipendenze e Documentazione (1-2 giorni)
1. **Aggiornamento pyproject.toml**
2. **Aggiornamento documentazione** con istruzioni per entrambi i motori NLP

#### Fase 4: Adattamento Pipeline e Testing (2-3 giorni)
1. **Adattamento pipeline di elaborazione** per supportare la nuova struttura
2. **Estensione dei test** per verificare entrambe le implementazioni
3. **Validazione della qualità** dei risultati tra le due implementazioni

### Dettagli di implementazione

#### Interfaccia NLP astratta (nlp_engine.py)
```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class NLPEngine(ABC):
    """Interfaccia astratta per motori NLP."""
    
    @abstractmethod
    def initialize(self, language: str = "it") -> bool:
        """Inizializza il motore NLP con una lingua specificata."""
        pass
    
    @abstractmethod
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """Analizza un testo e restituisce strutture linguistiche."""
        pass
    
    @abstractmethod
    def get_sentences(self, text: str) -> List[str]:
        """Divide il testo in frasi."""
        pass
    
    @abstractmethod
    def get_pos_tags(self, text: str) -> List[Dict[str, str]]:
        """Restituisce i tag POS per il testo."""
        pass
    
    @abstractmethod
    def is_header(self, text: str) -> bool:
        """Determina se il testo è un'intestazione basandosi su features linguistiche."""
        pass
    
    @abstractmethod
    def get_language(self, text: str) -> str:
        """Rileva la lingua del testo."""
        pass
```

#### Implementazione Stanza (stanza_engine.py)
```python
import stanza
from typing import List, Dict, Any, Optional
from .nlp_engine import NLPEngine
import logging

logger = logging.getLogger("raggiro.nlp.stanza")

class StanzaEngine(NLPEngine):
    """Implementazione del motore NLP usando Stanza."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Inizializza l'engine con configurazione opzionale."""
        self.config = config or {}
        self.nlp = None
        self.language = "it"
        self.processors = "tokenize,pos,lemma"
        
    def initialize(self, language: str = "it") -> bool:
        """Inizializza Stanza con i processori richiesti."""
        try:
            self.language = language
            # Stanza scarica automaticamente i modelli necessari
            stanza.download(language, processors=self.processors)
            self.nlp = stanza.Pipeline(language, processors=self.processors)
            logger.info(f"Inizializzato Stanza per lingua: {language}")
            return True
        except Exception as e:
            logger.error(f"Errore nell'inizializzazione di Stanza: {e}")
            return False
    
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """Analizza il testo usando Stanza."""
        if not self.nlp:
            self.initialize(self.language)
            
        if not text.strip():
            return {"sentences": [], "pos": [], "tokens": []}
            
        doc = self.nlp(text)
        
        result = {
            "sentences": [sent.text for sent in doc.sentences],
            "pos": [],
            "tokens": []
        }
        
        for sent in doc.sentences:
            for word in sent.words:
                result["pos"].append({
                    "text": word.text,
                    "pos": word.pos,
                    "lemma": word.lemma
                })
                result["tokens"].append(word.text)
                
        return result
    
    def get_sentences(self, text: str) -> List[str]:
        """Estrae le frasi dal testo."""
        analysis = self.analyze_text(text)
        return analysis["sentences"]
    
    def get_pos_tags(self, text: str) -> List[Dict[str, str]]:
        """Estrae i tag POS dal testo."""
        analysis = self.analyze_text(text)
        return analysis["pos"]
    
    def is_header(self, text: str) -> bool:
        """Determina se il testo è un'intestazione basandosi su POS."""
        if len(text) > 200 or not text.strip():
            return False
            
        analysis = self.analyze_text(text)
        pos_info = analysis["pos"]
        
        total_tokens = len(pos_info)
        if total_tokens == 0:
            return False
            
        # Conta sostantivi propri (PROPN)
        proper_nouns = sum(1 for token in pos_info if token["pos"] == "PROPN")
        # Conta verbi
        verbs = sum(1 for token in pos_info if token["pos"] in ["VERB", "AUX"])
        
        proper_noun_ratio = proper_nouns / total_tokens
        has_few_verbs = verbs <= 1
        
        return proper_noun_ratio > 0.5 and has_few_verbs and total_tokens < 10
    
    def get_language(self, text: str) -> str:
        """Ritorna la lingua configurata (Stanza non ha rilevazione automatica)."""
        return self.language
```

#### Modifiche a segmenter.py

```python
from raggiro.core.nlp_engine import NLPEngine
from raggiro.core.stanza_engine import StanzaEngine
from raggiro.core.spacy_engine import SpacyEngine  # Adapter per retrocompatibilità

class Segmenter:
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the segmenter."""
        self.config = config or {}
        
        # Configura il motore NLP in base alla configurazione
        segmentation_config = self.config.get("segmentation", {})
        engine_type = segmentation_config.get("nlp_engine", "stanza")
        
        # Crea il motore NLP appropriato
        if engine_type == "stanza":
            self.nlp_engine = StanzaEngine(segmentation_config)
        elif engine_type == "spacy":
            self.nlp_engine = SpacyEngine(segmentation_config)
        else:
            # Fallback a Stanza come default sicuro
            logger.warning(f"Motore NLP {engine_type} non supportato, utilizzo Stanza")
            self.nlp_engine = StanzaEngine(segmentation_config)
        
        # Inizializza il motore
        language = segmentation_config.get("language", "it")
        self.nlp_engine.initialize(language)
        
        # Altre inizializzazioni rimanenti...
```

### Vantaggi della migrazione a Stanza

1. **Manutenibilità migliorata**:
   - Isolamento della dipendenza NLP dietro interfaccia
   - Facilità di cambiare implementazione in futuro
   - Separazione chiara delle responsabilità

2. **Esperienza utente semplificata**:
   - Eliminazione passaggi manuali di installazione modelli
   - Riduzione errori di configurazione

3. **Estensibilità**:
   - Framework pronto per integrare altre soluzioni NLP
   - Possibilità di aggiungere implementazioni basate su HuggingFace o DocLing

4. **Compatibilità totale con uv**:
   - Installazione semplificata senza problemi di PyPI

### Considerazioni

- **Dimensioni download**: I modelli Stanza sono scaricati alla prima esecuzione ma sono più grandi (circa 200-500MB per lingua)
- **Performance**: Da testare su documenti di diverse dimensioni
- **Approccio ibrido**: Mantenere spaCy come opzione per compatibilità legacy

### Timeline suggerita

- Settimana 1: Sviluppo interfaccia astratta e implementazione Stanza
- Settimana 2: Refactoring segmenter.py e test comparativi
- Settimana 3: Documentazione, aggiornamento dipendenze e UI
- Settimana 4: Testing finale e rilascio

## Alternativa 2: Fork per DocLing

DocLing rappresenterebbe un salto tecnologico significativo, trasformando Raggiro da un sistema con componenti separati per NLP e parsing documentale a una soluzione integrata.

### Piano di fork e migrazione

#### Fase 1: Setup e prototipazione (1 settimana)
1. **Creazione del fork**:
   ```bash
   git clone https://github.com/lollonet/raggiro.git raggiro-docling
   cd raggiro-docling
   git checkout -b docling-migration
   ```

2. **Installazione delle dipendenze**:
   ```bash
   uv pip install docling
   ```

3. **Prototipo di elaborazione documentale**:
   Creare un modulo `raggiro/core/document_processor.py` usando DocLing per le funzionalità base.

#### Fase 2: Refactoring dei componenti core (2-3 settimane)

1. **Sostituire `extractor.py` con DocLing**:
   ```python
   import docling
   
   class DocumentExtractor:
       def __init__(self, config=None):
           self.config = config or {}
           self.docling_client = docling.Client()
       
       def extract_text(self, file_path):
           # DocLing gestisce tutti i formati in modo unificato
           doc = self.docling_client.process_document(file_path)
           return {
               "text": doc.get_text(),
               "pages": [page.get_text() for page in doc.pages],
               "metadata": doc.metadata,
               "structure": doc.get_structure()
           }
   ```

2. **Sostituire `segmenter.py` con l'API di struttura DocLing**:
   ```python
   class DocumentSegmenter:
       def __init__(self, config=None):
           self.config = config or {}
       
       def segment(self, docling_document):
           # DocLing gestisce già la segmentazione semantica
           chunks = []
           
           # Estrae i chunk in base alla struttura del documento
           for section in docling_document.get_sections():
               chunks.append({
                   "text": section.get_text(),
                   "title": section.get_title(),
                   "level": section.get_level(),
                   "metadata": {
                       "page": section.page_number,
                       "section_type": section.type
                   }
               })
               
           return chunks
   ```

3. **Migrazione della correzione ortografica**, sfruttando il preprocessing DocLing.

#### Fase 3: Integrazione con RAG (2 settimane)

1. **Aggiornare pipeline RAG** per usare documenti DocLing:
   ```python
   class RagPipeline:
       def process_document(self, file_path):
           # Processo unificato DocLing
           docling_client = docling.Client()
           document = docling_client.process_document(file_path)
           
           # Estrazione e chunking automatici
           chunks = document.get_chunks(
               strategy=self.config.get("chunking_strategy", "semantic"),
               chunk_size=self.config.get("chunk_size", 1000)
           )
           
           # Indicizzazione diretta
           vectors = self.embed_chunks(chunks)
           self.store_vectors(vectors)
           
           return {
               "document_id": document.id,
               "chunks": len(chunks),
               "vectors": len(vectors)
           }
   ```

2. **Miglioramento retriever** con le capacità DocLing.

#### Fase 4: Interfaccia utente e documentazione (1-2 settimane)

1. **Aggiornare l'interfaccia Streamlit** per mostrare nuove capacità.
2. **Aggiornare la documentazione** per evidenziare le nuove funzionalità.

#### Fase 5: Testing e finalizzazione (2 settimane)

1. **Creare test comparativi** tra versione originale e fork.
2. **Rilascio pubblico**.

### Architettura DocLing proposta

```
raggiro-docling/
├── raggiro/
│   ├── core/
│   │   ├── document_processor.py  # Wrapper DocLing principale
│   │   ├── structure_analyzer.py  # Analisi struttura documentale
│   │   ├── table_extractor.py     # Estrazione e normalizzazione tabelle
│   │   └── language_detector.py   # Rilevamento lingua con DocLing
│   ├── rag/
│   │   ├── docling_pipeline.py    # Pipeline RAG integrata
│   │   ├── docling_retriever.py   # Retrieval avanzato
│   │   └── docling_rewriter.py    # Riscrittura contestuale
│   └── gui/
│       ├── streamlit_docling.py   # UI con nuove capacità visualization
│       └── document_explorer.py   # Navigazione struttura documento
```

### Confronto specifico DocLing vs Stack attuale

| Funzionalità | Stack Attuale | DocLing |
|--------------|--------------|---------|
| **Parsing PDF** | PyMuPDF/pdfminer | Nativo con comprensione layout |
| **OCR** | Tesseract/pytesseract | Integrato con prestazioni superiori |
| **Analisi strutturale** | Algoritmi custom in segmenter.py | API strutturata nativa |
| **Analisi linguistica** | spaCy (problemi modelli) | Integrato con modelli precaricati |
| **Estrazione tabelle** | Limitata/manuale | Nativa con comprensione struttura |
| **Indicizzazione** | Processo multi-step | Pipeline unificata |
| **Visualizzazione** | Basica in Streamlit | Ricca con struttura preservata |

### Modifiche specifiche per dependency management

```toml
# pyproject.toml aggiornato
[project]
name = "raggiro-docling"
version = "0.2.0"
description = "Advanced document processing pipeline for RAG applications using DocLing"

dependencies = [
    # Core dependency
    "docling>=1.0.0",
    
    # Supporting packages
    "pandas>=2.0.0",
    "streamlit>=1.30.0",
    "qdrant-client>=1.11.0",
    "openai>=1.0.0",
    # Altri selezionati ma ridotti rispetto all'originale
]

# Rimossi dipendenze sostituite da DocLing:
# spacy, pymupdf, pdfminer.six, pytesseract, ecc.
```

### Roadmap e timeline

- **Mese 1**: Prototipo e proof-of-concept
  - Settimana 1-2: Implementazione extractor e segmenter
  - Settimana 3-4: Integrazione con RAG

- **Mese 2**: Sviluppo completo
  - Settimana 5-6: UI e visualizzazione
  - Settimana 7-8: Testing e ottimizzazioni

- **Mese 3**: Finalizzazione e documentazione
  - Settimana 9-10: Testing comparativo
  - Settimana 11-12: Rilascio e documentazione

### Valutazione ROI

- **Riduzione codebase**: -30-40% righe di codice grazie all'API unificata
- **Miglioramento prestazioni**: +50-70% in accuratezza su documenti complessi
- **Tempo di sviluppo futuro**: -40% per nuove funzionalità
- **Manutenzione**: -60% problemi legati a dipendenze multiple

## Analisi comparativa

| Criterio | Mantieni spaCy | Migrazione a Stanza | Fork per DocLing |
|----------|--------------|-----------------|--------------|
| **Complessità migrazione** | Nessuna | Media | Alta |
| **Tempo implementazione** | 0 giorni | 2-4 settimane | 2-3 mesi |
| **Rischio tecnico** | Basso | Medio | Medio-alto |
| **Risoluzione problemi attuali** | No | Parziale | Completa |
| **Miglioramento funzionalità** | Nessuno | Minimo | Significativo |
| **Riduzione codebase** | 0% | 5-10% | 30-40% |
| **Dipendenze esterne** | Molte | Molte | Poche |
| **Manutenibilità futura** | Bassa | Media | Alta |
| **Compatibilità con uv** | Problematica | Buona | Eccellente |

## Raccomandazioni

Basandoci sull'analisi, raccomandiamo un approccio in due fasi:

### Fase immediata (2-4 settimane)
Implementare la **migrazione a Stanza** come soluzione a breve termine per risolvere i problemi di compatibilità con uv, mantenendo l'architettura attuale e riducendo il rischio tecnico. Questo approccio fornisce:

- Risoluzione rapida dei problemi con i modelli spaCy
- Miglioramento dell'esperienza utente durante l'installazione
- Minimo impatto su altre parti del codice
- Base architetturale per future evoluzioni

### Fase strategica (3-6 mesi)
Iniziare parallelamente un **fork sperimentale con DocLing** per valutarne le reali potenzialità, con l'obiettivo di sviluppare una versione completamente riprogettata del sistema che possa:

- Semplificare drasticamente l'architettura
- Migliorare le capacità di elaborazione documentale
- Ridurre la manutenzione a lungo termine
- Posizionare Raggiro come soluzione all'avanguardia

Questa strategia a due livelli permette di risolvere i problemi immediati mentre si lavora per un miglioramento sostanziale dell'architettura nel medio termine.

---

*Documento creato il: 9 Aprile 2025*  
*Da rivedere: Trimestrale*