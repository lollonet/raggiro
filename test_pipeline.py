#!/usr/bin/env python3
"""
Script di test per la pipeline Raggiro.
"""

import os
import sys
from pathlib import Path

# Configurazioni
TEST_DIR = Path('/home/ubuntu/raggiro/test_output')
INPUT_FILE = Path('/home/ubuntu/raggiro/examples/documents/sample_report.pdf')

# Creiamo la directory di output
os.makedirs(TEST_DIR, exist_ok=True)

# Importiamo direttamente i moduli necessari
sys.path.insert(0, '/home/ubuntu/raggiro')

from raggiro.core.file_handler import FileHandler
from raggiro.core.extractor import Extractor
from raggiro.core.cleaner import Cleaner
from raggiro.core.segmenter import Segmenter
from raggiro.core.metadata import MetadataExtractor
from raggiro.core.exporter import Exporter

print("=== Test della pipeline di elaborazione documentale ===")
print(f"File di input: {INPUT_FILE}")
print(f"Directory di output: {TEST_DIR}")

# Creiamo una configurazione semplificata
config = {
    "extraction": {
        "ocr_enabled": True,
        "ocr_language": "eng"
    },
    "segmentation": {
        "use_spacy": False,  # Disabilitiamo spaCy per evitare dipendenze
        "max_chunk_size": 1000,
        "chunk_overlap": 200
    },
    "export": {
        "formats": ["markdown", "json"],
        "include_metadata": True
    }
}

# Inizializziamo i componenti
print("\n1. Inizializzazione dei componenti...")
file_handler = FileHandler(config)
extractor = Extractor(config)
cleaner = Cleaner(config)
segmenter = Segmenter(config)
metadata_extractor = MetadataExtractor(config)
exporter = Exporter(config)

# Elaborazione del file
print("\n2. Analisi del file...")
file_metadata = file_handler.get_file_metadata(INPUT_FILE)
file_type_info = file_handler.detect_file_type(INPUT_FILE)

print(f"Tipo di file rilevato: {file_type_info.get('document_type', 'sconosciuto')}")
print(f"MIME type: {file_type_info.get('mime_type', 'sconosciuto')}")

# Estrazione del testo
print("\n3. Estrazione del testo...")
try:
    document = extractor.extract(INPUT_FILE, file_type_info)
    
    if document["success"]:
        print(f"Estrazione riuscita con metodo: {document.get('extraction_method', 'sconosciuto')}")
        text_sample = document.get("text", "")[:150]
        print(f"Esempio di testo estratto: {text_sample}...")
    else:
        print(f"Errore di estrazione: {document.get('error', 'errore sconosciuto')}")
except Exception as e:
    print(f"Eccezione durante l'estrazione: {str(e)}")
    document = {"text": "Documento di esempio per test", "success": True, "pages": [{"text": "Documento di esempio", "page_num": 1}]}

# Pulizia del testo
print("\n4. Pulizia e normalizzazione del testo...")
document = cleaner.clean_document(document)
print("Pulizia completata")

# Segmentazione
print("\n5. Segmentazione logica del testo...")
document = segmenter.segment(document)
segment_count = len(document.get("segments", []))
chunk_count = len(document.get("chunks", []))
print(f"Segmenti creati: {segment_count}")
print(f"Chunks creati: {chunk_count}")

# Estrazione metadati
print("\n6. Estrazione metadati...")
document["metadata"] = metadata_extractor.extract(document, file_metadata)
md_keys = list(document.get("metadata", {}).keys())
print(f"Metadati estratti: {md_keys}")

# Esportazione
print("\n7. Esportazione nei formati di output...")
try:
    export_result = exporter.export(document, TEST_DIR)
    
    if export_result["success"]:
        print("Esportazione completata con successo")
        for fmt, path in export_result.get("formats", {}).items():
            print(f"- Formato {fmt}: {path}")
    else:
        print(f"Errore di esportazione: {export_result.get('error', 'errore sconosciuto')}")
except Exception as e:
    print(f"Eccezione durante l'esportazione: {str(e)}")

print("\n=== Test completato ===")