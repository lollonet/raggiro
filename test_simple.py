#!/usr/bin/env python3
"""
Script di test semplificato per la pipeline Raggiro.
"""

import json
import os
import sys
from pathlib import Path

# Configurazioni
TEST_DIR = Path('/home/ubuntu/raggiro/test_output')
INPUT_FILE = Path('/home/ubuntu/raggiro/examples/documents/sample_report.md')
OUTPUT_DIR = Path('/home/ubuntu/raggiro/test_output')

# Creiamo la directory di output
os.makedirs(TEST_DIR, exist_ok=True)

print("=== Test semplificato della pipeline documentale ===")
print(f"File di input: {INPUT_FILE}")
print(f"Directory di output: {OUTPUT_DIR}")

# Simulazione semplificata della pipeline
def simulate_pipeline(input_file, output_dir):
    print("\n1. Analisi del file...")
    # Otteniamo le informazioni sul file
    file_path = Path(input_file)
    file_metadata = {
        "filename": file_path.name,
        "extension": file_path.suffix,
        "path": str(file_path.absolute()),
        "size": os.path.getsize(file_path)
    }
    
    print(f"File: {file_metadata['filename']}")
    print(f"Dimensione: {file_metadata['size']} bytes")
    
    print("\n2. Estrazione del contenuto...")
    # Leggiamo il contenuto del file (nel caso reale sarebbe estratto dal PDF)
    with open(input_file, 'r') as f:
        text = f.read()
    
    text_sample = text[:150]
    print(f"Esempio di testo estratto: {text_sample}...")
    
    print("\n3. Pulizia del testo...")
    # Semplice pulizia
    cleaned_text = text.replace('\r', '\n')
    
    print("\n4. Segmentazione...")
    # Semplice segmentazione in paragrafi
    paragraphs = [p for p in cleaned_text.split("\n\n") if p.strip()]
    chunks = []
    
    # Creiamo dei chunk dal testo
    current_chunk = ""
    max_size = 1000
    
    for p in paragraphs:
        if len(current_chunk) + len(p) > max_size and current_chunk:
            chunks.append(current_chunk)
            current_chunk = p
        else:
            current_chunk += "\n\n" + p if current_chunk else p
    
    if current_chunk:
        chunks.append(current_chunk)
    
    print(f"Creati {len(paragraphs)} paragrafi e {len(chunks)} chunks")
    
    print("\n5. Estrazione metadati...")
    # Estrazione semplificata dei metadati
    metadata = {
        "filename": file_metadata["filename"],
        "file_path": file_metadata["path"],
        "file_type": file_metadata["extension"].lstrip("."),
        "word_count": len(text.split()),
        "char_count": len(text),
    }
    
    # Cerchiamo un titolo
    lines = text.split("\n")
    if lines and lines[0].startswith("# "):
        metadata["title"] = lines[0].replace("# ", "")
    
    # Cerchiamo una data
    for line in lines:
        if "date:" in line.lower():
            metadata["date"] = line.split(":", 1)[1].strip()
            break
    
    print(f"Metadati estratti: {list(metadata.keys())}")
    
    print("\n6. Esportazione...")
    # Documento finale
    document = {
        "text": cleaned_text,
        "metadata": metadata,
        "chunks": [{"id": f"chunk_{i+1}", "text": chunk} for i, chunk in enumerate(chunks)]
    }
    
    # Esportiamo in formato Markdown
    output_md = output_dir / f"{file_path.stem}.md"
    with open(output_md, 'w') as f:
        f.write(f"# {metadata.get('title', 'Documento')}\n\n")
        if "date" in metadata:
            f.write(f"Data: {metadata['date']}\n\n")
        f.write(cleaned_text)
    
    # Esportiamo in formato JSON
    output_json = output_dir / f"{file_path.stem}.json"
    with open(output_json, 'w') as f:
        json.dump(document, f, indent=2)
    
    print(f"File esportati:")
    print(f"- Markdown: {output_md}")
    print(f"- JSON: {output_json}")
    
    return {
        "input_file": str(input_file),
        "output_dir": str(output_dir),
        "formats": {
            "markdown": str(output_md),
            "json": str(output_json)
        },
        "success": True
    }

# Esegui la pipeline
result = simulate_pipeline(INPUT_FILE, OUTPUT_DIR)

print("\n=== Test completato ===")
print("Il documento è stato elaborato correttamente!")

# Mostra i contenuti della directory di output
print("\nContenuti della directory di output:")
for item in os.listdir(OUTPUT_DIR):
    print(f"- {item}")