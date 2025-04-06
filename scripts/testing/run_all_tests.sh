#!/bin/bash
# Script per elaborare e testare tutti i PDF nella directory tmp

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
TMP_DIR="$BASE_DIR/tmp"
TEST_OUTPUT_DIR="$BASE_DIR/test_output"

# Assicurati che le directory esistano
mkdir -p "$TEST_OUTPUT_DIR"

# Funzione per pulire nome file per directory
clean_filename() {
    local filename="$1"
    # Rimuovi estensione, caratteri speciali e sostituisci spazi con underscore
    echo "${filename%.*}" | sed 's/[^a-zA-Z0-9]/_/g' | tr -s '_'
}

# Processa ogni PDF
for pdf_file in "$TMP_DIR"/*.pdf; do
    if [ -f "$pdf_file" ]; then
        filename=$(basename "$pdf_file")
        echo "===== Elaborazione di: $filename ====="
        
        # Crea nome directory pulito
        clean_name=$(clean_filename "$filename")
        output_dir="${TEST_OUTPUT_DIR}/${clean_name}"
        
        # Elabora il documento
        # Cerca lo script in diverse posizioni possibili
        if [ -f "$BASE_DIR/examples/scripts/test_semantic_chunking.py" ]; then
            echo "Esecuzione script dalla directory scripts..."
            python "$BASE_DIR/examples/scripts/test_semantic_chunking.py" --input "$pdf_file" --output "$output_dir"
        elif [ -f "$BASE_DIR/examples/test_semantic_chunking.py" ]; then
            echo "Esecuzione script dalla directory examples..."
            python "$BASE_DIR/examples/test_semantic_chunking.py" --input "$pdf_file" --output "$output_dir"
        elif [ -f "$BASE_DIR/test_semantic_chunking.py" ]; then
            echo "Esecuzione script dalla directory principale..."
            python "$BASE_DIR/test_semantic_chunking.py" --input "$pdf_file" --output "$output_dir"
        else
            echo "ERRORE: Script test_semantic_chunking.py non trovato!"
            exit 1
        fi
        
        # Trova la configurazione di test appropriata
        yaml_file=""
        case "$filename" in
            "2020-Scrum-Guide-Italian.pdf")
                yaml_file="scrum_guide.yaml" ;;
            "WEF_Future_of_Jobs_Report_2025.pdf")
                yaml_file="future_of_jobs.yaml" ;;
            "Humanizar_it.pdf")
                yaml_file="humanizar.yaml" ;;
            "PSN_Allegato Tecnico_v2.0.pdf")
                yaml_file="psn_allegato.yaml" ;;
            "Hornresp manual (1).pdf")
                yaml_file="hornresp_manual.yaml" ;;
            "Canción de peregrino.pdf")
                yaml_file="cancion_peregrino.yaml" ;;
            "Capitolato Tecnico e Allegati 1.pdf")
                yaml_file="capitolato_tecnico.yaml" ;;
            "Kenny_Werner_Effortless_Mastery_Liberati.pdf")
                yaml_file="kenny_werner.yaml" ;;
            *)
                # Se non c'è una configurazione specifica, salta
                echo "Nessuna configurazione trovata per $filename, salto i test"
                continue ;;
        esac
        
        # Esegui i test promptfoo se abbiamo trovato la configurazione
        if [ -n "$yaml_file" ]; then
            echo "Esecuzione test con configurazione: $yaml_file"
            # Cerca il runner promptfoo in diverse posizioni
            if [ -f "$BASE_DIR/raggiro/testing/promptfoo_runner.py" ]; then
                echo "Esecuzione promptfoo dalla directory testing..."
                python "$BASE_DIR/raggiro/testing/promptfoo_runner.py" "$SCRIPT_DIR/$yaml_file" "$output_dir"
            elif [ -f "$BASE_DIR/testing/promptfoo_runner.py" ]; then
                echo "Esecuzione promptfoo dalla directory principale/testing..."
                python "$BASE_DIR/testing/promptfoo_runner.py" "$SCRIPT_DIR/$yaml_file" "$output_dir"
            elif [ -f "$BASE_DIR/test_prompts/promptfoo_runner.py" ]; then
                echo "Esecuzione promptfoo dalla directory test_prompts..."
                python "$BASE_DIR/test_prompts/promptfoo_runner.py" "$SCRIPT_DIR/$yaml_file" "$output_dir"
            else
                echo "AVVISO: Script promptfoo_runner.py non trovato, utilizzo script di fallback"
                # Usa uno script di fallback minimo
                echo "Contenuto di $SCRIPT_DIR/$yaml_file:"
                cat "$SCRIPT_DIR/$yaml_file"
                echo "-----------------------------------------------"
                echo "Directory di output: $output_dir"
                echo "Usa questo file in promptfoo manualmente"
            fi
        fi
        
        echo "===== Completato: $filename ====="
        echo ""
    fi
done

echo "Tutti i test completati!"