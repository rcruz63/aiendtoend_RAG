#!/bin/bash

# Definir los directorios
SOURCE_DIR="/Users/rcruz2/Documents/Mas50_2025"
DEST_DIR="catalogo_md"

# Comprobar que el comando markitdown existe
if ! command -v markitdown &> /dev/null; then
    echo "❌ Error: El comando 'markitdown' no está instalado o no se encuentra en el PATH."
    exit 1
fi

# Crear el directorio destino si no existe
mkdir -p "$DEST_DIR"

# Contar cuántos PDF hay en el directorio fuente
pdf_count=$(find "$SOURCE_DIR" -maxdepth 1 -name "*.pdf" | wc -l)
pdf_count=$(echo $pdf_count | xargs) # Eliminar espacios en blanco

if [ "$pdf_count" -eq 0 ]; then
    echo "No se encontraron archivos PDF en $SOURCE_DIR"
    exit 0
fi

echo "Iniciando conversión de PDFs a Markdown..."
echo "Directorio fuente: $SOURCE_DIR"
echo "Directorio destino: $DEST_DIR"
echo "Se encontraron $pdf_count archivos PDF para procesar."

# Procesar cada archivo PDF
find "$SOURCE_DIR" -maxdepth 1 -name "*.pdf" | while read pdf_file; do
    # Obtener solo el nombre del archivo
    filename=$(basename "$pdf_file")
    # Obtener el nombre sin extensión
    name_without_ext="${filename%.*}"
    # Definir el archivo de salida
    output_file="$DEST_DIR/${name_without_ext}.md"
    
    echo "Procesando: $filename -> $(basename "$output_file")"
    
    # Ejecutar markitdown y redirigir la salida al archivo
    if markitdown "$pdf_file" > "$output_file" 2>/dev/null; then
        echo "✅ Conversión completada: $output_file"
    else
        echo "❌ Error al procesar $pdf_file"
    fi
done

echo "Proceso completado." 