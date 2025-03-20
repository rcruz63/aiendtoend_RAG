import sqlite3
from pathlib import Path
import pickle
import numpy as np

def validate_database():
    db_path = Path("data") / "catalogo.db"
    
    # Conectar a la base de datos
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Obtener estadísticas básicas
    cursor.execute("SELECT COUNT(*) FROM chunks")
    total_chunks = cursor.fetchone()[0]
    print(f"\nTotal de chunks en la base de datos: {total_chunks}")
    
    # Obtener información de los documentos únicos
    cursor.execute("""
        SELECT DISTINCT titulo, ruta_archivo, COUNT(*) as num_chunks 
        FROM chunks 
        GROUP BY titulo, ruta_archivo
    """)
    documentos = cursor.fetchall()
    
    print("\nDocumentos procesados:")
    for doc in documentos:
        print(f"\nTítulo: {doc[0]}")
        print(f"Ruta: {doc[1]}")
        print(f"Número de chunks: {doc[2]}")
        
        # Mostrar los primeros 3 chunks de cada documento
        cursor.execute("""
            SELECT id, contenido, inicio, fin 
            FROM chunks 
            WHERE titulo = ? 
            LIMIT 3
        """, (doc[0],))
        
        chunks = cursor.fetchall()
        print("\nPrimeros 3 chunks:")
        for chunk in chunks:
            print(f"\nID: {chunk[0]}")
            print(f"Contenido: {chunk[1][:100]}...")  # Mostrar solo los primeros 100 caracteres
            print(f"Posición: {chunk[2]} - {chunk[3]}")
    
    # Verificar que los embeddings son válidos
    cursor.execute("SELECT id, embedding FROM chunks LIMIT 1")
    chunk = cursor.fetchone()
    if chunk:
        try:
            embedding = pickle.loads(chunk[1])
            print(f"\nVerificación de embedding:")
            print(f"ID del chunk: {chunk[0]}")
            print(f"Dimensiones del embedding: {embedding.shape}")
            print(f"Tipo de datos: {embedding.dtype}")
        except Exception as e:
            print(f"\nError al cargar el embedding: {e}")
    
    conn.close()

if __name__ == "__main__":
    validate_database() 