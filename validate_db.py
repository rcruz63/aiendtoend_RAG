import logging
from pathlib import Path
import apsw
import sqlite_vec
import numpy as np

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def get_connection():
    """Obtiene una conexión a la base de datos con la extensión sqlite-vec cargada"""
    db_path = Path("data") / "catalogo.db"
    conn = apsw.Connection(str(db_path))
    conn.enableloadextension(True)
    sqlite_vec.load(conn)
    conn.enableloadextension(False)
    return conn

def validar_base_datos():
    """Valida el contenido de la base de datos"""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # 1. Verificar estructura de las tablas
        logging.info("\n=== Estructura de las tablas ===")
        cursor.execute("SELECT name, type FROM sqlite_master WHERE type IN ('table', 'view')")
        tablas = cursor.fetchall()
        logging.info("Tablas y vistas encontradas:")
        for tabla in tablas:
            logging.info(f"- {tabla[0]} ({tabla[1]})")
        
        # 2. Contar registros en cada tabla
        logging.info("\n=== Conteo de registros ===")
        cursor.execute("SELECT COUNT(*) FROM chunks_metadata")
        total_chunks = cursor.fetchone()[0]
        logging.info(f"Total de chunks: {total_chunks}")
        
        # Verificar si la tabla de embeddings existe y contar registros
        try:
            cursor.execute("SELECT COUNT(*) FROM chunks_embeddings")
            total_embeddings = cursor.fetchone()[0]
            logging.info(f"Total de embeddings: {total_embeddings}")
        except Exception as e:
            logging.error(f"✗ Error al acceder a chunks_embeddings: {e}")
            total_embeddings = 0
        
        # 3. Verificar documentos procesados
        logging.info("\n=== Documentos procesados ===")
        cursor.execute("""
        SELECT DISTINCT ruta_archivo, titulo, COUNT(*) as num_chunks
        FROM chunks_metadata
        GROUP BY ruta_archivo, titulo
        ORDER BY ruta_archivo
        """)
        
        for doc in cursor.fetchall():
            logging.info(f"\nDocumento: {doc[1]}")
            logging.info(f"Ruta: {doc[0]}")
            logging.info(f"Chunks: {doc[2]}")
        
        # 4. Verificar integridad básica
        logging.info("\n=== Verificación de integridad ===")
        if total_embeddings == 0:
            logging.error("✗ No hay embeddings en la base de datos")
        elif total_chunks == total_embeddings:
            logging.info("✓ El número de chunks coincide con el número de embeddings")
        else:
            logging.error(f"✗ Discrepancia: {total_chunks} chunks vs {total_embeddings} embeddings")
        
        conn.close()
        
    except Exception as e:
        logging.error(f"Error al validar la base de datos: {e}")
        raise

if __name__ == "__main__":
    validar_base_datos() 