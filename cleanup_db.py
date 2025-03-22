import logging
from pathlib import Path
import apsw
import sqlite_vec

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

def limpiar_tablas_obsoletas():
    """Elimina las tablas obsoletas de la base de datos"""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Lista de tablas a eliminar
        tablas_obsoletas = [
            'chunks_embeddings_info',
            'chunks_embeddings_chunks',
            'chunks_embeddings_rowids',
            'chunks_embeddings_vector_chunks00'
        ]
        
        # Verificar qué tablas existen
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tablas_existentes = [t[0] for t in cursor.fetchall()]
        
        # Iniciar transacción
        cursor.execute("BEGIN TRANSACTION")
        
        # Eliminar cada tabla obsoleta si existe
        for tabla in tablas_obsoletas:
            if tabla in tablas_existentes:
                logging.info(f"Eliminando tabla: {tabla}")
                cursor.execute(f"DROP TABLE IF EXISTS {tabla}")
            else:
                logging.info(f"La tabla {tabla} no existe, saltando...")
        
        # Confirmar transacción
        cursor.execute("COMMIT")
        
        # Verificar tablas restantes
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tablas_restantes = [t[0] for t in cursor.fetchall()]
        logging.info("\nTablas restantes en la base de datos:")
        for tabla in tablas_restantes:
            logging.info(f"- {tabla}")
        
        conn.close()
        logging.info("\nLimpieza completada con éxito")
        
    except Exception as e:
        logging.error(f"Error durante la limpieza: {e}")
        if 'cursor' in locals():
            cursor.execute("ROLLBACK")
        raise

if __name__ == "__main__":
    # Preguntar confirmación
    print("\nEste script eliminará las tablas obsoletas de la base de datos.")
    print("Las siguientes tablas serán eliminadas:")
    print("- chunks_embeddings_info")
    print("- chunks_embeddings_chunks")
    print("- chunks_embeddings_rowids")
    print("- chunks_embeddings_vector_chunks00")
    print("\nLas tablas principales se mantendrán intactas.")
    respuesta = input("\n¿Desea continuar? (s/n): ")
    
    if respuesta.lower() == 's':
        limpiar_tablas_obsoletas()
    else:
        print("Operación cancelada") 