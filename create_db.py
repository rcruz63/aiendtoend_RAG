import sqlite3
import os
from pathlib import Path
from typing import List, Dict, Generator
import numpy as np
from tqdm import tqdm
from database import Database
import openai
from dotenv import load_dotenv
import argparse
import logging
from datetime import datetime
import sqlite_vec
import struct

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def serialize(vector: List[float]) -> bytes:
    """Serializa una lista de floats en formato de bytes compacto"""
    return struct.pack("%sf" % len(vector), *vector)

# Cargar variables de entorno
load_dotenv()

# Configurar OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

# DATA_PATH = "catalogos_md"
DATA_PATH = "test_catalogo"

def create_database():
    # Crear el directorio data si no existe
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Ruta de la base de datos
    db_path = data_dir / "catalogo.db"
    
    # Conectar a la base de datos (la creará si no existe)
    conn = sqlite3.connect(db_path)
    
    # Habilitar y cargar la extensión sqlite-vec
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    
    cursor = conn.cursor()
    
    # Crear tabla de control para la migración
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS db_version (
        version INTEGER PRIMARY KEY,
        fecha_migracion TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Verificar si ya hemos migrado a sqlite-vec
    cursor.execute('SELECT version FROM db_version')
    version = cursor.fetchone()
    
    if not version or version[0] < 1:
        logging.info("Primera ejecución con sqlite-vec, eliminando tablas antiguas...")
        # Eliminar tablas existentes si existen
        cursor.execute('DROP TABLE IF EXISTS chunks')
        cursor.execute('DROP TABLE IF EXISTS chunks_metadata')
        cursor.execute('DROP TABLE IF EXISTS chunks_embeddings')
        
        # Crear tabla de metadatos
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS chunks_metadata (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ruta_archivo TEXT NOT NULL,
            titulo TEXT NOT NULL,
            contenido TEXT NOT NULL,
            inicio INTEGER NOT NULL,
            fin INTEGER NOT NULL,
            fecha_creacion TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Crear tabla virtual para los embeddings
        cursor.execute('''
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_embeddings USING vec0(
            id INTEGER PRIMARY KEY,
            embedding FLOAT[1536]  -- OpenAI ada-002 usa 1536 dimensiones
        )
        ''')
        
        # Crear índices para mejorar el rendimiento
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_chunks_titulo ON chunks_metadata(titulo)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_chunks_ruta ON chunks_metadata(ruta_archivo)')
        
        # Marcar como migrado
        cursor.execute('INSERT OR REPLACE INTO db_version (version) VALUES (1)')
        conn.commit()
        logging.info("Migración a sqlite-vec completada")
    else:
        logging.info("Base de datos ya migrada a sqlite-vec, manteniendo datos existentes")
    
    conn.close()
    logging.info(f"Base de datos lista en: {db_path}")

def chunker(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    """
    Divide un texto en chunks de tamaño fijo con solapamiento.
    
    Args:
        text: El texto a dividir
        chunk_size: Tamaño de cada chunk
        overlap: Número de caracteres que se solapan entre chunks consecutivos
    
    Returns:
        Lista de chunks
    """
    if not text:
        return []
        
    chunks = []
    start = 0
    
    while start < len(text):
        # Si es el primer chunk, no necesitamos retroceder
        if start == 0:
            end = chunk_size
        else:
            # Para los siguientes chunks, retrocedemos 'overlap' caracteres
            end = start + chunk_size - overlap
            
        # Si llegamos al final del texto
        if end > len(text):
            end = len(text)
            
        # Añadir el chunk actual
        chunks.append(text[start:end])
        
        # Si llegamos al final del texto, terminamos
        if end == len(text):
            break
            
        # Actualizar el punto de inicio para el siguiente chunk
        start = end
        
    return chunks

def get_embedding(text: str, test_mode: bool = False) -> np.ndarray:
    """
    Obtiene el embedding de un texto usando la API de OpenAI.
    
    Args:
        text: El texto para el que queremos obtener el embedding
        
    Returns:
        Array numpy con el embedding
    """
    try:
        if test_mode:
            logging.info(f"Llamando a OpenAI API para obtener embedding de texto de {len(text)} caracteres")
            start_time = datetime.now()
        
        response = openai.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        
        if test_mode:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            logging.info(f"Respuesta recibida de OpenAI API en {duration:.2f} segundos")
        
        return np.array(response.data[0].embedding)
    except Exception as e:
        logging.error(f"Error al obtener embedding: {e}")
        raise

def cargar_documentos(data_path: str) -> Generator[Dict[str, str], None, None]:
    """
    Carga los documentos de manera eficiente usando un generador.
    """
    for archivo in Path(data_path).glob("**/*.md"):
        try:
            logging.info(f"Cargando archivo: {archivo}")
            with open(archivo, 'r', encoding='utf-8') as f:
                contenido = f.read()
            yield {
                'titulo': archivo.stem,
                'ruta_archivo': str(archivo),
                'contenido': contenido
            }
        except Exception as e:
            logging.error(f"Error al cargar {archivo}: {e}")

def documento_procesado(db: Database, ruta_archivo: str) -> bool:
    """
    Verifica si un documento ya ha sido procesado.
    
    Args:
        db: Instancia de la base de datos
        ruta_archivo: Ruta del archivo a verificar
        
    Returns:
        bool: True si el documento ya está procesado, False en caso contrario
    """
    conn = db.get_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
    SELECT COUNT(*) 
    FROM chunks_metadata 
    WHERE ruta_archivo = ?
    ''', (ruta_archivo,))
    
    count = cursor.fetchone()[0]
    conn.close()
    
    return count > 0

def procesar_documento(db: Database, documento: Dict[str, str], test_mode: bool = False) -> None:
    """
    Procesa un documento individual, lo divide en chunks y guarda sus embeddings.
    """
    # Verificar si el documento ya está procesado
    if documento_procesado(db, documento['ruta_archivo']):
        if test_mode:
            logging.info(f"El documento {documento['titulo']} ya está procesado, saltando...")
        return
    
    if test_mode:
        logging.info(f"Procesando documento: {documento['titulo']}")
        logging.info(f"Tamaño del documento: {len(documento['contenido'])} caracteres")
    
    # Dividir en chunks
    chunks = chunker(documento['contenido'])
    if test_mode:
        logging.info(f"Documento dividido en {len(chunks)} chunks")
    
    # Procesar cada chunk
    for i, chunk in enumerate(chunks, 1):
        try:
            if test_mode:
                logging.info(f"Procesando chunk {i}/{len(chunks)}")
                logging.info(f"Tamaño del chunk: {len(chunk)} caracteres")
            
            # Calcular el embedding
            embedding = get_embedding(chunk, test_mode)
            
            # Calcular las posiciones en el documento original
            inicio = documento['contenido'].find(chunk)
            fin = inicio + len(chunk)
            
            if test_mode:
                logging.info(f"Guardando chunk en la base de datos (posiciones {inicio}-{fin})")
            
            # Guardar el chunk con su embedding
            chunk_id = db.insert_chunk(
                ruta_archivo=documento['ruta_archivo'],
                titulo=documento['titulo'],
                contenido=chunk,
                embedding=embedding,
                inicio=inicio,
                fin=fin,
                test_mode=test_mode
            )
            
            if test_mode:
                logging.info(f"Chunk guardado con ID: {chunk_id}")
                
        except Exception as e:
            logging.error(f"Error al procesar chunk {i} en {documento['titulo']}: {e}")

def generate_rag(test_mode: bool = False):
    """
    Genera el RAG a partir de los documentos md.
    """
    # Seleccionar directorio de datos
    data_path = "test_catalogo" if test_mode else "catalogo_md"
    logging.info(f"Usando directorio de datos: {data_path}")
    
    # Crear la base de datos si no existe
    create_database()
    
    # Inicializar base de datos
    db = Database()
    
    # Procesar documentos
    documentos = cargar_documentos(data_path)
    for documento in tqdm(documentos, desc="Procesando documentos"):
        try:
            procesar_documento(db, documento, test_mode)
        except Exception as e:
            logging.error(f"Error al procesar {documento['titulo']}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Genera el RAG a partir de documentos markdown')
    parser.add_argument('-t', '--test', action='store_true', help='Ejecutar en modo test')
    args = parser.parse_args()
    
    generate_rag(test_mode=args.test)
