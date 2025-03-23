"""
Script para la creación y gestión de una base de datos vectorial SQLite para un sistema RAG.

Este script implementa la creación y gestión de una base de datos SQLite que almacena
documentos y sus embeddings vectoriales para su uso en un sistema de Recuperación Aumentada
por Generación (RAG). Utiliza la extensión sqlite-vec para manejar vectores eficientemente.

Características principales:
- Creación y gestión de base de datos SQLite con soporte vectorial
- Procesamiento de documentos Markdown
- Generación de embeddings usando OpenAI
- División de textos en chunks con solapamiento
- Almacenamiento eficiente de metadatos y embeddings

Requisitos:
- Python 3.6+
- OpenAI API key configurada en variables de entorno
- Extensión sqlite-vec instalada
- Dependencias listadas en requirements.txt

Autor: RCS
Fecha: 2025-03-22
"""

import os
from pathlib import Path
from typing import List, Dict, Generator, Tuple
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
import apsw
import platform

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Cargar variables de entorno
load_dotenv()

# Configurar OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

# DATA_PATH = "catalogos_md"
DATA_PATH = "test_catalogo"

def verificar_entorno():
    """
    Verifica que el entorno está correctamente configurado para la ejecución del script.
    
    Realiza las siguientes verificaciones:
    1. Sistema operativo y versión de Python
    2. Disponibilidad y carga de la extensión sqlite-vec
    3. Acceso al directorio de datos
    4. Configuración de la API key de OpenAI
    
    Raises:
        Exception: Si alguna de las verificaciones falla
        ValueError: Si no se encuentra la API key de OpenAI
    """
    logging.info(f"Sistema operativo: {platform.system()} {platform.release()}")
    logging.info(f"Python: {platform.python_version()}")
    
    try:
        # Verificar que podemos cargar la extensión sqlite-vec
        conn = apsw.Connection(':memory:')
        conn.enableloadextension(True)
        sqlite_vec.load(conn)
        conn.enableloadextension(False)
        conn.close()
        logging.info("✓ Extensión sqlite-vec cargada correctamente")
    except Exception as e:
        logging.error(f"✗ Error al cargar sqlite-vec: {e}")
        raise
    
    # Verificar que tenemos acceso al directorio de datos
    data_dir = Path("data")
    try:
        data_dir.mkdir(exist_ok=True)
        logging.info("✓ Directorio de datos accesible")
    except Exception as e:
        logging.error(f"✗ Error al acceder al directorio de datos: {e}")
        raise
    
    # Verificar que tenemos la API key de OpenAI
    if not os.getenv("OPENAI_API_KEY"):
        logging.error("✗ No se encontró OPENAI_API_KEY en las variables de entorno")
        raise ValueError("OPENAI_API_KEY no configurada")
    logging.info("✓ API key de OpenAI configurada")

def serialize(vector: List[float]) -> bytes:
    """
    Serializa una lista de números flotantes en un formato de bytes compacto.
    
    Args:
        vector (List[float]): Lista de números flotantes a serializar
        
    Returns:
        bytes: Datos serializados en formato de bytes
    """
    return struct.pack("%sf" % len(vector), *vector)

def init_database():
    """
    Inicializa la base de datos desde cero, eliminando todas las tablas existentes.
    
    Esta función:
    1. Elimina la base de datos si existe
    2. Crea una nueva base de datos
    3. Configura la extensión sqlite-vec
    4. Crea las tablas necesarias:
       - chunks_metadata: Almacena metadatos de los fragmentos de texto
       - chunks_embeddings: Almacena los vectores de embedding
    5. Crea índices para optimizar las consultas
    
    Raises:
        Exception: Si ocurre algún error durante la inicialización
    """
    # Crear el directorio data si no existe
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Ruta de la base de datos
    db_path = data_dir / "catalogo.db"
    
    # Eliminar la base de datos si existe
    if db_path.exists():
        logging.info("Eliminando base de datos existente...")
        db_path.unlink()
    
    # Conectar a la base de datos (la creará si no existe)
    conn = apsw.Connection(str(db_path))
    
    # Habilitar y cargar la extensión sqlite-vec
    conn.enableloadextension(True)
    sqlite_vec.load(conn)
    conn.enableloadextension(False)
    
    cursor = conn.cursor()
    
    try:
        # Iniciar transacción
        cursor.execute("BEGIN TRANSACTION")
        
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
        
        # Confirmar transacción
        cursor.execute("COMMIT")
        
    except Exception as e:
        # En caso de error, revertir cambios
        cursor.execute("ROLLBACK")
        raise e
    finally:
        conn.close()
        logging.info(f"Base de datos inicializada en: {db_path}")

def create_database():
    """
    Crea la base de datos si no existe, preservando los datos existentes.
    
    A diferencia de init_database(), esta función:
    1. No elimina la base de datos existente
    2. Crea las tablas solo si no existen
    3. Preserva todos los datos existentes
    
    La estructura de la base de datos incluye:
    - Tabla chunks_metadata: Almacena metadatos de los fragmentos
    - Tabla chunks_embeddings: Almacena vectores de embedding
    - Índices para optimización de consultas
    
    Raises:
        Exception: Si ocurre algún error durante la creación
    """
    # Crear el directorio data si no existe
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Ruta de la base de datos
    db_path = data_dir / "catalogo.db"
    
    # Conectar a la base de datos (la creará si no existe)
    conn = apsw.Connection(str(db_path))
    
    # Habilitar y cargar la extensión sqlite-vec
    conn.enableloadextension(True)
    sqlite_vec.load(conn)
    conn.enableloadextension(False)
    
    cursor = conn.cursor()
    
    try:
        # Iniciar transacción
        cursor.execute("BEGIN TRANSACTION")
        
        # Crear tabla de metadatos si no existe
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
        
        # Crear tabla virtual para los embeddings si no existe
        cursor.execute('''
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_embeddings USING vec0(
            id INTEGER PRIMARY KEY,
            embedding FLOAT[1536]  -- OpenAI text-embedding-3-large usa 3072 pero lo limitamos a1536 dimensiones
        )
        ''')
        
        # Crear índices si no existen
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_chunks_titulo ON chunks_metadata(titulo)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_chunks_ruta ON chunks_metadata(ruta_archivo)')
        
        # Confirmar transacción
        cursor.execute("COMMIT")
        
    except Exception as e:
        # En caso de error, revertir cambios
        cursor.execute("ROLLBACK")
        raise e
    finally:
        conn.close()
        logging.info(f"Base de datos lista en: {db_path}")

def chunker(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    """
    Divide un texto en fragmentos (chunks) de tamaño fijo con solapamiento.
    
    El solapamiento entre chunks ayuda a mantener el contexto y evitar la pérdida
    de información en los límites de los fragmentos.
    
    Args:
        text (str): Texto a dividir
        chunk_size (int): Tamaño de cada fragmento en caracteres
        overlap (int): Número de caracteres que se solapan entre fragmentos consecutivos
    
    Returns:
        list[str]: Lista de fragmentos de texto
    
    Example:
        >>> texto = "Este es un texto de ejemplo para dividir"
        >>> chunks = chunker(texto, chunk_size=10, overlap=3)
        >>> print(chunks)
        ['Este es un', 'un texto de', 'de ejemplo', 'lo para div']
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
    Obtiene el vector de embedding para un texto usando la API de OpenAI.
    
    OLD: Utiliza el modelo text-embedding-ada-002 de OpenAI para generar
    embeddings de 1536 dimensiones.

    NEW: Utiliza el modelo text-embedding-3-large de OpenAI para generar
    embeddings de 1536 dimensiones.
    
    Args:
        text (str): Texto para el que se quiere obtener el embedding
        test_mode (bool): Si True, muestra información detallada del proceso
    
    Returns:
        np.ndarray: Vector de embedding de dimensión 1536
    
    Raises:
        Exception: Si hay un error en la llamada a la API de OpenAI
    """
    try:
        if test_mode:
            logging.info(f"Llamando a OpenAI API para obtener embedding de texto de {len(text)} caracteres")
            start_time = datetime.now()
        
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("openai").setLevel(logging.WARNING)
        response = openai.embeddings.create(
            # model="text-embedding-ada-002",
            model="text-embedding-3-large",
            dimensions=1536,
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
    Carga documentos Markdown de forma eficiente usando un generador.
    
    Recorre recursivamente el directorio especificado buscando archivos .md
    y los carga uno a uno para evitar consumo excesivo de memoria.
    
    Args:
        data_path (str): Ruta al directorio que contiene los documentos
    
    Yields:
        Dict[str, str]: Diccionario con los siguientes campos:
            - titulo: Nombre del archivo sin extensión
            - ruta_archivo: Ruta completa al archivo
            - contenido: Contenido del archivo
    
    Raises:
        Exception: Si hay errores al leer algún archivo
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

def documento_procesado(db: Database, ruta_archivo: str) -> Tuple[bool, int, int]:
    """
    Verifica el estado de procesamiento de un documento en la base de datos.
    
    Args:
        db (Database): Instancia de la base de datos
        ruta_archivo (str): Ruta del archivo a verificar
    
    Returns:
        Tuple[bool, int, int]: 
            - bool: True si el documento está completamente procesado
            - int: Número de chunks existentes en la base de datos
            - int: ID del último chunk procesado
    """
    conn = db.get_connection()
    cursor = conn.cursor()
    
    try:
        # Obtener el número de chunks y el último ID
        cursor.execute('''
        SELECT COUNT(*), COALESCE(MAX(id), 0)
        FROM chunks_metadata 
        WHERE ruta_archivo = ?
        ''', (ruta_archivo,))
        
        chunks_existentes, ultimo_id = cursor.fetchone()
        return chunks_existentes, ultimo_id
        
    finally:
        conn.close()

def procesar_documento(db: Database, documento: Dict[str, str], chunk_size: int = 1000, overlap: int = 200, test_mode: bool = False) -> None:
    """
    Procesa un documento individual, dividiéndolo en chunks y guardando sus embeddings.
    
    El proceso incluye:
    1. Dividir el documento en chunks
    2. Verificar chunks existentes en la base de datos
    3. Procesar solo los chunks faltantes
    4. Almacenar nuevos chunks y embeddings
    
    Args:
        db (Database): Instancia de la base de datos
        documento (Dict[str, str]): Documento a procesar
        chunk_size (int): Tamaño de cada chunk en caracteres
        overlap (int): Número de caracteres que se solapan entre chunks
        test_mode (bool): Si True, muestra información detallada
    """
    logging.info(f"Procesando documento: {documento['titulo']}")
    
    if test_mode:
        logging.info(f"Tamaño del documento: {len(documento['contenido'])} caracteres")
        logging.info(f"Parámetros: chunk_size={chunk_size}, overlap={overlap}")
    
    # Dividir en chunks
    chunks = chunker(documento['contenido'], chunk_size=chunk_size, overlap=overlap)
    total_chunks = len(chunks)
    logging.info(f"Documento dividido en {total_chunks} chunks")
    
    # Verificar chunks existentes
    chunks_existentes, _ = documento_procesado(db, documento['ruta_archivo'])
    
    if chunks_existentes == total_chunks:
        logging.info(f"Documento {documento['titulo']} ya está completamente procesado ({chunks_existentes}/{total_chunks} chunks)")
        return
    elif chunks_existentes > 0:
        logging.warning(f"Documento {documento['titulo']} parcialmente procesado ({chunks_existentes}/{total_chunks} chunks)")
        logging.info(f"Continuando desde el chunk {chunks_existentes + 1}")
    
    # Procesar chunks faltantes
    for i, chunk in enumerate(chunks[chunks_existentes:], chunks_existentes + 1):
        try:
            if test_mode:
                logging.info(f"Procesando chunk {i}/{total_chunks}")
                logging.info(f"Tamaño del chunk: {len(chunk)} caracteres")
            else:
                # Mostrar progreso cada 100 chunks o al finalizar
                if i % 100 == 0 or i == total_chunks:
                    logging.info(f"Procesados {i}/{total_chunks} chunks")
            
            # Calcular las posiciones en el documento original
            inicio = documento['contenido'].find(chunk)
            fin = inicio + len(chunk)
            
            # Verificar si el chunk ya existe en la base de datos
            chunks_existentes = db.get_chunks_archivo(documento['ruta_archivo'])
            chunk_existente = next(
                (c for c in chunks_existentes if c['inicio'] == inicio and c['fin'] == fin),
                None
            )
            
            if chunk_existente:
                if test_mode:
                    logging.info(f"Chunk ya existe en la base de datos (ID: {chunk_existente['id']})")
                continue
            
            # Calcular el embedding solo si el chunk no existe
            embedding = get_embedding(chunk, test_mode)
            
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
            raise

def generate_rag(test_mode: bool = False, chunk_size: int = 1000, overlap: int = 200):
    """
    Genera el sistema RAG procesando todos los documentos Markdown.
    
    Este proceso:
    1. Configura la base de datos
    2. Carga y procesa los documentos
    3. Genera y almacena embeddings
    
    Args:
        test_mode (bool): Si True, usa un directorio de prueba y muestra más información
        chunk_size (int): Tamaño de cada chunk en caracteres
        overlap (int): Número de caracteres que se solapan entre chunks
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
            procesar_documento(db, documento, chunk_size, overlap, test_mode)
        except Exception as e:
            logging.error(f"Error al procesar {documento['titulo']}: {e}")

def main():
    """
    Función principal que coordina la ejecución del script.
    
    Procesa los argumentos de línea de comandos:
    --test (-t): Ejecuta en modo prueba con logging detallado
    --init (-i): Inicializa la base de datos desde cero
    --chunk-size (-c): Tamaño de cada chunk en caracteres (default: 1000)
    --overlap (-o): Número de caracteres que se solapan entre chunks (default: 200)
    
    El proceso completo incluye:
    1. Verificación del entorno
    2. Inicialización/creación de la base de datos
    3. Generación del sistema RAG
    """
    parser = argparse.ArgumentParser(description='Genera embeddings para documentos markdown')
    parser.add_argument('-t', '--test', action='store_true', help='Modo test con logging detallado')
    parser.add_argument('-i', '--init', action='store_true', help='Inicializar base de datos desde cero')
    parser.add_argument('-c', '--chunk-size', type=int, default=1000, help='Tamaño de cada chunk en caracteres (default: 1000)')
    parser.add_argument('-o', '--overlap', type=int, default=200, help='Número de caracteres que se solapan entre chunks (default: 200)')
    args = parser.parse_args()
    
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO if args.test else logging.WARNING,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    logging.getLogger("openai").setLevel(logging.WARNING)
    
    # Verificar entorno
    verificar_entorno()
    
    # Inicializar base de datos si se solicita
    if args.init:
        init_database()
    else:
        # Crear base de datos si no existe
        create_database()
    
    # Generar RAG con los parámetros especificados
    generate_rag(test_mode=args.test, chunk_size=args.chunk_size, overlap=args.overlap)

if __name__ == "__main__":
    main()
