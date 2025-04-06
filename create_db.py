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
import hashlib

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
       - chunks_cache: Almacena hashes de chunks para caché
       - documentos_params: Almacena parámetros de procesamiento por documento
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
        
        # Crear tabla para la caché de hashes
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS chunks_cache (
            hash TEXT PRIMARY KEY,
            chunk_id INTEGER NOT NULL,
            fecha_creacion TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (chunk_id) REFERENCES chunks_metadata(id)
        )
        ''')
        
        # Crear tabla para almacenar parámetros de procesamiento por documento
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS documentos_params (
            ruta_archivo TEXT PRIMARY KEY,
            chunk_size INTEGER NOT NULL,
            overlap INTEGER NOT NULL,
            fecha_actualizacion TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Crear índices para mejorar el rendimiento
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_chunks_titulo ON chunks_metadata(titulo)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_chunks_ruta ON chunks_metadata(ruta_archivo)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_chunks_cache_hash ON chunks_cache(hash)')
        
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
    - Tabla chunks_cache: Almacena hashes de chunks para evitar recálculos
    - Tabla documentos_params: Almacena parámetros de procesamiento por documento
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
        
        # Crear tabla para la caché de hashes si no existe
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS chunks_cache (
            hash TEXT PRIMARY KEY,
            chunk_id INTEGER NOT NULL,
            fecha_creacion TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (chunk_id) REFERENCES chunks_metadata(id)
        )
        ''')
        
        # Crear tabla para almacenar parámetros de procesamiento por documento
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS documentos_params (
            ruta_archivo TEXT PRIMARY KEY,
            chunk_size INTEGER NOT NULL,
            overlap INTEGER NOT NULL,
            fecha_actualizacion TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Crear índices si no existen
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_chunks_titulo ON chunks_metadata(titulo)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_chunks_ruta ON chunks_metadata(ruta_archivo)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_chunks_cache_hash ON chunks_cache(hash)')
        
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

def documento_procesado(db: Database, ruta_archivo: str, chunk_size: int = 1000, overlap: int = 200, preservar_embeddings: bool = False) -> Tuple[bool, int, int]:
    """
    Verifica el estado de procesamiento de un documento en la base de datos.
    
    Esta función:
    1. Verifica si el documento ya ha sido procesado
    2. Comprueba si los parámetros de procesamiento (chunk_size y overlap) han cambiado
    3. Si los parámetros son diferentes, actualiza la información en documentos_params
    4. Determina cuántos chunks ya están procesados con los parámetros actuales
    5. Si preservar_embeddings es True, intenta reutilizar los embeddings existentes
    
    Args:
        db (Database): Instancia de la base de datos
        ruta_archivo (str): Ruta del archivo a verificar
        chunk_size (int): Tamaño de chunk actual a procesar
        overlap (int): Tamaño de solapamiento actual a procesar
        preservar_embeddings (bool): Si True, intenta preservar embeddings existentes
    
    Returns:
        Tuple[bool, int, int]: 
            - bool: True si el documento está completamente procesado
            - int: Número de chunks existentes en la base de datos
            - int: ID del último chunk procesado
    """
    conn = db.get_connection()
    cursor = conn.cursor()
    
    try:
        # Verificar si tenemos parámetros almacenados para este documento
        cursor.execute('''
        SELECT chunk_size, overlap FROM documentos_params 
        WHERE ruta_archivo = ?
        ''', (ruta_archivo,))
        
        params_existentes = cursor.fetchone()
        
        # Obtener contenido del documento para calcular el número de chunks
        with open(ruta_archivo, 'r', encoding='utf-8') as f:
            contenido = f.read()
            
        # Calcular nuevo número de chunks con los parámetros actuales
        nuevos_chunks = len(chunker(contenido, chunk_size=chunk_size, overlap=overlap))
        
        # Obtener chunks existentes
        cursor.execute('''
        SELECT COUNT(*), COALESCE(MAX(id), 0)
        FROM chunks_metadata 
        WHERE ruta_archivo = ?
        ''', (ruta_archivo,))
        
        chunks_existentes, ultimo_id = cursor.fetchone()
        
        # Si no hay parámetros almacenados o son diferentes a los actuales
        if not params_existentes:
            # Documento no procesado previamente o sin parámetros registrados
            # Registrar los nuevos parámetros
            cursor.execute("BEGIN TRANSACTION")
            cursor.execute('''
            INSERT OR REPLACE INTO documentos_params 
            (ruta_archivo, chunk_size, overlap, fecha_actualizacion)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            ''', (ruta_archivo, chunk_size, overlap))
            cursor.execute("COMMIT")
            
            logging.info(f"Registrados parámetros para {ruta_archivo}: chunk_size={chunk_size}, overlap={overlap}")
            return chunks_existentes, ultimo_id
            
        else:
            # Tenemos parámetros almacenados, verificar si son diferentes
            stored_chunk_size, stored_overlap = params_existentes
            
            if stored_chunk_size != chunk_size or stored_overlap != overlap:
                logging.warning(f"Parámetros para {ruta_archivo} han cambiado:")
                logging.warning(f"  Almacenados: chunk_size={stored_chunk_size}, overlap={stored_overlap}")
                logging.warning(f"  Actuales: chunk_size={chunk_size}, overlap={overlap}")
                
                # Calcular número de chunks con los parámetros almacenados
                chunks_anteriores = len(chunker(contenido, chunk_size=stored_chunk_size, overlap=stored_overlap))
                
                if chunks_existentes == chunks_anteriores:
                    # El documento está completamente procesado con los parámetros anteriores
                    logging.warning(f"El documento estaba completamente procesado con los parámetros anteriores ({chunks_existentes}/{chunks_anteriores} chunks)")
                    
                    if preservar_embeddings:
                        logging.info(f"Modo preservar embeddings activado: se intentarán reutilizar embeddings existentes")
                        
                        # Actualizar los parámetros
                        cursor.execute("BEGIN TRANSACTION")
                        cursor.execute('''
                        UPDATE documentos_params 
                        SET chunk_size = ?, overlap = ?, fecha_actualizacion = CURRENT_TIMESTAMP
                        WHERE ruta_archivo = ?
                        ''', (chunk_size, overlap, ruta_archivo))
                        cursor.execute("COMMIT")
                        
                        # Indicamos que hay 0 chunks procesados para que se procesen todos,
                        # pero procesar_documento intentará reutilizar los embeddings existentes
                        return 0, 0
                    
                    # Decidimos si es más eficiente mantener los chunks existentes o reprocesar
                    if nuevos_chunks < chunks_existentes:
                        # Es más eficiente reprocesar (menos chunks totales)
                        logging.warning(f"Los nuevos parámetros resultan en menos chunks ({nuevos_chunks} vs {chunks_existentes})")
                        logging.warning(f"Se recomienda reprocesar el documento desde cero para mayor eficiencia")
                        
                        # Aquí NO eliminaremos los chunks automáticamente, solo damos la recomendación
                        # Actualizamos los parámetros
                        cursor.execute("BEGIN TRANSACTION")
                        cursor.execute('''
                        UPDATE documentos_params 
                        SET chunk_size = ?, overlap = ?, fecha_actualizacion = CURRENT_TIMESTAMP
                        WHERE ruta_archivo = ?
                        ''', (chunk_size, overlap, ruta_archivo))
                        cursor.execute("COMMIT")
                        
                        # Devolver 0 para que se procese desde el inicio
                        return 0, 0
                    else:
                        # Los chunks existentes se mantienen, se procesarán los nuevos
                        logging.warning(f"Reutilizando chunks existentes y procesando nuevos chunks según sea necesario")
                        
                        # Actualizar los parámetros
                        cursor.execute("BEGIN TRANSACTION")
                        cursor.execute('''
                        UPDATE documentos_params 
                        SET chunk_size = ?, overlap = ?, fecha_actualizacion = CURRENT_TIMESTAMP
                        WHERE ruta_archivo = ?
                        ''', (chunk_size, overlap, ruta_archivo))
                        cursor.execute("COMMIT")
                        
                        # Devolver los chunks existentes
                        return chunks_existentes, ultimo_id
                else:
                    # El documento estaba parcialmente procesado
                    logging.warning(f"Documento parcialmente procesado con parámetros anteriores ({chunks_existentes}/{chunks_anteriores} chunks)")
                    
                    if preservar_embeddings:
                        logging.info(f"Modo preservar embeddings activado: se intentarán reutilizar embeddings existentes")
                    
                    # Actualizar los parámetros
                    cursor.execute("BEGIN TRANSACTION")
                    cursor.execute('''
                    UPDATE documentos_params 
                    SET chunk_size = ?, overlap = ?, fecha_actualizacion = CURRENT_TIMESTAMP
                    WHERE ruta_archivo = ?
                    ''', (chunk_size, overlap, ruta_archivo))
                    cursor.execute("COMMIT")
                    
                    # Si preservar_embeddings, indicamos 0 chunks pero procesar_documento 
                    # intentará reutilizar los existentes
                    return 0, 0
            else:
                # Los parámetros no han cambiado
                logging.info(f"Parámetros no han cambiado para {ruta_archivo}")
                return chunks_existentes, ultimo_id
        
    except Exception as e:
        logging.error(f"Error al verificar estado de documento: {e}")
        if 'cursor' in locals() and cursor:
            try:
                cursor.execute("ROLLBACK")
            except:
                pass
        raise
    finally:
        conn.close()

def procesar_documento(db: Database, documento: Dict[str, str], chunk_size: int = 1000, overlap: int = 200, test_mode: bool = False, preservar_embeddings: bool = False) -> None:
    """
    Procesa un documento individual, dividiéndolo en chunks y guardando sus embeddings.
    
    El proceso incluye:
    1. Dividir el documento en chunks
    2. Verificar chunks existentes en la base de datos
    3. Procesar solo los chunks faltantes usando caché de hashes
    4. Almacenar nuevos chunks y embeddings
    5. Preservar embeddings ya calculados cuando sea posible
    
    Args:
        db (Database): Instancia de la base de datos
        documento (Dict[str, str]): Documento a procesar
        chunk_size (int): Tamaño de cada chunk en caracteres
        overlap (int): Número de caracteres que se solapan entre chunks
        test_mode (bool): Si True, muestra información detallada
        preservar_embeddings (bool): Si True, intenta preservar embeddings existentes
    """
    logging.info(f"Procesando documento: {documento['titulo']}")
    
    if test_mode:
        logging.info(f"Tamaño del documento: {len(documento['contenido'])} caracteres")
        logging.info(f"Parámetros: chunk_size={chunk_size}, overlap={overlap}")
    
    # Dividir en chunks
    chunks = chunker(documento['contenido'], chunk_size=chunk_size, overlap=overlap)
    total_chunks = len(chunks)
    logging.info(f"Documento dividido en {total_chunks} chunks")
    
    # Verificar chunks existentes con nueva lógica de preservación
    chunks_existentes, _ = documento_procesado(db, documento['ruta_archivo'], chunk_size, overlap, preservar_embeddings)
    
    if chunks_existentes == total_chunks:
        logging.info(f"Documento {documento['titulo']} ya está completamente procesado ({chunks_existentes}/{total_chunks} chunks)")
        return
    elif chunks_existentes > 0:
        logging.warning(f"Documento {documento['titulo']} parcialmente procesado ({chunks_existentes}/{total_chunks} chunks)")
        logging.info(f"Continuando desde el chunk {chunks_existentes + 1}")
    
    # Conexión para verificar caché
    conn = db.get_connection()
    cursor = conn.cursor()
    
    try:
        # Si tenemos que procesar todos los chunks desde el inicio pero hay chunks existentes,
        # intentamos reutilizar los embeddings que ya calculamos previamente
        if chunks_existentes == 0 and total_chunks > 0:
            # Verificar si hay chunks previos para este documento
            cursor.execute('''
            SELECT COUNT(*) FROM chunks_metadata
            WHERE ruta_archivo = ?
            ''', (documento['ruta_archivo'],))
            
            chunks_previos = cursor.fetchone()[0]
            
            if chunks_previos > 0:
                logging.info(f"Encontrados {chunks_previos} chunks previamente procesados que intentaremos reutilizar")
                
                # Cargar todos los chunks previos en memoria
                cursor.execute('''
                SELECT contenido, id FROM chunks_metadata
                WHERE ruta_archivo = ?
                ''', (documento['ruta_archivo'],))
                
                chunks_previos_dict = {row[0]: row[1] for row in cursor.fetchall()}
                logging.info(f"Cargados {len(chunks_previos_dict)} chunks previos en memoria")
        else:
            chunks_previos_dict = {}
        
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
                
                # Calcular hash del chunk
                chunk_hash = calculate_chunk_hash(chunk)
                
                # Verificar si el hash ya existe en la caché
                cursor.execute('''
                SELECT c.chunk_id, m.ruta_archivo
                FROM chunks_cache c
                JOIN chunks_metadata m ON c.chunk_id = m.id
                WHERE c.hash = ?
                ''', (chunk_hash,))
                
                resultado = cursor.fetchone()
                
                if resultado:
                    chunk_id, ruta_archivo = resultado
                    if test_mode:
                        logging.info(f"Chunk encontrado en caché (ID: {chunk_id}, Archivo: {ruta_archivo})")
                    continue
                
                # Verificar si este chunk existe en la colección de chunks previos
                chunk_id_previo = chunks_previos_dict.get(chunk, None)
                
                if chunk_id_previo is not None:
                    # El chunk ya existía, podemos reutilizar su embedding
                    if test_mode:
                        logging.info(f"Reutilizando embedding de chunk previo (ID: {chunk_id_previo})")
                    
                    # Iniciar transacción
                    cursor.execute("BEGIN TRANSACTION")
                    
                    # Crear un nuevo registro en chunks_metadata
                    cursor.execute('''
                    INSERT INTO chunks_metadata 
                    (ruta_archivo, titulo, contenido, inicio, fin)
                    VALUES (?, ?, ?, ?, ?)
                    ''', (documento['ruta_archivo'], documento['titulo'], chunk, inicio, fin))
                    
                    # Obtener el ID del nuevo registro
                    cursor.execute("SELECT last_insert_rowid()")
                    nuevo_chunk_id = cursor.fetchone()[0]
                    
                    # Copiar el embedding del chunk previo
                    cursor.execute('''
                    INSERT INTO chunks_embeddings (id, embedding)
                    SELECT ?, embedding FROM chunks_embeddings WHERE id = ?
                    ''', (nuevo_chunk_id, chunk_id_previo))
                    
                    # Añadir a la caché
                    cursor.execute('''
                    INSERT INTO chunks_cache (hash, chunk_id)
                    VALUES (?, ?)
                    ''', (chunk_hash, nuevo_chunk_id))
                    
                    cursor.execute("COMMIT")
                    
                    if test_mode:
                        logging.info(f"Embedding reutilizado con nuevo ID: {nuevo_chunk_id}")
                    
                    continue
                
                # Si no está en caché ni en chunks previos, calcular el embedding
                embedding = get_embedding(chunk, test_mode)
                
                if test_mode:
                    logging.info(f"Guardando chunk en la base de datos (posiciones {inicio}-{fin})")
                
                # Guardar el chunk con su embedding (esto también guarda el hash en caché)
                chunk_id = db.insert_chunk(
                    ruta_archivo=documento['ruta_archivo'],
                    titulo=documento['titulo'],
                    contenido=chunk,
                    embedding=embedding,
                    inicio=inicio,
                    fin=fin,
                    test_mode=test_mode
                )
                
            except Exception as e:
                logging.error(f"Error al procesar chunk {i} en {documento['titulo']}: {e}")
                raise
    finally:
        conn.close()

def generate_rag(test_mode: bool = False, chunk_size: int = 1000, overlap: int = 200, preservar_embeddings: bool = False):
    """
    Genera el sistema RAG procesando todos los documentos Markdown.
    
    Este proceso:
    1. Configura la base de datos
    2. Carga y procesa los documentos
    3. Genera y almacena embeddings
    4. Opcionalmente preserva embeddings existentes
    
    Args:
        test_mode (bool): Si True, usa un directorio de prueba y muestra más información
        chunk_size (int): Tamaño de cada chunk en caracteres
        overlap (int): Número de caracteres que se solapan entre chunks
        preservar_embeddings (bool): Si True, intenta preservar embeddings existentes
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
            procesar_documento(
                db=db, 
                documento=documento, 
                chunk_size=chunk_size, 
                overlap=overlap, 
                test_mode=test_mode,
                preservar_embeddings=preservar_embeddings
            )
        except Exception as e:
            logging.error(f"Error al procesar {documento['titulo']}: {e}")

def calculate_chunk_hash(chunk: str) -> str:
    """
    Calcula el hash SHA-256 de un chunk de texto.
    
    Args:
        chunk (str): Texto del chunk
        
    Returns:
        str: Hash SHA-256 hexadecimal del chunk
    """
    return hashlib.sha256(chunk.encode('utf-8')).hexdigest()

def recalcular_cache():
    """
    Recalcula completamente la caché de hashes para todos los chunks existentes.
    
    Esta función:
    1. Elimina todos los registros de la tabla chunks_cache
    2. Recorre todos los chunks en chunks_metadata
    3. Calcula el hash para cada chunk
    4. Inserta los nuevos hashes en la caché
    
    Es útil cuando se ha cambiado el algoritmo de hash o se han procesado
    chunks sin registrar sus hashes.
    """
    logging.info("Iniciando recálculo completo de la caché de hashes...")
    
    # Crear la base de datos si no existe
    create_database()
    
    # Inicializar base de datos
    db = Database()
    conn = db.get_connection()
    cursor = conn.cursor()
    
    try:
        # Iniciar transacción
        cursor.execute("BEGIN TRANSACTION")
        
        # Eliminar todos los registros de la caché
        cursor.execute("DELETE FROM chunks_cache")
        logging.info("Caché de hashes eliminada")
        
        # Obtener todos los chunks existentes
        cursor.execute("""
        SELECT id, contenido FROM chunks_metadata
        """)
        
        chunks = cursor.fetchall()
        total_chunks = len(chunks)
        logging.info(f"Encontrados {total_chunks} chunks para procesar")
        
        # Procesar cada chunk
        for i, (chunk_id, contenido) in enumerate(chunks, 1):
            # Calcular hash
            chunk_hash = calculate_chunk_hash(contenido)
            
            # Insertar en la caché
            cursor.execute("""
            INSERT INTO chunks_cache (hash, chunk_id)
            VALUES (?, ?)
            """, (chunk_hash, chunk_id))
            
            # Mostrar progreso
            if i % 1000 == 0 or i == total_chunks:
                logging.info(f"Procesados {i}/{total_chunks} chunks")
        
        # Confirmar transacción
        cursor.execute("COMMIT")
        logging.info("Caché de hashes recalculada correctamente")
        
    except Exception as e:
        # En caso de error, revertir cambios
        cursor.execute("ROLLBACK")
        logging.error(f"Error al recalcular la caché: {e}")
        raise e
    finally:
        conn.close()

def main():
    """
    Función principal que coordina la ejecución del script.
    
    Procesa los argumentos de línea de comandos:
    --test (-t): Ejecuta en modo prueba con logging detallado
    --init (-i): Inicializa la base de datos desde cero
    --chunk-size (-c): Tamaño de cada chunk en caracteres (default: 1000)
    --overlap (-o): Número de caracteres que se solapan entre chunks (default: 200)
    --recalcular-cache (-r): Recalcula la caché de hashes para todos los chunks
    --forzar-recalculo (-f): Fuerza el recálculo de todos los chunks ignorando los existentes
    --preservar-embeddings (-p): Intenta preservar embeddings existentes cuando cambian los parámetros
    
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
    parser.add_argument('-r', '--recalcular-cache', action='store_true', help='Recalcula la caché de hashes para todos los chunks')
    parser.add_argument('-f', '--forzar-recalculo', action='store_true', help='Fuerza el recálculo de todos los chunks ignorando los existentes')
    parser.add_argument('-p', '--preservar-embeddings', action='store_true', help='Preservar embeddings existentes cuando cambian los parámetros')
    args = parser.parse_args()
    
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO if args.test else logging.WARNING,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    logging.getLogger("openai").setLevel(logging.WARNING)
    
    # Verificar entorno
    verificar_entorno()
    
    # Recalcular caché si se solicita
    if args.recalcular_cache:
        logging.info("Iniciando recálculo de caché...")
        recalcular_cache()
        return
    
    # Inicializar base de datos si se solicita
    if args.init:
        logging.warning("¡ATENCIÓN! Se eliminarán todos los datos existentes.")
        confirmacion = input("¿Estás seguro de querer inicializar la base de datos desde cero? (s/n): ")
        if confirmacion.lower() != 's':
            logging.info("Inicialización cancelada.")
            return
        init_database()
    else:
        # Crear base de datos si no existe
        create_database()
    
    # Si se fuerza el recálculo, eliminar los parámetros de documentos procesados
    if args.forzar_recalculo:
        conn = apsw.Connection(str(Path("data") / "catalogo.db"))
        cursor = conn.cursor()
        cursor.execute("DELETE FROM documentos_params")
        conn.close()
        logging.warning("Se forzará el reprocesamiento de todos los documentos")
    
    # Generar RAG con los parámetros especificados
    generate_rag(
        test_mode=args.test, 
        chunk_size=args.chunk_size, 
        overlap=args.overlap,
        preservar_embeddings=args.preservar_embeddings
    )
    
    # Si no se solicita recalcular la caché explícitamente pero se ha inicializado la base de datos,
    # verificar si la caché está incompleta y recalcularla si es necesario
    if not args.recalcular_cache and not args.init:
        conn = apsw.Connection(str(Path("data") / "catalogo.db"))
        cursor = conn.cursor()
        
        # Contar chunks y registros en caché
        cursor.execute('''
        SELECT COUNT(*) FROM chunks_metadata
        ''')
        total_chunks = cursor.fetchone()[0]
        
        cursor.execute('''
        SELECT COUNT(*) FROM chunks_cache
        ''')
        total_cache = cursor.fetchone()[0]
        
        conn.close()
        
        # Si hay menos registros en la caché que chunks, recalcular
        if total_chunks > 0 and total_cache < total_chunks:
            logging.warning(f"Caché incompleta: {total_cache}/{total_chunks} chunks en caché")
            logging.info("Recalculando caché automáticamente...")
            recalcular_cache()

if __name__ == "__main__":
    main()
