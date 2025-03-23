"""
Módulo para la gestión de la base de datos vectorial SQLite.

Este módulo implementa la clase Database que maneja todas las operaciones de base de datos
para el sistema RAG, incluyendo:
- Almacenamiento de chunks de texto y sus embeddings
- Búsqueda de similitud vectorial
- Gestión de metadatos de documentos
- Operaciones CRUD básicas

La base de datos utiliza la extensión sqlite-vec para realizar búsquedas de similitud
vectorial eficientes, permitiendo encontrar los chunks más relevantes para una consulta.

Autor: RCS
Fecha: 2024-03-22
"""

from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Any, Tuple
import numpy as np
import logging
import sqlite_vec
import struct
import apsw

def serialize(vector: List[float]) -> bytes:
    """
    Serializa una lista de números flotantes en un formato de bytes compacto.
    
    Esta función es utilizada para convertir los vectores de embedding en un formato
    que puede ser almacenado y procesado eficientemente por sqlite-vec.
    
    Args:
        vector (List[float]): Lista de números flotantes a serializar
        
    Returns:
        bytes: Datos serializados en formato de bytes
    """
    return struct.pack("%sf" % len(vector), *vector)

class Database:
    """
    Clase para gestionar la base de datos vectorial SQLite.
    
    Esta clase maneja todas las operaciones de base de datos necesarias para el sistema RAG,
    incluyendo la inserción, recuperación y búsqueda de chunks de texto y sus embeddings.
    
    La base de datos utiliza dos tablas principales:
    - chunks_metadata: Almacena información sobre los fragmentos de texto
    - chunks_embeddings: Almacena los vectores de embedding
    
    Attributes:
        db_path (Path): Ruta al archivo de base de datos SQLite
    """
    
    def __init__(self):
        """
        Inicializa la conexión a la base de datos.
        
        La base de datos se encuentra en el directorio 'data' con el nombre 'catalogo.db'.
        """
        self.db_path = Path("data") / "catalogo.db"
        
    def get_connection(self):
        """
        Obtiene una conexión a la base de datos con la extensión sqlite-vec habilitada.
        
        Returns:
            apsw.Connection: Conexión a la base de datos configurada
        """
        conn = apsw.Connection(str(self.db_path))
        # Habilitar y cargar la extensión sqlite-vec
        conn.enableloadextension(True)
        sqlite_vec.load(conn)
        conn.enableloadextension(False)
        return conn
    
    def insert_chunk(self, ruta_archivo: str, titulo: str, contenido: str, 
                    embedding: np.ndarray, inicio: int, fin: int, test_mode: bool = False) -> int:
        """
        Inserta un chunk y su embedding en la base de datos.
        
        Esta operación es atómica y utiliza transacciones para garantizar la integridad
        de los datos. Si ocurre algún error, la transacción se revierte automáticamente.
        
        Args:
            ruta_archivo (str): Ruta del archivo original
            titulo (str): Título del documento
            contenido (str): Contenido del chunk
            embedding (np.ndarray): Vector de embedding
            inicio (int): Posición inicial en el documento original
            fin (int): Posición final en el documento original
            test_mode (bool): Si es True, muestra información detallada
            
        Returns:
            int: ID del chunk insertado
            
        Raises:
            Exception: Si ocurre un error durante la inserción
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            # Iniciar transacción
            cursor.execute("BEGIN TRANSACTION")
            
            # Insertar metadatos
            cursor.execute('''
            INSERT INTO chunks_metadata 
            (ruta_archivo, titulo, contenido, inicio, fin)
            VALUES (?, ?, ?, ?, ?)
            ''', (ruta_archivo, titulo, contenido, inicio, fin))
            
            # Obtener el ID del último registro insertado
            cursor.execute("SELECT last_insert_rowid()")
            chunk_id = cursor.fetchone()[0]
            
            # Serializar el embedding
            embedding_bytes = serialize(embedding.tolist())
            
            # Insertar embedding
            cursor.execute('''
            INSERT INTO chunks_embeddings (id, embedding)
            VALUES (?, ?)
            ''', (chunk_id, embedding_bytes))
            
            # Confirmar transacción
            cursor.execute("COMMIT")
            
            if test_mode:
                logging.info(f"Chunk insertado con ID: {chunk_id}")
            
            return chunk_id
            
        except Exception as e:
            # En caso de error, revertir cambios
            cursor.execute("ROLLBACK")
            logging.error(f"Error al insertar chunk: {e}")
            raise
        finally:
            conn.close()
    
    def get_chunk(self, chunk_id: int, test_mode: bool = False) -> Optional[Dict[str, Any]]:
        """
        Obtiene un chunk y su embedding por ID.
        
        Args:
            chunk_id (int): ID del chunk a recuperar
            test_mode (bool): Si es True, muestra información detallada
            
        Returns:
            Optional[Dict[str, Any]]: Diccionario con la información del chunk o None si no existe
        """
        if test_mode:
            logging.info(f"Obteniendo chunk con ID: {chunk_id}")
            start_time = datetime.now()
        
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Obtener metadatos
        cursor.execute('''
        SELECT m.*, e.embedding 
        FROM chunks_metadata m
        JOIN chunks_embeddings e ON m.id = e.id
        WHERE m.id = ?
        ''', (chunk_id,))
        
        chunk = cursor.fetchone()
        conn.close()
        
        if not chunk:
            if test_mode:
                logging.warning(f"No se encontró el chunk con ID: {chunk_id}")
            return None
        
        if test_mode:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            logging.info(f"Chunk recuperado en {duration:.2f} segundos")
        
        return {
            'id': chunk[0],
            'ruta_archivo': chunk[1],
            'titulo': chunk[2],
            'contenido': chunk[3],
            'inicio': chunk[4],
            'fin': chunk[5],
            'fecha_creacion': chunk[6],
            'embedding': np.array(chunk[7])
        }
    
    def buscar_chunks_similares(self, embedding: np.ndarray, 
                              top_k: int = 5, test_mode: bool = False) -> List[Dict[str, Any]]:
        """
        Busca los chunks más similares a un embedding dado.
        
        Utiliza la función de similitud de sqlite-vec para encontrar los chunks
        más cercanos al vector de embedding proporcionado.
        
        Args:
            embedding (np.ndarray): Vector de embedding para la búsqueda
            top_k (int): Número de resultados a retornar
            test_mode (bool): Si es True, muestra información detallada
            
        Returns:
            List[Dict[str, Any]]: Lista de chunks ordenados por similitud
        """
        if test_mode:
            logging.info(f"Buscando {top_k} chunks similares")
            start_time = datetime.now()
        
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Usar la función de similitud de sqlite-vec
        cursor.execute('''
        SELECT
            m.*,
            e.embedding,
            distance
        FROM chunks_embeddings e
        LEFT JOIN chunks_metadata m ON m.id = e.id
        WHERE e.embedding MATCH ?
            AND k = ?
        ORDER BY distance
        ''', (serialize(embedding.tolist()), top_k))
        
        chunks = cursor.fetchall()
        conn.close()
        
        if test_mode:
            logging.info(f"Recuperados {len(chunks)} chunks similares")
        
        resultados = [{
            'id': chunk[0],
            'ruta_archivo': chunk[1],
            'titulo': chunk[2],
            'contenido': chunk[3],
            'inicio': chunk[4],
            'fin': chunk[5],
            'fecha_creacion': chunk[6],
            'embedding': np.array(chunk[7]),
            'similitud': chunk[8]
        } for chunk in chunks]
        
        if test_mode:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            logging.info(f"Búsqueda completada en {duration:.2f} segundos")
        
        return resultados
    
    def get_chunks_archivo(self, ruta_archivo: str, test_mode: bool = False) -> List[Dict[str, Any]]:
        """
        Obtiene todos los chunks de un archivo específico.
        
        Args:
            ruta_archivo (str): Ruta del archivo del que se quieren obtener los chunks
            test_mode (bool): Si es True, muestra información detallada
            
        Returns:
            List[Dict[str, Any]]: Lista de chunks ordenados por posición en el archivo
        """
        if test_mode:
            logging.info(f"Obteniendo chunks del archivo: {ruta_archivo}")
            start_time = datetime.now()
        
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT m.*, e.embedding 
        FROM chunks_metadata m
        JOIN chunks_embeddings e ON m.id = e.id
        WHERE m.ruta_archivo = ?
        ORDER BY m.inicio
        ''', (ruta_archivo,))
        
        chunks = cursor.fetchall()
        conn.close()
        
        if test_mode:
            logging.info(f"Recuperados {len(chunks)} chunks del archivo")
        
        resultados = [{
            'id': chunk[0],
            'ruta_archivo': chunk[1],
            'titulo': chunk[2],
            'contenido': chunk[3],
            'inicio': chunk[4],
            'fin': chunk[5],
            'fecha_creacion': chunk[6],
            'embedding': np.array(chunk[7])
        } for chunk in chunks]
        
        if test_mode:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            logging.info(f"Operación completada en {duration:.2f} segundos")
        
        return resultados 