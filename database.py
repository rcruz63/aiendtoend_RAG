from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Any, Tuple
import numpy as np
import logging
import sqlite_vec
import struct
import apsw

def serialize(vector: List[float]) -> bytes:
    """Serializa una lista de floats en formato de bytes compacto"""
    return struct.pack("%sf" % len(vector), *vector)

class Database:
    def __init__(self):
        self.db_path = Path("data") / "catalogo.db"
        
    def get_connection(self):
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
        
        Args:
            ruta_archivo: Ruta del archivo original
            titulo: Título del documento
            contenido: Contenido del chunk
            embedding: Vector de embedding
            inicio: Posición inicial en el documento original
            fin: Posición final en el documento original
            test_mode: Si es True, muestra información detallada
            
        Returns:
            int: ID del chunk insertado
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
        """Obtiene un chunk y su embedding por ID"""
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
        """Busca los chunks más similares a un embedding dado"""
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
        """Obtiene todos los chunks de un archivo"""
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