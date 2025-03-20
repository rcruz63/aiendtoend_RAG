import sqlite3
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Any, Tuple
import numpy as np
import pickle

class Database:
    def __init__(self):
        self.db_path = Path("data") / "catalogo.db"
        
    def get_connection(self):
        return sqlite3.connect(self.db_path)
    
    def insert_chunk(self, ruta_archivo: str, titulo: str, contenido: str, 
                    embedding: np.ndarray, inicio: int, fin: int) -> int:
        """Inserta un chunk con su embedding y retorna su ID"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Convertir el embedding a bytes
        embedding_bytes = pickle.dumps(embedding)
        
        cursor.execute('''
        INSERT INTO chunks (ruta_archivo, titulo, contenido, embedding, inicio, fin)
        VALUES (?, ?, ?, ?, ?, ?)
        ''', (ruta_archivo, titulo, contenido, embedding_bytes, inicio, fin))
        
        chunk_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return chunk_id
    
    def get_chunk(self, chunk_id: int) -> Optional[Dict[str, Any]]:
        """Obtiene un chunk y su embedding por ID"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM chunks WHERE id = ?', (chunk_id,))
        chunk = cursor.fetchone()
        conn.close()
        
        if not chunk:
            return None
            
        # Convertir el embedding de bytes a numpy array
        embedding = pickle.loads(chunk[4])
        
        return {
            'id': chunk[0],
            'ruta_archivo': chunk[1],
            'titulo': chunk[2],
            'contenido': chunk[3],
            'embedding': embedding,
            'inicio': chunk[5],
            'fin': chunk[6],
            'fecha_creacion': chunk[7]
        }
    
    def buscar_chunks_similares(self, embedding: np.ndarray, 
                              top_k: int = 5) -> List[Dict[str, Any]]:
        """Busca los chunks más similares a un embedding dado"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Obtener todos los chunks
        cursor.execute('SELECT * FROM chunks')
        chunks = cursor.fetchall()
        conn.close()
        
        # Calcular similitudes
        similitudes = []
        for chunk in chunks:
            chunk_embedding = pickle.loads(chunk[4])
            similitud = np.dot(embedding, chunk_embedding) / (
                np.linalg.norm(embedding) * np.linalg.norm(chunk_embedding)
            )
            similitudes.append((similitud, chunk))
        
        # Ordenar por similitud y tomar los top_k
        similitudes.sort(reverse=True)
        resultados = []
        
        for _, chunk in similitudes[:top_k]:
            chunk_embedding = pickle.loads(chunk[4])
            resultados.append({
                'id': chunk[0],
                'ruta_archivo': chunk[1],
                'titulo': chunk[2],
                'contenido': chunk[3],
                'embedding': chunk_embedding,
                'inicio': chunk[5],
                'fin': chunk[6],
                'fecha_creacion': chunk[7]
            })
        
        return resultados
    
    def get_chunks_archivo(self, ruta_archivo: str) -> List[Dict[str, Any]]:
        """Obtiene todos los chunks de un archivo"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT * FROM chunks 
        WHERE ruta_archivo = ?
        ORDER BY inicio
        ''', (ruta_archivo,))
        
        chunks = cursor.fetchall()
        conn.close()
        
        return [{
            'id': chunk[0],
            'ruta_archivo': chunk[1],
            'titulo': chunk[2],
            'contenido': chunk[3],
            'embedding': pickle.loads(chunk[4]),
            'inicio': chunk[5],
            'fin': chunk[6],
            'fecha_creacion': chunk[7]
        } for chunk in chunks] 