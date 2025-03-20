import sqlite3
import os
from pathlib import Path
from typing import List, Dict, Generator
import numpy as np
from tqdm import tqdm
from database import Database
import openai
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Configurar OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

DATA_PATH = "catalogos_md"

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

def get_embedding(text: str) -> np.ndarray:
    """
    Obtiene el embedding de un texto usando la API de OpenAI.
    
    Args:
        text: El texto para el que queremos obtener el embedding
        
    Returns:
        Array numpy con el embedding
    """
    try:
        response = openai.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        return np.array(response.data[0].embedding)
    except Exception as e:
        print(f"Error al obtener embedding: {e}")
        raise

def cargar_documentos() -> Generator[Dict[str, str], None, None]:
    """
    Carga los documentos de manera eficiente usando un generador.
    """
    for archivo in Path(DATA_PATH).glob("**/*.md"):
        try:
            with open(archivo, 'r', encoding='utf-8') as f:
                contenido = f.read()
            yield {
                'titulo': archivo.stem,
                'ruta_archivo': str(archivo),
                'contenido': contenido
            }
        except Exception as e:
            print(f"Error al cargar {archivo}: {e}")

def procesar_documento(db: Database, documento: Dict[str, str]) -> None:
    """
    Procesa un documento individual, lo divide en chunks y guarda sus embeddings.
    """
    # Dividir en chunks
    chunks = chunker(documento['contenido'])
    
    # Procesar cada chunk
    for chunk in chunks:
        try:
            # Calcular el embedding
            embedding = get_embedding(chunk)
            
            # Calcular las posiciones en el documento original
            inicio = documento['contenido'].find(chunk)
            fin = inicio + len(chunk)
            
            # Guardar el chunk con su embedding
            db.insert_chunk(
                ruta_archivo=documento['ruta_archivo'],
                titulo=documento['titulo'],
                contenido=chunk,
                embedding=embedding,
                inicio=inicio,
                fin=fin
            )
        except Exception as e:
            print(f"Error al procesar chunk en {documento['titulo']}: {e}")

def generate_rag():
    """
    Genera el RAG a partir de los documentos md.
    """
    # Crear la base de datos si no existe
    create_database()
    
    # Inicializar base de datos
    db = Database()
    
    # Procesar documentos
    documentos = cargar_documentos()
    for documento in tqdm(documentos, desc="Procesando documentos"):
        try:
            procesar_documento(db, documento)
        except Exception as e:
            print(f"Error al procesar {documento['titulo']}: {e}")

def create_database():
    # Crear el directorio data si no existe
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Ruta de la base de datos
    db_path = data_dir / "catalogo.db"
    
    # Conectar a la base de datos (la creará si no existe)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Crear tabla de chunks
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS chunks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ruta_archivo TEXT NOT NULL,
        titulo TEXT NOT NULL,
        contenido TEXT NOT NULL,
        embedding BLOB NOT NULL,  # Almacenamos el embedding como bytes
        inicio INTEGER NOT NULL,  # Posición inicial en el documento original
        fin INTEGER NOT NULL,     # Posición final en el documento original
        fecha_creacion TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Crear índices para mejorar el rendimiento
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_chunks_titulo ON chunks(titulo)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_chunks_ruta ON chunks(ruta_archivo)')
    
    # Guardar los cambios y cerrar la conexión
    conn.commit()
    conn.close()
    
    print(f"Base de datos creada exitosamente en: {db_path}")

if __name__ == "__main__":
    generate_rag()
