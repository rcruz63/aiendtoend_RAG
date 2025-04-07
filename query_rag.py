"""
Script para realizar consultas a la base de datos vectorial SQLite del sistema RAG.

Este script permite realizar consultas semánticas a la base de datos de chunks y embeddings,
devolviendo los fragmentos más relevantes según la similitud con la consulta proporcionada.

Características principales:
- Consulta semántica usando embeddings
- Reutilización de funciones del sistema RAG existente
- Modo de depuración detallado
- Presentación formateada de resultados

Requisitos:
- Base de datos RAG existente
- OpenAI API key configurada
- Dependencias del sistema RAG principal

Uso:
    python consultar_db.py "¿Cuál es el proceso de instalación?" [-t/--test]

Autor: RCS
Fecha: 2025-03-22
"""

import argparse
import logging
from datetime import datetime
from typing import List, Dict, Any
import openai
from dotenv import load_dotenv
import os

from database import Database
from create_db import get_embedding

# Ajustar el nivel de logging para la biblioteca openai
logging.getLogger("openai").setLevel(logging.WARNING)

# Configurar logging básico
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def formatear_resultado(resultado: Dict[str, Any]) -> str:
    """
    Formatea un resultado individual para su presentación en pantalla.
    
    Crea una representación visual clara del chunk encontrado, incluyendo:
    - Ruta del archivo fuente
    - Título del documento
    - Nivel de similitud
    - Contenido del chunk
    
    Args:
        resultado (Dict[str, Any]): Diccionario con los datos del chunk encontrado,
            debe contener las claves: ruta_archivo, titulo, similitud, contenido
        
    Returns:
        str: Texto formateado con separadores visuales para mejor legibilidad
    """
    return f"""
{'='*80}
Archivo: {resultado['ruta_archivo']}
Título: {resultado['titulo']}
Similitud: {resultado['similitud']:.4f}
{'-'*40}
{resultado['contenido']}
"""

def crear_prompt(query: str, resultados: List[Dict[str, Any]]) -> str:
    """
    Crea el prompt para OpenAI combinando la consulta y el contexto relevante.
    
    Construye un prompt estructurado que:
    1. Instruye al modelo a basarse solo en el contexto proporcionado
    2. Incluye la pregunta del usuario
    3. Organiza los chunks de contexto de manera clara
    4. Solicita una respuesta específica
    
    Args:
        query (str): Pregunta o consulta del usuario
        resultados (List[Dict[str, Any]]): Lista de chunks relevantes encontrados,
            cada uno debe contener ruta_archivo y contenido
        
    Returns:
        str: Prompt formateado para enviar a OpenAI
    """
    # Formatear el contexto de los chunks
    contexto = "\n\n".join([
        f"Fragmento de '{r['ruta_archivo']}':\n{r['contenido']}"
        for r in resultados
    ])
    
    # Construir el prompt
    return f"""Por favor, responde a la siguiente pregunta basándote únicamente en el contexto proporcionado.
Si la información en el contexto no es suficiente para responder, indícalo claramente.

Pregunta: {query}

Contexto relevante:
{contexto}

Respuesta:"""

def obtener_respuesta_openai(prompt: str, test_mode: bool = False) -> str:
    """
    Obtiene una respuesta de OpenAI usando el modelo GPT-4.
    
    Realiza una llamada a la API de OpenAI con:
    - Modelo GPT-4
    - Temperatura moderada (0.7) para balance entre creatividad y precisión
    - Límite de tokens para respuestas concisas
    - Instrucciones específicas para el sistema
    
    Args:
        prompt (str): Prompt completo con pregunta y contexto
        test_mode (bool): Si True, muestra información de depuración y tiempos
        
    Returns:
        str: Respuesta generada por OpenAI
        
    Raises:
        Exception: Si hay un error en la llamada a la API de OpenAI
    """
    if test_mode:
        logging.info("Enviando prompt a OpenAI")
        start_time = datetime.now()
    
    try:
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Eres un asistente experto que responde preguntas basándose únicamente en el contexto proporcionado."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        
        respuesta = response.choices[0].message.content
        
        if test_mode:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            logging.info(f"Respuesta de OpenAI recibida en {duration:.2f} segundos")
        
        return respuesta
        
    except Exception as e:
        logging.error(f"Error al obtener respuesta de OpenAI: {e}")
        raise

def formatear_respuesta_final(respuesta: str, resultados: List[Dict[str, Any]]) -> str:
    """
    Formatea la respuesta final incluyendo las fuentes consultadas.
    
    Crea una presentación completa que incluye:
    1. La respuesta generada por OpenAI
    2. Lista de fuentes utilizadas
    3. Separadores visuales para mejor legibilidad
    
    Args:
        respuesta (str): Respuesta generada por OpenAI
        resultados (List[Dict[str, Any]]): Lista de chunks utilizados como contexto,
            cada uno debe contener ruta_archivo
        
    Returns:
        str: Respuesta final formateada con fuentes
    """
    # Obtener fuentes únicas ordenadas
    fuentes = sorted(set(r['ruta_archivo'] for r in resultados))
    
    # Formatear la respuesta con las fuentes
    return f"""
{respuesta}

Fuentes consultadas:
{'-' * 40}
{chr(10).join(f"• {fuente}" for fuente in fuentes)}
"""

def realizar_consulta(query: str, test_mode: bool = False) -> str:
    """
    Realiza una consulta completa al sistema RAG.
    
    Proceso completo:
    1. Genera el embedding de la consulta
    2. Busca chunks similares en la base de datos
    3. Crea un prompt con el contexto
    4. Obtiene respuesta de OpenAI
    5. Formatea la respuesta final
    
    Args:
        query (str): Texto de la consulta a realizar
        test_mode (bool): Si True, muestra información detallada de depuración
        
    Returns:
        str: Respuesta final formateada con fuentes
        
    Raises:
        Exception: Si hay un error en cualquier paso del proceso
    """
    if test_mode:
        logging.info(f"Procesando consulta: {query}")
        start_time = datetime.now()
    
    # Inicializar base de datos
    db = Database()
    
    # 1. Obtener embedding y buscar chunks similares
    query_embedding = get_embedding(query, db, test_mode)
    resultados = db.buscar_chunks_similares(
        embedding=query_embedding,
        top_k=5,
        test_mode=test_mode
    )
    
    if not resultados:
        return "No se encontró información relevante para responder a tu pregunta."
    
    # 2. Crear prompt con el contexto
    prompt = crear_prompt(query, resultados)
    
    if test_mode:
        logging.info("Prompt generado:")
        logging.info(prompt)
    
    # 3. Obtener respuesta de OpenAI
    respuesta = obtener_respuesta_openai(prompt, test_mode)
    
    # 4. Formatear respuesta final con fuentes
    respuesta_final = formatear_respuesta_final(respuesta, resultados)
    
    if test_mode:
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        logging.info(f"Proceso completo finalizado en {duration:.2f} segundos")
    
    return respuesta_final

def main():
    """
    Función principal que coordina la ejecución del script.
    
    Proceso:
    1. Procesa argumentos de línea de comandos
    2. Configura el nivel de logging
    3. Realiza la consulta
    4. Maneja errores y excepciones
    
    Argumentos:
        query: Texto de la consulta a realizar
        -t/--test: Activa modo de depuración con logging detallado
    """
    parser = argparse.ArgumentParser(
        description='Realiza consultas al sistema RAG'
    )
    parser.add_argument(
        'query',
        help='Texto de la consulta a realizar'
    )
    parser.add_argument(
        '-t', '--test',
        action='store_true',
        help='Modo test con logging detallado'
    )
    args = parser.parse_args()
    
    # Configurar nivel de logging según modo
    logging.getLogger().setLevel(logging.INFO if args.test else logging.WARNING)
    
    try:
        # Realizar consulta completa y mostrar resultado
        respuesta = realizar_consulta(args.query, args.test)
        print(respuesta)
            
    except Exception as e:
        logging.error(f"Error al procesar la consulta: {e}")
        raise

if __name__ == "__main__":
    # Cargar variables de entorno
    load_dotenv()
    
    # Verificar API key de OpenAI
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("No se encontró OPENAI_API_KEY en las variables de entorno")
    
    main()