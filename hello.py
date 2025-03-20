import openai
import os
from dotenv import load_dotenv

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

## def get_embedding(text)
## TODO: workhome Implementar la función get_embedding

## def populate_embeddings(chunks)
## TODO: workhome Implementar la función populate_embeddings
## Opciones PostGres/Supabase
## SQLlite
## Memoria / Python
## Especializadas en esto: Pinecone

## def query_embeddings(query)
## Recibo la pregunta, hago el emmbedding, busco en la base de datos, armo el prompt. 
## Prompt El usuario ha preguntado esto y el contexto es este (sumando los chunks)
## TODO: workhome Implementar la función query_embeddings




def main():
    # Carga las variables de entorno desde el archivo .env
    load_dotenv()
    client = openai.OpenAI()
    
    # Configura la API key desde las variables de entorno
    openai.api_key = os.getenv("OPENAI_API_KEY")
    
    prompt = "Hello, how are you?"
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000,
    )
    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
