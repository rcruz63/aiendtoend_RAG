import openai
import os
from dotenv import load_dotenv

def chunker(text, chunk_size=1000, overlap=200  ):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i : i + chunk_size])
    return chunks

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
