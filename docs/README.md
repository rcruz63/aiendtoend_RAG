# Sistema RAG (Retrieval Augmented Generation)

## Descripción General
Este sistema implementa un RAG (Retrieval Augmented Generation) que permite realizar consultas semánticas sobre una colección de documentos. El sistema utiliza embeddings vectoriales y una base de datos SQLite con soporte vectorial para realizar búsquedas eficientes.

## Características Principales
- Procesamiento de documentos Markdown
- Generación de embeddings usando OpenAI
- Almacenamiento eficiente en base de datos SQLite
- Búsqueda semántica de información
- Generación de respuestas usando GPT-4
- Validación de integridad de datos

## Requisitos del Sistema
- Python 3.6+
- OpenAI API key
- Extensión sqlite-vec
- Dependencias listadas en requirements.txt

## Estructura del Proyecto
```
.
├── data/               # Directorio para la base de datos
├── docs/              # Documentación del proyecto
├── test_catalogo/     # Documentos de prueba
├── create_db.py       # Script para crear la base de datos
├── database.py        # Clase para gestión de la base de datos
├── query_rag.py       # Script para realizar consultas
├── validate_db.py     # Script para validar la base de datos
└── requirements.txt   # Dependencias del proyecto
```

## Instalación
1. Clonar el repositorio
2. Crear un entorno virtual:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # En Linux/Mac
   # o
   .venv\Scripts\activate  # En Windows
   ```
3. Instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```
4. Configurar variables de entorno:
   ```bash
   cp .env.example .env
   # Editar .env con tu OPENAI_API_KEY
   ```

## Uso
1. Crear la base de datos:
   ```bash
   python create_db.py
   ```

2. Realizar consultas:
   ```bash
   python query_rag.py "Tu pregunta aquí"
   ```

3. Validar la base de datos:
   ```bash
   python validate_db.py
   ```

## Modo de Prueba
Para ejecutar en modo de prueba con logging detallado:
```bash
python query_rag.py "Tu pregunta" -t
```

## Mantenimiento
Para información detallada sobre el mantenimiento y desarrollo, consultar [GUIA_DESARROLLO.md](GUIA_DESARROLLO.md).

## Autor
RCS

## Fecha
2024-03-22 