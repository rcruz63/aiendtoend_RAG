# Guía de Desarrollo y Mantenimiento

## Arquitectura del Sistema

### Componentes Principales
1. **Base de Datos (database.py)**
   - Gestión de conexiones SQLite
   - Operaciones CRUD para chunks y embeddings
   - Búsqueda de similitud vectorial

2. **Creación de Base de Datos (create_db.py)**
   - Procesamiento de documentos Markdown
   - Generación de embeddings
   - Inicialización de la base de datos

3. **Consultas (query_rag.py)**
   - Procesamiento de consultas
   - Búsqueda de chunks relevantes
   - Generación de respuestas con GPT-4

4. **Validación (validate_db.py)**
   - Verificación de integridad
   - Comprobación de estructura
   - Análisis de datos

### Flujo de Datos
1. Los documentos Markdown se procesan y dividen en chunks
2. Cada chunk se convierte en un embedding vectorial
3. Los chunks y embeddings se almacenan en la base de datos
4. Las consultas se convierten en embeddings para búsqueda
5. Los chunks más relevantes se usan como contexto para GPT-4

## Base de Datos

### Estructura
```sql
-- Tabla de metadatos
CREATE TABLE chunks_metadata (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ruta_archivo TEXT NOT NULL,
    titulo TEXT NOT NULL,
    contenido TEXT NOT NULL,
    inicio INTEGER NOT NULL,
    fin INTEGER NOT NULL,
    fecha_creacion TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Tabla virtual para embeddings
CREATE VIRTUAL TABLE chunks_embeddings USING vec0(
    id INTEGER PRIMARY KEY,
    embedding FLOAT[1536]
);
```

### Índices
- `idx_chunks_titulo`: Optimiza búsquedas por título
- `idx_chunks_ruta`: Optimiza búsquedas por ruta de archivo

## Mantenimiento

### Tareas Rutinarias
1. **Validación de Base de Datos**
   ```bash
   python validate_db.py
   ```
   - Verifica integridad de datos
   - Comprueba coincidencia chunks-embeddings
   - Reporta problemas encontrados

2. **Actualización de Documentos**
   ```bash
   python create_db.py
   ```
   - Procesa nuevos documentos
   - Actualiza embeddings existentes
   - Mantiene la integridad de la base de datos

### Solución de Problemas Comunes

1. **Error de Conexión a Base de Datos**
   - Verificar permisos del directorio `data/`
   - Comprobar que sqlite-vec está instalado correctamente
   - Revisar logs para errores específicos

2. **Problemas con Embeddings**
   - Verificar API key de OpenAI
   - Comprobar límites de la API
   - Revisar formato de los chunks

3. **Errores de Integridad**
   - Ejecutar `validate_db.py`
   - Revisar discrepancias en conteos
   - Verificar estructura de tablas

## Desarrollo

### Añadir Nuevas Características
1. Crear rama de desarrollo
2. Implementar cambios
3. Actualizar documentación
4. Probar con datos de ejemplo
5. Validar base de datos
6. Crear pull request

### Pruebas
- Usar directorio `test_catalogo/` para pruebas
- Activar modo test con `-t` o `--test`
- Revisar logs detallados
- Validar resultados

### Mejores Prácticas
1. **Código**
   - Seguir PEP 8
   - Documentar funciones y clases
   - Manejar excepciones apropiadamente
   - Usar tipos estáticos

2. **Base de Datos**
   - Usar transacciones
   - Mantener índices actualizados
   - Validar integridad regularmente

3. **API**
   - Manejar límites de rate
   - Implementar retry logic
   - Logging apropiado

## Optimización

### Rendimiento
1. **Base de Datos**
   - Usar índices apropiadamente
   - Optimizar consultas
   - Mantener estadísticas actualizadas

2. **Embeddings**
   - Ajustar tamaño de chunks
   - Optimizar solapamiento
   - Cachear resultados frecuentes

3. **Consultas**
   - Limitar número de chunks
   - Optimizar prompts
   - Ajustar parámetros de búsqueda

### Escalabilidad
- Considerar sharding para grandes volúmenes
- Implementar caché distribuido
- Optimizar uso de memoria

## Seguridad

### Consideraciones
1. **API Keys**
   - Usar variables de entorno
   - No commitear keys
   - Rotar keys regularmente

2. **Base de Datos**
   - Restringir permisos
   - Validar inputs
   - Sanitizar queries

3. **Logs**
   - No exponer información sensible
   - Implementar rotación
   - Configurar niveles apropiados

## Soporte

### Recursos
- Documentación de OpenAI
- Documentación de sqlite-vec
- PEP 8 Style Guide
- SQLite Documentation

### Contacto
Para soporte técnico o preguntas:
- [Información de contacto]

## Versiones

### Historial
- v1.0.0 (2024-03-22): Versión inicial
  - Implementación básica
  - Soporte para Markdown
  - Búsqueda semántica
  - Validación de datos 