# RAG de viajes

Este desarrollo pretende crear un agente IA que resuelva dudas sobre los catalogos proporcionados utilizando una IA.

Partirndo de los catalosgos pdf existentes en el directorio `/Users/rcruz2/Documents/Mas50_2025`y utilizando la herramienta markitdown se generan una versión markdown de los catalogos en el directorio `catalogos_md`

Partiendo de la versión markdown se generaran un conjunto de embbedings para cada uno de los archivos md. Cada uno de estos archivos se partiran en fragmentos de N caracteres, con un decalaje de m caracteres, y se generaran los embeddings para cada uno de ellos.

Se almacenaran en una base de datos simple como sqllite o posgres.

La creación de la bases de datos y el almacenamiento de los embeddings se realiza con el script `create_db.py`. Debe comprobar si existe ya la base de datos y si existe debe abrirla, si no existe debe crearla.

Para cada uno de los documentos md se deben partir en framgmentos, generar los embeddings y almacenarlos en la base de datos.

Tendremos otro script para realizar la consulta. Recibe la consulta por parametro, debe generar un ebbeding de la consulta, buscar los fragmentos de los documentos que tengan mas similitud con el embedding de la consulta y devolver la información de los fragmentos.

A continuación debe concatenar el texto de los fragmentos para aportarlo como contexto al LLM y realizar la consulta.






