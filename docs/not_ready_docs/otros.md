- Formatos de datos: para datos binarios como imagenes, audio o similar podemos usar
  formatos como JPEG que estan comprimidos.
- Para metadaos (labels), datos tabulares, texto, utilizar parquets, json, txt esta bien

Podemos usar data warehouse para ETL (datos ta transformados) o Data Lakes para ELT
(datos almacenados para transformar). En la práctica ambos conviven. Yo supongo que es
porque transformar todo un flujo de datos continuo puede ser costro en el tiempo,
recursos y puede no ser requerido o quedar obsoleto, o equipos pueden requerir diferentes
tipos de transformaciones.

Existe herramientas de orquestacion con Prefect, Dogster que son alternativas a Airflow.
