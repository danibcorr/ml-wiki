# Modulo 1

Necesitamos registrar los modelos o cambios que realizamos en un experimental tracker, para almacenar registros.

ALmacenar un modelo en un model registry, que suele almacenarse en el mismo lugar que el experiment tracker.

HAremos esto con MLflow.

Modular el código principal en diferentes partes, principalmente para crear un pipeline y que pueda ser utilizado para procesos diferentes, al final es reutilizar procesos para reducir tiempos de puesta en producción para tareas similares.

También debemos monitorizar el funcionamiento del modelo cuando se realiza el deploymnet, cuando se utiliza para inferencia, para recopilar datos, ver el funcionamiento, etc.

Existen diferentes niveles

+ NO MLOPS (nivel 0): no automaticaciones y todo el codigo en JUpyter.
+ DevOps (nivel 1): existe automaticación, buenas prácticas, existen métricas, etc pero no se centran en el ML. Entra la parte de ingeniería.
+ Automated Training (nivel 2): existe un pipeline de entrenamiento, hay un tracking de los experimentos, y existe menorr fricción cuando se realiza un deployment.
+ Automated deployment (nivel 3): facil de desplegar un modelo, existen tests como canary o A/B, existe monitorización de los modelos.
+ FUll MLOps automation: todo el pipeline al completo automatizado.

# Modulo 2

## Versionado de los datos con DVC

DVC necesita de Git para el versionado, por lo que debemos crear un repositorio primero e inicializar Git con git init. Luego utilizamos 

dvc init 

cuando hallamos instalado el paquete pip de dvc.

Creamos un .gitignore para indicar los archivos, los nombres de los datasets, que no vamos a trackear con Git ya que eso lo hará dvc.

Ahora para añadir los datos al tracker de DVC utilizamos:

dvc add directorio_dataset

Pero el dataset solo estará disponible de manera local, por lo que ninguna persona de nuetro grupo u oganizacion podra acceder a el. Para ello DVC cuenta con el comando remote. Remote permite almacenar el dataset en ubicaciones de almacenamiento distribuidas para tus conjuntos de datos y modelos ML (similares a las remotas Git, pero para activos en caché). Esta característica opcional se utiliza normalmente para compartir o realizar copias de seguridad de todos o algunos de sus datos. Se admiten varios tipos: Amazon S3, Google Drive, SSH, HTTP, sistemas de archivos locales, entre otros.

Por ejemplo podemos utilizar el siguiente comando para agregar una carpeta de Google Drive como lugar de almacenamiento:

dvc remote add -d storage gdrive://ID_CARPETA

Del comando anterior, el ID_CARPETA es el identificador que aparece al final del enlace URL de la carpeta. En este caso incluso pedirá, si no está instalado, instalar un paquete pip, utilizamos el comando

pip install dvc_gdrive  

Una vez instalado podemos hacer un push de dvc

dvc push

El motivo de hacerlo así es que los datos pueden ser voluminosos y no es viable almacenarlos en GItHub, es mas, podría superar el limite de almacenamiento. Pero lo que podemos guardar en GitHub es el fichero .dvc, que contiene el mismo nombre y la misma extension que el que tenía el conjunto de datos con el fin de apuntar a los archivos de dvc.

Con ese archivo una persona podría utilizar el comando

dvc pull

para obtener el ultimo dataset.

Este proceso se debe repetir siempre que se realice alguna modificación del conjunto de datos.

## EXperiment TRacking con MLflow

Vamos a indicar primero que MLflow almacena todos los artefactos generados en una base de datos, para ello, vamos a utilizar como backend sqlite pero MLflow permite otras bases de datos.

mlflow ui --backend-store-uri sqlite:///mlflow.db

## Machine Learning Pipelines con Mage

Tambien conocido como Workflow orchestration, crear un script de Python no es una opcion por problemas de escalabilidad, compartir código, despliegue, complejidad, etc.

Lo primero es descargar la imagen de Mage con Docker para evitar problemas de dependencias con paquetes pip del entorno a utilizar.

Para ello utilizamos el comando siguiente que realiza un port mapping de la imagen al puerto del host, establece un nombre para el contenedor, monta el directorio actual (por lo que hay que ejecutar el comando en el mismo directorio que el del proyecto) en el directorio /home/src del contenedor, ejecuta la imagen mageai y ejecuta el script para inicializar el proyecto MageChurn (este nombre se puede cambiar)

docker run -it -p 6789:6789 --name mage_churn_mlops -v "$(pwd)":/home/src mageai/mageai /app/run_app.sh mage start MageChurn

Una vez ejecutado, podemos ir al siguiente enlace http://localhost:6789 para ver la interfaz gráfica de Mage.ai

