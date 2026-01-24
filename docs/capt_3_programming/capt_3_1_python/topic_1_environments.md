---
authors: Daniel Bazo Correa
description:
  Creación y gestión de entornos virtuales de Python con VENV, Anaconda y Poetry
title: Gestión de entornos en Python
---

## Bibliografía

- [VENV Docs](https://docs.python.org/3/library/venv.html)
- [Poetry Docs](https://python-poetry.org/)
- [Anaconda Docs](https://docs.anaconda.com/)
- [uv Docs](https://docs.astral.sh/uv/)

## 1. Introducción

Existen varias opciones para la gestión de paquetes en Python. Independientemente de la
que elijas, mi recomendación es optar siempre por la alternativa más simple y
minimalista, aunque también dependerá mucho del entorno de desarrollo que tengas en la
empresa o de a lo que estés acostumbrado. Recuerda que, al final, estas siguen siendo
herramientas que tienes que considerar y, si te aportan mejoras, implementarlas
gradualmente. Cuanto menor sea el número de dependencias y más ligero sea el entorno,
más fácil será llevarlo a producción (por ejemplo, en una imagen de Docker), compartirlo
con tu equipo o mantenerlo en el tiempo.

En la actualidad, las opciones que más recomiendo son **Poetry** y **uv**. Ambas
herramientas agilizan la creación y gestión de entornos, permiten mantener las
configuraciones del proyecto de forma organizada mediante un archivo `pyproject.toml` y,
sobre todo, favorecen la **reproducibilidad** de los proyectos.

### 1.1. Anaconda

<p align="center">
  <img src="/assets/img/docs/logos/anaconda-logo.png" width="500"/>
  <br />
  <em>Logo de Anaconda</em>
</p>

**Anaconda** es una plataforma de código abierto diseñada para la creación y gestión de
entornos virtuales en Python, enfocada en proyectos de ciencia de datos y aprendizaje
automático. Proporciona una distribución de Python con numerosas bibliotecas
preinstaladas, un gestor de paquetes propio y herramientas como
[_Jupyter_](https://jupyter.org/).

La gestión de paquetes en Anaconda se realiza mediante
_[Conda](https://anaconda.org/anaconda/repo)_, aunque también es posible utilizar
_[PIP](https://pypi.org/)_. Sin embargo, no es recomendable mezclar ambos, ya que pueden
surgir errores en la compatibilidad de paquetes.

Durante años fue la plataforma más usada en ciencia de datos, ya que ofrecía un
ecosistema completo (Jupyter, Spyder, RStudio, etc.) de manera muy sencilla. Sin
embargo, con el tiempo ha presentado limitaciones como una **licencia más restrictiva**
para empresas y un **exceso de dependencias por defecto**.

Hoy en día existen alternativas más eficientes, modernas y ligeras, como **Poetry** y
**uv**, que utilizan el gestor de entornos virtuales de Python y evitan instalar
paquetes innecesarios.

### 1.2. VENV

Por otro lado, [`VENV`](https://docs.python.org/3/library/venv.html) es una alternativa
más ligera para la creación de entornos virtuales sin las dependencias adicionales de
Anaconda, y que ya viene por defecto cuando instalamos Python. En este caso, la gestión
de paquetes se lleva a cabo con _[PIP](https://pypi.org/)_.

### 1.3. Poetry

<p align="center">
  <img src="https://python-poetry.org/images/logo-origami.svg" width="100"/>
  <br />
  <em>Logo de Poetry</em>
</p>

[`Poetry`](https://python-poetry.org/) es otra herramienta de gestión de dependencias en
proyectos de Python.

Permite, entre otras cosas, administrar dependencias por grupos (_producción_,
_pruebas_, _documentación_, etc.), eliminando la necesidad de crear múltiples ficheros
de requisitos de dependencias (los `requirements.txt`) o de tener un único fichero.

También permite crear y manejar entornos virtuales automáticamente, y facilitar la
creación de _wheels_ para empaquetar proyectos y publicarlos en
_[PyPI](https://pypi.org/)_ o en tu repositorio de paquetes privado.

### 1.4. uv

[`uv`](https://docs.astral.sh/uv/) es una de las herramientas más recientes y eficientes
para la gestión de entornos virtuales y dependencias en Python, y es la que
personalmente utilizo y recomiendo. Su objetivo principal es simplificar y acelerar
tareas que tradicionalmente requieren múltiples herramientas, como `pip`, `poetry` o
`venv`.

Una de sus principales ventajas es la posibilidad de crear un entorno virtual por
proyecto. Esta es la mejor práctica, ya que así evitamos mezclar dependencias entre
proyectos, lo que puede llevar a conflictos entre versiones.

Lo que más me ha impresionado de `uv` es su velocidad, en parte gracias a que utiliza
Rust para instalar y resolver dependencias en milisegundos, superando a `pip` y
`poetry`. Además, permite crear un sistema similar a `cargo` de Rust, basado en archivos
`pyproject.toml`, donde podemos definir metadatos de nuestro proyecto, gestionar
paquetes con sus versiones, especificar la versión de Python requerida, así como
configuraciones específicas de proyectos que instalemos, como linters o similares.

Por otro lado, `uv` permite la **gestión automática de entornos**, no requiere
configuraciones adicionales para crear y mantener entornos virtuales, por lo que no
necesitas tener instalado Python en el sistema que estés utilizando, `uv` lo hace de
forma automática.

## 2. Utilidades para la gestión de entornos

### 2.1. Creación de un entorno virtual

Un **entorno virtual** es como una “caja aislada” donde instalamos las librerías que
necesita un proyecto en particular, sin afectar al resto del sistema ni a otros
proyectos. Dependiendo de la herramienta que elijas, el proceso puede variar un poco.
Aquí tienes las opciones más utilizadas:

=== "VENV"

      1.  **Actualizar el sistema** (para tener las últimas mejoras y seguridad):

         ```bash
         sudo apt update && sudo apt upgrade -y
         ```

      2.  **Añadir un repositorio con versiones recientes de Python** (opcional, solo si tu
         sistema no tiene la versión que necesitas):

         ```bash
         sudo add-apt-repository ppa:deadsnakes/ppa
         sudo apt update
         ```

      3.  **Instalar una versión específica de Python** (ejemplo: Python 3.10):

         ```bash
         sudo apt install python3.10
         ```

      4.  **Instalar VENV y herramientas básicas** (`pip` y cabeceras de desarrollo):

         ```bash
         sudo apt install python3.10-venv python3.10-dev python3-pip
         ```

      5.  **Crear el entorno virtual** dentro del directorio del proyecto:

         ```bash
         python -m venv nombre_del_entorno
         ```

      6.  **Activar el entorno** (a partir de aquí, todo lo que instales quedará dentro de
         esta “caja aislada”):

         ```bash
         source nombre_del_entorno/bin/activate
         ```

=== "Anaconda"

      7. **Descargar e instalar Anaconda** desde la
         [página oficial](https://www.anaconda.com/download).

      8. **Abrir la terminal de Anaconda Prompt** (en Windows se instala junto con Anaconda).

      9. **Crear un nuevo entorno**:

         ```bash
         conda create --name nombre_del_entorno
         ```

      10. **Activar el entorno**:

         ```bash
         conda activate nombre_del_entorno
         ```

      11. **(Opcional) Usar `pip` dentro de Anaconda**: Se puede, aunque no se recomienda
         mezclar `conda` y `pip`, porque puede dar problemas de compatibilidad.

         ```bash
         conda install anaconda::pip
         pip install --upgrade pip
         ```

=== "Poetry"

      12. **Instalar Poetry**:

         ```bash
         pip install poetry
         ```

      13. **Configurar Poetry para que cree entornos dentro del proyecto** (esto es lo más
         práctico y viene por defecto en las versiones recientes):

         ```bash
         poetry config virtualenvs.in-project true
         ```

      14. **Crear un nuevo proyecto** (Poetry genera la estructura básica con carpetas y un
         archivo `pyproject.toml`):

         ```bash
         poetry new nombre_del_proyecto
         ```

      15. **Instalar dependencias y generar el entorno automáticamente**:

         ```bash
         poetry install
         ```

=== "uv"

      16. **Instalar uv** (en Linux/macOS):

         ```bash
         curl -LsSf https://astral.sh/uv/install.sh | sh
         ```

      17. **Crear un nuevo proyecto**:

         ```bash
         uv init nombre_del_proyecto
         cd nombre_del_proyecto
         ```

      18. **Crear el entorno virtual e instalar dependencias**:

         ```bash
         uv venv
         uv pip install nombre_del_paquete
         ```

### 2.2. Gestión de la caché

Para liberar espacio o solucionar problemas con dependencias, se puede purgar la caché
con los siguientes comandos:

=== "PIP"

      ```bash
      pip cache purge
      ```

=== "Anaconda"

      ```bash
      conda clean --all
      ```

=== "Poetry"

      ```bash
      poetry cache clear --all
      ```

=== "uv"

      ```bash
      uv cache clean
      ```

### 2.3. Actualización de paquetes

Mantener las dependencias actualizadas es clave para el correcto funcionamiento del
proyecto.

=== "PIP"

      1. Actualizar todos los paquetes

         Puedes utilizar el siguiente comando para actualizar todos los paquetes:

         ```bash
         pip freeze | grep -v "^\-e" | cut -d = -f 1 | xargs -n1 pip install -U
         ```

         Donde:

         - `pip freeze`: Genera una lista de los paquetes instalados.
         - `grep -v "^\-e"`: Excluye las instalaciones en modo editable.
         - `cut -d = -f 1`: Extrae solo los nombres de los paquetes, sin las versiones.
         - `xargs -n1 pip install -U`: Actualiza cada paquete.

      2. Actualizar un paquete específico

         Para actualizar un paquete específico:

         ```bash
         pip install --upgrade nombre_del_paquete
         ```

=== "Anaconda"

      1. Actualizar todos los paquetes

         Aunque Anaconda permite la instalación de paquetes con PIP, se recomienda evitar
         mezclar paquetes del repositorio de Anaconda y PIP, ya que esto podría causar
         conflictos. Si decides usar paquetes de Anaconda, puedes actualizar todos los
         paquetes con:

         ```bash
         conda update --all
         ```

      2. Actualizar un paquete específico

         Para actualizar un paquete específico:

         ```bash
         conda update nombre_del_paquete
         ```

=== "Poetry"

      1. Actualizar todos los paquetes

         ```bash
         poetry update
         ```

      2. Actualizar un paquete específico

         Para actualizar un paquete específico:

         ```bash
         poetry update nombre_del_paquete
         ```

=== "uv"

      ```bash
      uv pip install --upgrade nombre_del_paquete
      ```

### 2.4. Instalación de paquetes desde un archivo de requisitos

Cuando un proyecto necesita dependencias específicas, es útil usar un archivo
`requirements.txt`:

1. **Crear un archivo `requirements.txt`** con los paquetes y versiones deseadas:

   ```plaintext
   numpy==1.21.0
   pandas>=1.3.0
   requests
   ```

2. **Instalar los paquetes desde el archivo**:

=== "PIP"

      ```bash
      pip install -r requirements.txt
      ```

=== "Poetry"

      ```bash
      poetry install
      ```

=== "uv"

      ```bash
      uv pip install -r requirements.txt
      ```

### 2.5. Eliminar un entorno

=== "VENV, Poetry, uv"

      En la mayoría de los casos, los entornos creados con ``VENV``, ``Poetry`` y ``uv`` se alojan dentro del propio directorio del proyecto. Por ello, si ya no los necesitas, basta con eliminar la carpeta correspondiente para borrar por completo el entorno junto con toda su información.

      ```bash
      rm -rf nombre_del_entorno
      ```

=== "Anaconda"

      1. **Listar los entornos disponibles**:

         ```bash
         conda env list
         ```

      2. **Eliminar un entorno**:

         ```bash
         conda env remove --name nombre_del_entorno
         ```

### 2.6. Integración del entorno con Jupyter

Para utilizar un entorno virtual dentro de **Jupyter**, es necesario seguir estos pasos:

1. **Instalar `ipykernel` en el entorno**: Primero, debes añadir `ipykernel` como
   dependencia dentro del entorno virtual. Para ello, instala el paquete utilizando el
   gestor de dependencias correspondiente (por ejemplo, `pip`, `poetry`, `uv` o
   `conda`).
2. **Registrar el entorno en Jupyter**: Este paso es necesario únicamente cuando el
   entorno virtual se encuentra en un directorio diferente al del proyecto. En la
   mayoría de los entornos de desarrollo, como **VSCode**, si el entorno está dentro del
   directorio del proyecto, se detectará automáticamente y podrás seleccionar el kernel
   asociado sin pasos adicionales. En caso de que necesites registrar el entorno
   manualmente, ejecuta:

   ```bash
   python -m ipykernel install --user --name=nombre_del_entorno
   ```

En el caso de utilizar `uv`, si has empleado el comando `uv venv`, por defecto `uv`
creará un entorno en la raíz del proyecto en la que te encuentras, con la versión de
Python especificada en el `pyproject.toml`. Con ello, al utilizar Jupyter Notebooks,
VSCode detectará directamente que el entorno se encuentra en la raíz del proyecto sin
necesidad de ejecutar los comandos anteriores.

### 2.7. Eliminación de paquetes instalados

=== "PIP"

      Eliminar todos los paquetes

      ```bash
      pip list --format=freeze > installed.txt
      pip uninstall -r installed.txt -y
      ```

      Eliminar un paquete específico

      ```bash
      pip uninstall nombre_del_paquete
      ```

=== "Anaconda"

      ```bash
      conda remove nombre_del_paquete
      ```

=== "Poetry"

      ```bash
      poetry remove nombre_del_paquete
      ```

=== "uv"

      ```bash
      uv pip uninstall nombre_del_paquete
      ```
