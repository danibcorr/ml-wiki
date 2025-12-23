---
sidebar_position: 1
authors:
  - name: Daniel Bazo Correa
description:
  Creación y gestión de entornos virtuales de Python con VENV, Anaconda y Poetry
title: Gestión de entornos en Python
toc_max_heading_level: 3
---

import Tabs from '@theme/Tabs'; import TabItem from '@theme/TabItem';

## Bibliografía

- [VENV Docs](https://docs.python.org/3/library/venv.html)
- [Poetry Docs](https://python-poetry.org/)
- [Anaconda Docs](https://docs.anaconda.com/)
- [uv Docs](https://docs.astral.sh/uv/)

## 1. Introducción

Existen varias opciones para la **gestión de paquetes en Python**. Independientemente de
la que elijas, mi recomendación es optar siempre por la alternativa más **simple y
minimalista**. Cuanto menor sea el número de dependencias y más ligero el entorno, más
fácil será llevarlo a producción (por ejemplo, en una imagen de Docker), compartirlo con
tu equipo o mantenerlo en el tiempo.

En la actualidad, las opciones que más recomiendo son **Poetry** y **uv**. Ambas
herramientas agilizan la creación y gestión de entornos, permiten mantener las
configuraciones del proyecto de forma organizada mediante un archivo `pyproject.toml` y,
sobre todo, favorecen la **reproducibilidad** de los proyectos, algo esencial tanto en
desarrollo como en despliegue.

### 1.1. Anaconda

<p align="center">
  <img src={require("../../../static/img/docs/logos/anaconda-logo.png").default} width="500"/>
  <br />
  <em>Logo de Anaconda</em>
</p>

**Anaconda** es una plataforma de código abierto diseñada para la creación y gestión de
entornos virtuales en Python, enfocada en proyectos de ciencia de datos y aprendizaje
automático. Proporciona una distribución de Python con numerosas bibliotecas
preinstaladas, un gestor de paquetes eficiente y herramientas avanzadas como
[_Jupyter_](https://jupyter.org/).

La gestión de paquetes en Anaconda se realiza mediante
[_Conda_](https://anaconda.org/anaconda/repo), aunque también es posible utilizar
[_PIP_](https://pypi.org/). Durante años fue la plataforma más usada en ciencia de
datos, ya que ofrecía un ecosistema completo (Jupyter, Spyder, RStudio, etc.) de manera
muy sencilla. Sin embargo, con el tiempo ha presentado limitaciones:

- **Licencia más restrictiva** para empresas.
- **Exceso de dependencias por defecto**, lo que dificulta entornos de producción
  ligeros.

Hoy en día existen alternativas más eficientes, modernas y ligeras, como **Poetry** y
**uv**, que utilizan el gestor de entornos virtuales de Python y evitan instalar
paquetes innecesarios.

### 1.2. VENV

Por otro lado, [_VENV_](https://docs.python.org/3/library/venv.html) es una alternativa
más ligera para la creación de entornos virtuales sin las dependencias adicionales de
Anaconda. En este caso, la gestión de paquetes se lleva a cabo con
[_PIP_](https://pypi.org/).

### 1.3. Poetry

<p align="center">
  <img src="https://python-poetry.org/images/logo-origami.svg" width="100"/>
  <br />
  <em>Logo de Poetry</em>
</p>

[_Poetry_](https://python-poetry.org/) es otra herramienta de gestión de dependencias en
proyectos de Python. Permite, entre otras cosas:

- Administrar dependencias por grupos (_producción_, _pruebas_, _documentación_, etc.).
- Crear y manejar entornos virtuales automáticamente.
- Facilitar la creación de _wheels_ para empaquetar proyectos y publicarlos en
  [_PyPI_](https://pypi.org/).

### 1.4. uv

`uv` es una de las herramientas más recientes y eficientes para la gestión de entornos
virtuales y dependencias en Python. Su objetivo principal es simplificar y acelerar
tareas que tradicionalmente requieren múltiples herramientas, como `pip`, `poetry` o
`venv`. Desarrollada por **Astral**, los creadores de Ruff, `uv` se destaca por su
rapidez y su integración directa con el flujo de trabajo moderno de proyectos Python.

La instalación de `uv` se realiza a través de la terminal y puede depender del sistema
operativo utilizado. Una de sus principales ventajas es la posibilidad de crear un
entorno virtual por proyecto, lo que permite instalar únicamente las dependencias
necesarias para cada proyecto, evitando conflictos de versiones.

#### 1.4.1. Operaciones básicas de `uv`

- `uv python list` → Muestra las versiones de Python disponibles.
- `uv python install <version>` → Instala una versión específica de Python.
- `uv run` → Ejecuta scripts dentro del entorno virtual.
- `uv init` → Inicializa un proyecto gestionado por `uv`.
- `uv add` → Añade nuevas dependencias al proyecto.
- `uv sync` → Sincroniza el entorno con las dependencias declaradas.
- `uv tree` → Muestra la estructura de dependencias.
- `uv venv` → Gestiona entornos virtuales automáticamente.

#### 1.4.2. Ventajas de `uv`

- **Velocidad extrema:** Utiliza Rust para instalar y resolver dependencias en
  milisegundos, superando a `pip` y `poetry`.
- **Modelo familiar:** Emplea un sistema similar a `cargo` de Rust, basado en archivos
  `pyproject.toml`.
- **Gestión automática de entornos:** No requiere configuraciones adicionales para crear
  y mantener entornos virtuales.
- **Solución unificada:** Reemplaza de forma eficiente a `pip`, `pip-tools`, `poetry` y
  `venv` con una sola herramienta moderna.

Para más información y documentación, se puede consultar el repositorio oficial de
[_uv_](https://docs.astral.sh/uv/).

## 2. Utilidades para la gestión de entornos

### 2.1. Creación de un entorno virtual

Un **entorno virtual** es como una “caja aislada” donde instalamos las librerías que
necesita un proyecto en particular, sin afectar al resto del sistema ni a otros
proyectos. Dependiendo de la herramienta que elijas, el proceso puede variar un poco.
Aquí tienes las opciones más utilizadas:

<Tabs>
   <TabItem value="venv" label="VENV">

      1. **Actualizar el sistema** (para tener las últimas mejoras y seguridad):

         ```bash
         sudo apt update && sudo apt upgrade -y
         ```

      2. **Añadir un repositorio con versiones recientes de Python** (opcional, solo si tu
         sistema no tiene la versión que necesitas):

         ```bash
         sudo add-apt-repository ppa:deadsnakes/ppa
         sudo apt update
         ```

      3. **Instalar una versión específica de Python** (ejemplo: Python 3.10):

         ```bash
         sudo apt install python3.10
         ```

      4. **Instalar VENV y herramientas básicas** (`pip` y cabeceras de desarrollo):

         ```bash
         sudo apt install python3.10-venv python3.10-dev python3-pip
         ```

      5. **Crear el entorno virtual** dentro del directorio del proyecto:

         ```bash
         python -m venv nombre_del_entorno
         ```

      6. **Activar el entorno** (a partir de aquí, todo lo que instales quedará dentro de esta
         “caja aislada”):

         ```bash
         source nombre_del_entorno/bin/activate
         ```

   </TabItem>

   <TabItem value="anaconda" label="Anaconda">

      1. **Descargar e instalar Anaconda** desde la
         [página oficial](https://www.anaconda.com/download).

      2. **Abrir la terminal de Anaconda Prompt** (en Windows se instala junto con Anaconda).

      3. **Crear un nuevo entorno**:

         ```bash
         conda create --name nombre_del_entorno
         ```

      4. **Activar el entorno**:

         ```bash
         conda activate nombre_del_entorno
         ```

      5. **(Opcional) Usar `pip` dentro de Anaconda**: Se puede, aunque no se recomienda
         mezclar `conda` y `pip`, porque puede dar problemas de compatibilidad.

         ```bash
         conda install anaconda::pip
         pip install --upgrade pip
         ```

   </TabItem>

   <TabItem value="poetry" label="Poetry">

      1. **Instalar Poetry**:

         ```bash
         pip install poetry
         ```

      2. **Configurar Poetry para que cree entornos dentro del proyecto** (esto es lo más
         práctico y viene por defecto en las versiones recientes):

         ```bash
         poetry config virtualenvs.in-project true
         ```

      3. **Crear un nuevo proyecto** (Poetry genera la estructura básica con carpetas y un
         archivo `pyproject.toml`):

         ```bash
         poetry new nombre_del_proyecto
         ```

      4. **Instalar dependencias y generar el entorno automáticamente**:

         ```bash
         poetry install
         ```

   </TabItem>

   <TabItem value="uv" label="uv">

      1. **Instalar uv** (en Linux/macOS):

         ```bash
         curl -LsSf https://astral.sh/uv/install.sh | sh
         ```

      2. **Crear un nuevo proyecto**:

         ```bash
         uv init nombre_del_proyecto
         cd nombre_del_proyecto
         ```

      3. **Crear el entorno virtual e instalar dependencias**:

         ```bash
         uv venv
         uv pip install nombre_del_paquete
         ```

   </TabItem>
</Tabs>

### 2.2. Gestión de la caché

Para liberar espacio o solucionar problemas con dependencias, se puede purgar la caché
con los siguientes comandos:

<Tabs>
   <TabItem value="pip" label="PIP">
      ```bash
      pip cache purge
      ```
   </TabItem>
   <TabItem value="anaconda" label="Anaconda">
      ```bash
      conda clean --all
      ```
   </TabItem>
   <TabItem value="poetry" label="Poetry">
      ```bash
      poetry cache clear --all .
      ```
   </TabItem>
   <TabItem value="uv" label="uv">
      ```bash
      uv cache clean
      ```
   </TabItem>
</Tabs>

### 2.3. Actualización de paquetes

Mantener las dependencias actualizadas es clave para el correcto funcionamiento del
proyecto.

<Tabs>
   <TabItem value="pip" label="PIP">

      ##### Actualizar todos los paquetes

      Puedes utilizar el siguiente comando para actualizar todos los paquetes:

      ```bash
      pip freeze | grep -v "^\-e" | cut -d = -f 1 | xargs -n1 pip install -U
      ```

      Donde:

      - `pip freeze`: Genera una lista de los paquetes instalados.
      - `grep -v "^\-e"`: Excluye las instalaciones en modo editable.
      - `cut -d = -f 1`: Extrae solo los nombres de los paquetes, sin las versiones.
      - `xargs -n1 pip install -U`: Actualiza cada paquete.

      ##### Actualizar un paquete específico

      Para actualizar un paquete específico:

      ```bash
      pip install --upgrade nombre_del_paquete
      ```

   </TabItem>
   <TabItem value="anaconda" label="Anaconda">

      ##### Actualizar todos los paquetes

      Aunque Anaconda permite la instalación de paquetes con PIP, se recomienda evitar
      mezclar paquetes del repositorio de Anaconda y PIP, ya que esto podría causar
      conflictos. Si decides usar paquetes de Anaconda, puedes actualizar todos los
      paquetes con:

      ```bash
      conda update --all
      ```

      ##### Actualizar un paquete específico

      Para actualizar un paquete específico:

      ```bash
      conda update nombre_del_paquete
      ```

   </TabItem>
   <TabItem value="poetry" label="Poetry">

      ##### Actualizar todos los paquetes

      ```bash
      poetry update
      ```

      ##### Actualizar un paquete específico

      Para actualizar un paquete específico:

      ```bash
      poetry update nombre_del_paquete
      ```

   </TabItem>

   <TabItem value="uv" label="uv">
      ```bash
      uv pip install --upgrade nombre_del_paquete
      ```
   </TabItem>
</Tabs>

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

   <Tabs>
      <TabItem value="pip" label="PIP">
         ```bash
         pip install -r requirements.txt
         ```
      </TabItem>

      <TabItem value="poetry" label="Poetry">
         ```bash
         poetry install
         ```
      </TabItem>

      <TabItem value="uv" label="uv">
         ```bash
         uv pip install -r requirements.txt
         ```
      </TabItem>
   </Tabs>

### 2.5. Eliminar un entorno

<Tabs>
   <TabItem value="venv" label="VENV, Poetry, uv">

      En la mayoría de los casos, los entornos creados con ``VENV``, ``Poetry`` y ``uv`` se alojan dentro del propio directorio del proyecto. Por ello, si ya no los necesitas, basta con eliminar la carpeta correspondiente para borrar por completo el entorno junto con toda su información.

      ```bash
      rm -rf nombre_del_entorno
      ```

   </TabItem>
   <TabItem value="anaconda" label="Anaconda">

      1. **Listar los entornos disponibles**:

         ```bash
         conda env list
         ```

      2. **Eliminar un entorno**:

         ```bash
         conda env remove --name nombre_del_entorno
         ```

   </TabItem>
</Tabs>

### 2.6. Integración del entorno con Jupyter

Para utilizar un entorno virtual dentro de **Jupyter**, es necesario seguir estos pasos:

1. **Instalar `ipykernel` en el entorno** Primero, debes añadir `ipykernel` como
   dependencia dentro del entorno virtual. Para ello, instala el paquete utilizando el
   gestor de dependencias correspondiente (por ejemplo, `pip`, `poetry`, `uv` o
   `conda`).

2. **Registrar el entorno en Jupyter** Este paso es necesario únicamente cuando el
   entorno virtual se encuentra en un directorio diferente al del proyecto. En la
   mayoría de los entornos de desarrollo, como **VSCode**, si el entorno está dentro del
   directorio del proyecto, se detectará automáticamente y podrás seleccionar el kernel
   asociado sin pasos adicionales.

   En caso de que necesites registrar el entorno manualmente, ejecuta:

   ```bash
   python -m ipykernel install --user --name=nombre_del_entorno
   ```

### 2.7. Eliminación de paquetes instalados

<Tabs>
   <TabItem value="pip" label="PIP">

      Eliminar todos los paquetes

      ```bash
      pip list --format=freeze > installed.txt
      pip uninstall -r installed.txt -y
      ```

      Eliminar un paquete específico

      ```bash
      pip uninstall nombre_del_paquete
      ```

   </TabItem>
   <TabItem value="anaconda" label="Anaconda">

      ```bash
      conda remove nombre_del_paquete
      ```

   </TabItem>
   <TabItem value="poetry" label="Poetry">

      ```bash
      poetry remove nombre_del_paquete
      ```

   </TabItem>
   <TabItem value="uv" label="uv">
      ```bash
      uv pip uninstall nombre_del_paquete
      ```
   </TabItem>
</Tabs>
