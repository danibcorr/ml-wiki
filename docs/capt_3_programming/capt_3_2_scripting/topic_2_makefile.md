---
sidebar_position: 2
authors:
  - name: Daniel Bazo Correa
description: Automatiza tus procesos con Makefile.
title: Makefile
toc_max_heading_level: 3
---

## Bibliografía

- [Makefile Tutorial](https://makefiletutorial.com/)

## 1. Introducción

<p align="center">
  <img src="/assets/img/docs/logos/makefile-logo.png" width="300"/>
  <br />
  <em>Logo de Makefile</em>
</p>

Un **Makefile** es un archivo de configuración utilizado por la herramienta `make`, que
permite automatizar el proceso de compilación y ejecución de proyectos de software.
Generalmente, se emplea en entornos con sistemas operativos basados en GNU/Linux y puede
contener comandos en [Bash](./topic_1_bash.md).

El uso de Makefiles ofrece varios beneficios en el desarrollo de software, pero el que
más destacada es la capacidad de automatización de tareas repetitivas.

## 2. Sintaxis básica

### 2.1. Estructura de una regla

Una regla en un Makefile define el proceso de construcción de un objetivo (**_target_**)
a partir de sus prerrequisitos. La estructura básica de una regla es la siguiente:

```makefile
targets: prerequisites
  comando
  comando
  comando
```

- **_targets:_** Son los nombres de los archivos o tareas que se crearán, separados por
  espacios. Usualmente, se define un único _target_ por regla.
- **_prerequisites:_** Son los archivos o dependencias necesarios para generar el
  _target_, también separados por espacios.
- **comandos:** Son las instrucciones que se ejecutan para crear el _target_. Cada
  comando debe comenzar con un carácter de tabulación, no con espacios.

!!! example "Ejemplo"

```makefile
install: pyproject.toml
    poetry install
```

En este ejemplo:

- **`install`** es el _target_ u "objetivo". Representa la tarea a realizar o el archivo
  que se va a generar. En este caso, es una tarea llamada `install` que indica que se
  deben instalar las dependencias del proyecto.
- **`pyproject.toml`** es el _prerrequisito_. Es un archivo que debe existir antes de
  ejecutar la tarea. En este caso, `pyproject.toml` es un archivo de configuración de
  Poetry, que se usa para gestionar las dependencias de un proyecto Python.
- **`poetry install`** es el comando que se ejecutará para completar la tarea. En este
  caso, el comando instala las dependencias especificadas en `pyproject.toml`.

Cuando ejecutas el comando `make install`, Make verifica si el archivo `pyproject.toml`
existe. Si el archivo está presente, Make ejecutará `poetry install` para instalar las
dependencias del proyecto. Si el archivo no existe, Make mostrará un error.

### 2.2. Comentarios

Los comentarios en un Makefile se escriben utilizando el símbolo `#`. Estos comentarios
no afectan la ejecución del archivo y sirven para describir el propósito de las reglas o
comandos.

!!! example "Ejemplo"

```makefile
# Esta regla instala las dependencias de Poetry
install: pyproject.toml
    poetry install
```

### 2.3. Variables

Las variables en Makefiles permiten almacenar y reutilizar valores, facilitando la
personalización de comandos o rutas.

!!! example "Ejemplo"

```makefile
TEST_FILE ?= ./tests

# Regla para ejecutar tests al código
tests:
    @echo "Testeando el código..."
    poetry run pytest -v $(TEST_FILE)
```

En este caso:

- **`TEST_FILE ?= ./tests`** define la variable `TEST_FILE`, que almacena la ruta del
  directorio donde se ejecutarán los tests. El uso de `?=` permite definir un valor por
  defecto si no se proporciona otro al ejecutar la regla.
- En la regla `tests`, el comando `poetry run pytest -v $(TEST_FILE)` utiliza la
  variable `TEST_FILE` para ejecutar los tests en el directorio especificado.

Para ejecutar la regla `tests` y especificar un archivo diferente al definido en la
variable, se utiliza el siguiente comando:

```sh
make tests TEST_FILE=./tests/test_ejemplo.py
```

Esto ejecutará los tests usando el archivo `test_ejemplo.py` en lugar del directorio por
defecto.

#### 2.3.1. Variables automáticas

Make proporciona variables automáticas que son útiles para simplificar reglas. Estas
variables permiten referirse de manera dinámica a los _targets_ y prerrequisitos sin
necesidad de escribirlos explícitamente cada vez.

| Variable | Descripción                                               |
| -------- | --------------------------------------------------------- |
| `$@`     | Nombre del _target_ actual.                               |
| `$<`     | Primer prerrequisito de la regla.                         |
| `$^`     | Todos los prerrequisitos de la regla.                     |
| `$*`     | Parte del nombre (_stem_) que coincide con el patrón `%`. |
| `$(@D)`  | Directorio del _target_ actual.                           |
| `$(@F)`  | Nombre del archivo del _target_ actual.                   |

!!! example "Ejemplo"

Supongamos que estamos procesando datos en Python y queremos automatizar la creación de
un archivo comprimido a partir de varios archivos generados:

```makefile
all: process_data archive

process_data: input1.txt input2.txt script.py
    python $< -o $@

archive: output.zip
    echo "Archivo comprimido generado: $@"

output.zip: processed_data/input1.csv processed_data/input2.csv
    zip $@ $^
```

- `$<`: Se refiere al primer prerrequisito. En la regla `process_data`, `$<` será
  `input1.txt`.
- `$@`: Representa el _target_ actual. En la regla `archive`, `$@` será `output.zip`.
- `$^`: Contiene todos los prerrequisitos. En la regla `output.zip`, `$^` incluye
  `processed_data/input1.csv processed_data/input2.csv`.

El script `script.py` podría utilizar los prerrequisitos para producir archivos de
salida:

```python
import sys
import os

def process_file(input_file, output_dir):

    output_file = os.path.join(output_dir, os.path.basename(input_file).replace(".txt", ".csv"))
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            outfile.write(f"Processed: {line}")
    return output_file

if __name__ == "__main__":

    input_file = sys.argv[1]
    output_dir = "processed_data"
    os.makedirs(output_dir, exist_ok=True)
    output_file = process_file(input_file, output_dir)
    print(f"Output saved to {output_file}")
```

Al ejecutar `make all` este produce:

- Archivos procesados en `processed_data/` a partir de los prerrequisitos.
- Archivo comprimido `output.zip` que contiene los archivos procesados.

En la regla `process_data`, Python recibe como entrada el primer prerrequisito
`input1.txt` a través de `$<` y genera la salida en `processed_data/input1.csv`
automáticamente.

#### 2.3.2. Variables específicas de objetivo y patrones

En Make, las variables específicas de objetivo y patrones permiten definir
configuraciones particulares para ciertos objetivos o archivos.

!!! example "Ejemplo"

Si queremos procesar archivos .csv para generar archivos .txt con una configuración
específica para cada tipo de archivo:

```makefile
# Se agrega una opción de optimización para la generación de archivos intermedios
%.txt: PYTHONFLAGS += --optimize

# Definir la regla para procesar archivos .csv y convertirlos en .txt
%.txt: %.csv
    python3 process_data.py $< $@
```

En este ejemplo:

- El patrón `%.txt` indica que cualquier archivo con extensión `.csv` se convertirá en
  un archivo `.txt`. Por ejemplo, data.csv se convertirá en data.txt.
- La variable específica `PYTHONFLAGS += --optimize` se aplicará al comando Python solo
  cuando se genere un archivo `.txt`.
- La regla `%.txt: %.csv` usa el archivo `.csv` como entrada (representado por `$<`) y
  genera un archivo `.txt` como salida (representado por `$@`).

## 3. Funciones avanzadas

### 3.1. Funciones para cadenas de texto

Make ofrece funciones que facilitan la manipulación de cadenas de texto.

#### 3.1.1. Función `subst`

La función `subst` reemplaza un texto por otro en una cadena.

note Sintaxis

```makefile
$(subst from,to,text)
```

- **from**: Texto a reemplazar.
- **to**: Texto de reemplazo.
- **text**: Cadena donde se hace la búsqueda.

!!! example "Ejemplo"

```makefile
SOURCES = file1.cpp file2.cpp file3.cpp
OBJECTS = $(subst .cpp,.o,$(SOURCES))
```

Aquí, `subst` reemplaza `.cpp` por `.o` en la lista de archivos, generando
`file1.o file2.o file3.o`.

#### 3.1.2. Función `patsubst`

La función `patsubst` permite hacer sustituciones usando patrones (como `%`).

note Sintaxis

```makefile
$(patsubst pattern,replacement,text)
```

- **pattern**: Patrón a buscar (puede usar `%`).
- **replacement**: Texto con el cual reemplazar el patrón.
- **text**: Texto donde buscar el patrón.

!!! example "Ejemplo"

```makefile
SOURCES = file1.cpp file2.cpp file3.cpp
OBJECTS = $(patsubst %.cpp,%.o,$(SOURCES))
```

Este ejemplo reemplaza `.cpp` por `.o`, igual que el anterior, pero usando un patrón
para mayor flexibilidad.

#### 3.1.3. Funciones `filter` y `filter-out`

Estas funciones permiten filtrar listas.

- **`filter`**: Mantiene las palabras que coinciden con un patrón.
- **`filter-out`**: Elimina las palabras que coinciden con un patrón.

note Sintaxis

```makefile
$(filter pattern...,text)
$(filter-out pattern...,text)
```

!!! example "Ejemplo"

```makefile
SOURCES = file1.c file2.cpp file3.h
C_FILES = $(filter %.c,$(SOURCES))
```

Aquí, `filter` selecciona solo los archivos `.c`, resultando en `file1.c`.

#### 3.1.4. Función `foreach`

La función `foreach` permite iterar sobre una lista y aplicar una operación a cada
elemento.

note Sintaxis

```makefile
$(foreach var,list,text)
```

- **var**: Variable que tomará cada valor de la lista.
- **list**: Lista sobre la que se iterará.
- **text**: Texto que se evaluará para cada valor.

!!! example "Ejemplo"

```makefile
DIRS = dir1 dir2 dir3
CLEAN_DIRS = $(foreach dir,$(DIRS),$(dir)/clean)
```

Este ejemplo crea la lista `CLEAN_DIRS` con las rutas `dir1/clean`, `dir2/clean` y
`dir3/clean`.

#### 3.1.5. Función `if`

La función `if` permite ejecutar algo según una condición.

note Sintaxis

```makefile
$(if condition,then-part[,else-part])
```

- **condition**: Condición a evaluar.
- **then-part**: Acción si la condición es verdadera.
- **else-part**: Acción si la condición es falsa (opcional).

!!! example "Ejemplo"

```makefile
USE_DEBUG = yes
CFLAGS = $(if $(USE_DEBUG),-g,-O2)
```

Si `USE_DEBUG` es `yes`, se añade `-g` para depuración. Si no, se usa `-O2` para
optimización.

### 3.2. Directivas

Las directivas en Make controlan el flujo de ejecución, la inclusión de archivos y otras
configuraciones avanzadas. Son herramientas poderosas para modularizar y personalizar el
comportamiento de un Makefile.

#### 3.2.1. Directiva `include`

La directiva `include` permite incluir otros Makefiles dentro de uno principal. Esto
ayuda a organizar el código de manera modular y facilita el mantenimiento al separar
configuraciones y reglas en archivos diferentes.

!!! example "Ejemplo"

```makefile
include config.mk
```

Este comando incluirá el contenido de `config.mk` en el Makefile actual, lo que permite
reutilizar configuraciones o reglas comunes en varios Makefiles.

#### 3.2.2. Directiva `VPATH`

La directiva `VPATH` especifica directorios adicionales donde Make buscará los archivos
necesarios, como los archivos fuente o de cabecera. Esto es útil cuando los archivos no
están en el mismo directorio que el Makefile y se quiere mantener una estructura de
proyecto ordenada.

!!! example "Ejemplo"

```makefile
VPATH = src:include
```

En este caso, Make buscará primero en el directorio `src` y luego en `include` para
encontrar los archivos necesarios. Esto es útil cuando tienes los archivos fuente y los
archivos de cabecera en directorios separados.

#### 3.2.3. Directiva `.PHONY`

La directiva `.PHONY` se utiliza para declarar objetivos que no corresponden a archivos
reales en el sistema de archivos. Esto es importante para evitar que Make intente buscar
archivos con el mismo nombre que el objetivo y así prevenir conflictos.

!!! example "Ejemplo"

```makefile
.PHONY: clean all
```

Aquí, `clean` y `all` son objetivos "falsos", ya que no representan archivos reales en
el sistema, sino tareas o comandos que Make debe ejecutar.

#### 3.2.4. Directiva `.DELETE_ON_ERROR`

La directiva `.DELETE_ON_ERROR` indica que Make debe eliminar un archivo de objetivo si
un comando falla durante su ejecución. Esto es útil para evitar que queden archivos
incompletos o corruptos cuando un proceso de compilación falla.

note Sintaxis

```makefile
.DELETE_ON_ERROR:
```

Esto asegura que cualquier archivo generado se eliminará si ocurre un error en su
construcción, manteniendo el sistema limpio.

### 3.3. Condicionales

Makefiles permiten el uso de estructuras condicionales para adaptar las reglas según
diferentes entornos o configuraciones. Esto es útil para crear Makefiles más flexibles y
reutilizables.

note Sintaxis

```makefile
ifeq (condición)
    acción
else
    acción
endif
```

- **condición**: La condición a evaluar (puede ser una variable o expresión).
- **acción**: La acción a realizar si la condición es verdadera o falsa.

!!! example "Ejemplo"

```makefile
ifeq ($(USE_DEBUG),yes)
    CFLAGS = -g
else
    CFLAGS = -O2
endif
```

En este caso, si la variable `USE_DEBUG` es `yes`, Make utilizará las banderas de
compilación para depuración (`-g`). Si no, se utilizarán las banderas de optimización
(`-O2`).

### 3.4. Macros y funciones

Make permite definir macros y funciones personalizadas para agrupar comandos y mejorar
la legibilidad del Makefile. Estas macros ayudan a evitar la repetición y facilitan la
reutilización del código.

note Sintaxis de una macro

```makefile
define nombre_de_macro
    comandos
endef
```

- **nombre_de_macro**: El nombre de la macro que se define.
- **comandos**: Los comandos que ejecutará la macro.

!!! example "Ejemplo" de macro

```makefile
define compile_rule
    $(CC) $(CFLAGS) -c $< -o $@
endef
```

Aquí, `compile_rule` es una macro que contiene la regla para compilar archivos `.c` en
archivos `.o`.

!!! example "Ejemplo" de invocación de macro

```makefile
%.o: %.c
    $(call compile_rule)
```

La regla `%.o: %.c` invoca la macro `compile_rule` con `$(call compile_rule)`, lo que
permite reutilizar la misma lógica de compilación en múltiples reglas.

## 4. Mejores prácticas y estilos

El uso adecuado de Makefiles no solo facilita la compilación y gestión de proyectos,
sino que también mejora la legibilidad y mantenimiento a largo plazo. A continuación, se
presentan algunas de las mejores prácticas y estilos recomendados.

### 4.1. Organización de Makefiles

Es recomendable organizar el Makefile de manera que sea fácil de leer y mantener.
Algunas sugerencias incluyen:

1. **Separar las reglas y configuraciones:** Definir las variables al inicio del
   Makefile y agrupar las reglas relacionadas. Esto facilita el mantenimiento y
   comprensión del archivo.
2. **Uso de comentarios:** Añadir comentarios claros y concisos para explicar las
   reglas, variables y funciones dentro del Makefile.
3. **Modularización:** Dividir los Makefiles grandes en varios archivos pequeños y
   organizados, utilizando la directiva `include`.

!!! example "Ejemplo" de organización

```makefile
# Variables de configuración
CC = gcc
CFLAGS = -Wall -O2

# Objetivos
all: programa

# Reglas de compilación
programa: main.o utils.o
    $(CC) $(CFLAGS) -o programa main.o utils.o

main.o: main.c
    $(CC) $(CFLAGS) -c main.c

utils.o: utils.c
    $(CC) $(CFLAGS) -c utils.c

# Limpiar los archivos generados
clean:
    rm -f *.o programa
```

En este ejemplo, las variables de configuración se definen al inicio del archivo,
seguidas de los objetivos, las reglas de compilación y finalmente la regla de limpieza.

### 4.2. Depuración

La depuración de Makefiles puede ser compleja si no se siguen ciertas prácticas. Algunas
técnicas útiles incluyen:

1. **Ejecución en seco (`-n`):** Esta opción permite ver qué comandos se ejecutarían sin
   realmente ejecutarlos, lo cual es útil para verificar el flujo de ejecución.

   note Sintaxis

   ```sh
   make -n
   ```

   Esto imprimirá los comandos que se ejecutarían sin hacer ninguna modificación en los
   archivos.

2. **Depuración detallada (`-d`):** Proporciona una salida de depuración detallada para
   ayudar a identificar errores en la ejecución del Makefile.

   note Sintaxis

   ```sh
   make -d
   ```

   Esto muestra información detallada sobre cómo Make procesa las reglas y dependencias.

3. **Impresión de variables:** Para comprobar el valor de las variables, se puede
   definir una regla que las imprima.

   !!! example "Ejemplo"

   ```makefile
   debug:
       @echo "CFLAGS = $(CFLAGS)"
   ```

   Al ejecutar:

   ```sh
   make debug
   ```

   Se imprimirá el valor actual de la variable `CFLAGS`, ayudando a verificar su
   configuración y valor.
