# Software Engineering for Data Scientist

# Capítulo 1: Calidad del Código

El código de alta calidad puede definirse de varias maneras. Podría ser el código que se ejecuta más rápido o el que es más fácil de leer. Otra posible definición es que el buen código es fácil de mantener. Es decir, si el proyecto cambia, debería ser fácil volver al código y modificarlo para reflejar los nuevos requisitos. Se debe esperar cambiar el código a medida que evoluciona el proyecto.

![Untitled](Untitled.png)

Un aspecto crucial para mantener la calidad del código es la modularización. La modularización del código implica dividir un programa en subprogramas más pequeños o módulos, cada uno de los cuales realiza una tarea específica. Esto mejora la legibilidad y facilita el mantenimiento del código.

# Capítulo 2: Mejorar el Rendimiento del Código

## Estrategias de Optimización

- **Elección del algoritmo:** Las decisiones que tomes en el código que escribes pueden marcar una gran diferencia en su rendimiento.
- **Elección de la estructura de datos:** Dependiendo de la tarea que necesites realizar, diferentes estructuras de datos pueden tener diferentes compensaciones.
- **Uso de funciones incorporadas:** Casi siempre es más eficiente usar esa función que escribir la tuya propia. Muchas de estas funciones incorporadas están implementadas en C.
- **Compilación de Python:** Puedes hacer que tu código se ejecute más rápido compilando Python a un lenguaje de nivel inferior utilizando herramientas como Cython, Numba y PyPy. Tu elección de cuál usarás dependerá de tu caso de uso. Numba contiene un subconjunto de Python; Cython es un superconjunto de Python con opciones adicionales en C; y PyPy es una reimplantación de Python usando compilación justo a tiempo.
- **Código asíncrono**
- **Computación paralela y distribuida**

## Medición del Tiempo de Ejecución

Para medir el tiempo de ejecución de un bloque de código, puedes utilizar el módulo `time` de Python de la siguiente manera:

```python
import time

start = time.time()
slow_way_to_calculate_mode(random_integers)
end = time.time()

print(end-start)

```

En un Jupyter notebook, puedes usar la función mágica `%%timeit` al principio de cada bloque de código para medir el tiempo de ejecución.

También puedes utilizar `timeit` en un script normal de Python de la siguiente manera:

```python
import numpy as np
import timeit

random_integers = np.random.randint(1, 100_000, 1000)

def slow_way_to_calculate_mode(list_of_numbers):
    result_dict = {}
    for i in list_of_numbers:
        if i not in result_dict:
            result_dict[i] = 1
        else:
            result_dict[i] += 1

    mode_vals = []
    max_frequency = max(result_dict.values())
    for key, value in result_dict.items():
        if value == max_frequency:
            mode_vals.append(key)

    return mode_vals

mode_timer = timeit.Timer(stmt="slow_way_to_calculate_mode(random_integers)",
                          setup="from __main__ import"\
                          "slow_way_to_calculate_mode,random_integers")

time_taken = mode_timer.timeit(number=10)

print(f"Tiempo de ejecución: {time_taken} segundos")

```

## Perfilado de Código

El perfilado de código es una técnica que permite medir el rendimiento de diferentes partes de un programa. El perfilado puede ayudarte a identificar qué partes de tu código son más lentas y podrían beneficiarse de la optimización.

`cProfile` es el perfilador incorporado de Python, y puedes usarlo para obtener una visión general básica de los lugares donde se encuentran los cuellos de botella en un script más largo. En este ejemplo, pondré el generador de números aleatorios dentro de la función de modo de la sección anterior para que haya más que ver en el perfilador:

```python
import numpy as np
from collections import Counter

def mode_using_counter(n_integers):
    random_integers = np.random.randint(1, 100_000, n_integers)
    c = Counter(random_integers)
    return c.most_common(1)[0][0]
```

Para ejecutar el perfilador, usa el comando:

```python
%%prun
mode_using_counter(10_000_000)
```

La columna `tottime` en esta salida muestra dónde pasó la mayor parte del tiempo el ordenador. Todos los demás pasos tomaron muy poco tiempo. La desventaja de usar `cProfile` es que necesitas mapear cada una de estas llamadas de función a líneas dentro de tu código.

También puedes usar el paquete SnakeViz para obtener una visualización gráfica de los resultados de `cProfile`. Puedes instalar SnakeViz con el siguiente comando:

```
$ pip install snakeviz
```

Luego, si estás trabajando en el Jupyter Notebook puedes usar la extensión SnakeViz. Puedes cargar la extensión con el siguiente comando:

```
%load_ext snakeviz
```

Y luego puedes ejecutar SnakeViz usando:

```
%%snakeviz
mode_using_counter(10_000_000)
```

Memray es una herramienta de perfilado de memoria desarrollada por Bloomberg que puede darte diferentes informes sobre el uso de memoria de tu código. Puedes instalar Memray usando este comando:

```
$ pip install memray
```

Veamos cómo usar Memray con un script de Python independiente que contiene la función `mode_using_counter`. Aquí está el script completo:

```python
import numpy as np
from collections import Counter

def mode_using_counter(n_integers):
    random_integers = np.random.randint(1, 100_000, n_integers)
    c = Counter(random_integers)
    return c.most_common(1)[0][0]

if __name__ == '__main__':
    print(mode_using_counter(10_000_000))
```

Necesitarás ejecutar Memray usando el siguiente comando para recopilar datos sobre el uso de memoria de tu script:

```
$ memray run mode_using_counter.py
```

# Capítulo 3: Optimización de la Memoria y el Rendimiento

A diferencia de una lista regular de Python, cuando NumPy asigna espacio para un array, no permite ningún espacio extra. Por lo tanto, si añades más elementos a un array de NumPy, todo el array necesita ser movido a una nueva ubicación de memoria cada vez. Esto significa que añadir a un array de NumPy es de complejidad *O(n)*. Definitivamente vale la pena inicializar tu array con la cantidad correcta de espacio, y una forma fácil de hacer esto es usar `np.zeros`, de la siguiente manera:

```python
array_to_fill = np.zeros(1000)

```

Con ello ahorramos mucha memoria, ya que las listas de forma nativa en Python son dinámicas. Los arrays de NumPy se cargan en memoria.

Otro de los métodos que menciona para reducir la memoria es el uso de valores de precisión acordes al rango utilizado. Ya que en general tanto Pandas como Numpy por defecto suelen procesar los datos en fp64, por lo que podemos procesar los datos en 32, 16 o incluso menos bits de información.

## Operaciones de Array Usando Dask

Si has probado las estrategias en la sección anterior para mejorar el rendimiento de tu código usando arrays de NumPy, pero tu código aún necesita ser optimizado aún más, la biblioteca Dask es una gran opción. Te permite realizar operaciones de array en paralelo, para una computación más rápida y para datos que no caben en la memoria de tu ordenador. Dask proporciona una interfaz muy similar a los arrays estándar de NumPy, pero añade un poco de complejidad extra, por lo que vale la pena usarlo sólo si necesitas el aumento de rendimiento. Te permite ejecutar cálculos en varios núcleos a la vez en tu portátil y en sistemas distribuidos (clusters).

Dask funciona dividiendo un array en trozos, ejecutando cálculos en uno o varios trozos a la vez, luego combina los resultados. Por ejemplo, si quieres encontrar el valor máximo de un array muy grande, podrías dividir ese array en un número de trozos, encontrar el máximo de cada trozo, luego encontrar el máximo de todos los resultados de cada trozo combinado. No todas las operaciones pueden ser paralelizadas de esta manera, pero si esto se aplica al problema en el que estás trabajando, puede hacer que tu código sea mucho más eficiente.

Dask también te permite ejecutar cálculos en datos que son más grandes que la memoria de tu sistema. Debido a que no todos los trozos se cargan y evalúan a la vez, no necesitas cargar todo el array en memoria, y cada trozo puede ser evaluado secuencialmente.

Puedes instalar Dask con el siguiente comando:

```python
$ python -m pip install "dask[complete]"

```

Puedes ejecutar la misma operación con NumPy y con Dask, y comparar la cantidad de tiempo que tarda.

Un experimento que puedes hacer es encontrar el valor máximo de un array grande. Puedes crear un array de NumPy lleno de enteros aleatorios usando `np.random.randint()`, y el código de abajo crea un array de 1 billón de enteros:

```python
large_np_array = np.random.randint(1, 100000, 1000000000)

```

Puedes medir el tiempo que tarda en hacer este cálculo en un array estándar de NumPy:

```python
>>>%%timeit -r1 -n7
>>>np.max(large_np_array)
...30.7s±0ns per loop (mean±std.dev. of 1 run, 7 loops each)

```

El array de Dask es una estructura de datos diferente a un array de NumPy. Muchos métodos de NumPy están replicados en Dask, por lo que puedes crear un array de Dask de enteros aleatorios con este código:

```python
import dask.array as da

large_dask_array = da.random.randint(1, 100_000, 1_000_000_000)

```

También puedes crear un array de Dask a partir de un array de NumPy, de la siguiente manera:

```python
large_dask_array = da.from_array(large_np_array)

```

Hay un paso extra con Dask en comparación con NumPy. Primero necesitas inicializar la operación, en este caso con el método `.max()`. Luego necesitas calcular la operación usando el método `.compute()`. Puedes medir el tiempo para este paso, para comparar con el array de NumPy:

```python
>>>%%timeit -r1 -n7
>>>array_max = large_dask_array.max()
>>>array_max.compute()
...1.51s±0ns per loop (mean±std.dev. of 1 run, 7 loops each)

```

¡Encontrar el máximo en un array de Dask es aproximadamente 20 veces más rápido que con un array de NumPy!

El cálculo en cada trozo también puede ser realizado por un núcleo o máquina diferente. Dask Distributed programa las tareas para ti. Necesitas crear un objeto `Client` para usar esto:

```python
from dask.distributed import Client

client = Client(n_workers=4)
client

```

Una vez que este cliente está listo, puedes usar arrays de Dask como antes, y los cálculos se distribuirán entre el número de trabajadores que especifiques.

Si quieres aprender más sobre Dask, consulta la documentación de Dask para ver algunos grandes ejemplos.

## Pandas

Por defecto, un DataFrame de pandas se carga en memoria. Así que si tu DataFrame es más grande que la memoria de tu ordenador, tienes un problema.

Primero, puedes simplemente cargar sólo las columnas en las que quieres trabajar. La reciente introducción de PyArrow como un backend opcional de pandas también da soporte para tipos de datos más eficientes en memoria.

Otra opción para grandes cantidades de datos es usar la biblioteca Dask. Como viste con NumPy, si tus datos son demasiado grandes para caber en memoria, Dask puede dividirlos. También es una gran opción si tu procesamiento de datos es lento y quieres paralelizarlo en varios núcleos o máquinas. Dask tiene su propia estructura de datos DataFrame, y puedes crear uno directamente desde tus datos o desde un DataFrame de pandas existente.