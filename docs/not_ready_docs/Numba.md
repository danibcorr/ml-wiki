# Numba

- Numba can be used to accelerate Python functions for the CPU, as well as for NVIDIA GPUs.
- Podemos acelerar operaciones elemento a elemento en arrays de Numpy así como realizar operaciones de mover datos de CPU a GPU de manera más eficiente.
- Al igual que Cuda en C, podemos crear Kernels customs para Python utilizando Numba, ejecutando codigo en paralelo.

## Que es numba

Numba es un compilador de funciones just-in-time y type-specializaing para acelerar calculo numerico en Python tanto en CPU como en GPU.

- Numba compila funciones de Python, no la aplicacion al completo, por lo que no es un sustituto del interprete de Python.
- Numba acelera las funciones generando implementaciones especificas del tipo del dato que se utiliza, en vez de hacerlo dynamic typing, que es el funcionamiento por defecto de Python.
- Que sea just-in-time, significa que compila la funcion cuando se le llama, con el fin de que el compilador conozca que argumentos van a ser utilizados en la funcion ademas de su ejecucion de forma interactiva en los cuadernos Jupyter.
- Numba se centra principalmente en tipos de datos numericos, enteros, flotantes, numeros complejos…Y obtenedremos el mejor soporte siempre que utilicemos array de NumPy.

## Alternativas a Numba

- **CUDA C/C++**:
    - The most common, performant, and flexible way to utilize CUDA
    - Accelerates C/C++ applications
- **pyCUDA**:
    - Exposes the entire CUDA C/C++ API
    - Is the most performant CUDA option available for Python
    - Requires writing C code in your Python, and in general, a lot of code modifications
- **Numba**: → Al final es la mejor opción en tiempo/beneficio
    - Potentially less performant than pyCUDA
    - Does not (yet?) expose the entire CUDA C/C++ API
    - Still enables massive acceleration, often with very little code modification
    - Allows developers the convenience of writing code directly in Python
    - Also optimizes Python code for the CPU

## Que hace por debajo

Cuando se llama a una funcion de Numba con el decorador de jit o njit, el compilador compila el codigo de Python a codigo maquina para el tipo especifico de los datos que se están utiliazndo en la funcion. 

Numba tambien guarda la funcion original de Python en el atributo  `.py_func` lo que permite llamar a la funcion junto con ese atributo para comprobar el resultado de la respuesta:

```python
from numba import jit
import math

# This is the function decorator syntax and is equivalent to `hypot = jit(hypot)`.
# The Numba compiler is just a function you can call whenever you want!
@jit
def hypot(x, y):
    # Implementation from https://en.wikipedia.org/wiki/Hypot
    x = abs(x);
    y = abs(y);
    t = min(x, y);
    x = max(x, y);
    t = t / x;
    return x * math.sqrt(1+t*t)
    
hypot(3.0, 4.0)

hypot.py_func(3.0, 4.0)
```

PERO, si nos encontramos con versiones ya implementadas de Python, estas serán generalmente incluso más rápidas que Numba, ya que Numba introduce una pequeña sobrecarga.

Esto es lo que ocurre por debajo al compilar una funcion con Numba:

![image.png](Numba/image.png)

Podemos ver el resultado del tipo de la inferencia utilizando el metodo .inspect_types() que hace un print del codigo fuente

```python
hypot.inspect_types()
```

## El uso de njit

Es la version recomendada del decorador y la más eficiente ya que fuerza a mostrar los errores de los tipos de estructuras o funciones que no son directamente compatibles con Numba y que no se pueden compilar, sin hacer object-mode (utilizar el tipo de variable original de Python sin especializar el tipo en Numba), que se le pasa a la funcion.

## Numba en la GPU para operaciones elemento-a-elemento, vectorizacion

Compilaremos funciones universales de NumPy, tambien conocidas como ufuncs, para la GPU.

- El hardware de la GPU esta pensad/disenada para la paralelizacion de datos. Por lo quue se obtiene el maximo throughput cuando la GPU esta calculando la misma operacion para diferentes elementos al mismo tiempo. → Las funciones universales de NumPy realizan la misma operacion en cada elemento de un array de NumPy, son naturalmente paralelizables, por lo quue se ajustan muy bien a la naturaleza de la GPU.

El realidad las funciones universales de NumPy son funciones que pueden tomar arrays de NumPy de cualquier dimension o escalares y operar elemento-a-elemento.

Un ejemplo para la CPU:

```python
from numba import vectorize

@vectorize
def add_ten(num):
    return num + 10 # This scalar operation will be performed on each element
    
nums = np.arange(10)
add_ten(nums) # pass the whole array into the ufunc, it performs the operation on each element
```

Un ejemplo para la GPU:

```python
@vectorize(['int64(int64, int64)'], target='cuda') # Type signature and target are required for the GPU
def add_ufunc(x, y):
    return x + y
    
add_ufunc(a, b)
```

Vemos que en el caso de la GPU especificamos el targe, CUDA en este caso, asi como el typing especifco de las variables (lo que esta dentro de los parentesis) y el typing especifoc de lo que devuelve la funcion (lo que esta fuera del parentesis). En este caso, la funcion toma como argumentos dos parametros de tipo entero de 64 bits y devuelve el mismo tipo de 64 bits entero.

Por debajo:

- Compiled a CUDA kernel to execute the ufunc operation in parallel over all the input elements.
- Allocated GPU memory for the inputs and the output.
- Copied the input data to the GPU.
- Executed the CUDA kernel (GPU function) with the correct kernel dimensions given the input sizes.
- Copied the result back from the GPU to the CPU.
- Returned the result as a NumPy array on the host.

Algunas consideraciones a tener en cuenta cuando queramos obtener funciones paralelizables con la GPU:

- **Our inputs are too small**: the GPU achieves performance through parallelism, operating on thousands of values at once. Our test inputs have only 4 and 16 integers, respectively. We need a much larger array to even keep the GPU busy.
- **Our calculation is too simple**: Sending a calculation to the GPU involves quite a bit of overhead compared to calling a function on the CPU. If our calculation does not involve enough math operations (often called "arithmetic intensity"), then the GPU will spend most of its time waiting for data to move around.
- **We copy the data to and from the GPU**: While in some scenarios, paying the cost of copying data to and from the GPU can be worth it for a single function, often it will be preferred to to run several GPU operations in sequence. In those cases, it makes sense to send data to the GPU and keep it there until all of our processing is complete.
- **Our data types are larger than necessary**: Our example uses `int64` when we probably don't need it. Scalar code using data types that are 32 and 64-bit run basically the same speed on the CPU, and for integer types the difference may not be drastic, but 64-bit floating point data types may have a significant performance cost on the GPU, depending on the GPU type. Basic arithmetic on 64-bit floats can be anywhere from 2x (Pascal-architecture Tesla) to 24x (Maxwell-architecture GeForce) slower than 32-bit floats. If you are using more modern GPUs (Volta, Turing, Ampere), then this could be far less of a concern. NumPy defaults to 64-bit data types when creating arrays, so it is important to set the [`dtype`](https://docs.scipy.org/doc/numpy-1.14.0/reference/arrays.dtypes.html) attribute or use the [`ndarray.astype()`](https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.ndarray.astype.html) method to pick 32-bit types when you need them.

Given the above, let's try an example that is faster on the GPU by performing an operation with much greater arithmetic intensity, on a much larger input, and using a 32-bit data type.

```python
import math # Note that for the CUDA target, we need to use the scalar functions from the math module, not NumPy

SQRT_2PI = np.float32((2*math.pi)**0.5)  # Precompute this constant as a float32.  Numba will inline it at compile time.

# Para la GPU tenemos que especificar el typing de las variables y del resultado
# Para algunas operaciones como el coseno, el seno, exp, las versiones de NumPy
# no seran compatibles, por lo que tendremos que utilizar la libreria math.
# Ademas este tipo de operaciones funcionan muy bien en la GPU
@vectorize(['float32(float32, float32, float32)'], target='cuda')
def gaussian_pdf(x, mean, sigma):
    '''Compute the value of a Gaussian probability density function at x with given mean and sigma.'''
    return math.exp(-0.5 * ((x - mean) / sigma)**2) / (sigma * SQRT_2PI)
    
import numpy as np
# Evaluate the Gaussian a million times!
x = np.random.uniform(-3, 3, size=1000000).astype(np.float32)
mean = np.float32(0.0)
sigma = np.float32(1.0)

# Quick test on a single element just to make sure it works
gaussian_pdf(x[0], 0.0, 1.0)

import scipy.stats # for definition of gaussian distribution, so we can compare CPU to GPU time
norm_pdf = scipy.stats.norm
%timeit norm_pdf.pdf(x, loc=mean, scale=sigma)

%timeit gaussian_pdf(x, mean, sigma)

@vectorize
def cpu_gaussian_pdf(x, mean, sigma):
    '''Compute the value of a Gaussian probability density function at x with given mean and sigma.'''
    return math.exp(-0.5 * ((x - mean) / sigma)**2) / (sigma * SQRT_2PI)
    
%timeit cpu_gaussian_pdf(x, mean, sigma)
```

Tienes razón en algunos aspectos, pero permíteme aclarar y expandir un poco más sobre las diferencias entre `vectorize` y `njit` en Numba:

1. **`njit` (No Python JIT - Just-In-Time compilation)**:
    - `njit` es un decorador de Numba que compila funciones de Python en código máquina para mejorar su rendimiento.
    - Al usar `njit`, Numba elimina la sobrecarga del intérprete de Python, lo que permite que las funciones se ejecuten mucho más rápido.
    - `njit` no paraleliza automáticamente el código. Sin embargo, si el código es apto para la paralelización, puedes usar el argumento `parallel=True` con `@jit` para permitir que Numba paralelice ciertas operaciones automáticamente.
    - `njit` es generalmente adecuado para funciones que contienen bucles y cálculos numéricos intensivos.
2. **`vectorize`**:
    - `vectorize` es un decorador que te permite definir funciones que operan elemento a elemento sobre arrays de Numpy, similar a cómo funcionan las funciones universales (ufuncs) de Numpy.
    - Con `vectorize`, puedes especificar el tipo de datos de entrada y salida, y Numba generará una función que aplica tu función a cada elemento del array de manera eficiente.
    - Puedes usar el argumento `target='parallel'` para habilitar la paralelización automática en la operación elemento a elemento, lo que puede mejorar el rendimiento en sistemas con múltiples núcleos.
    - `vectorize` es ideal para operaciones que son naturalmente elemento a elemento y no dependen de cálculos entre elementos.

## Numba en la GPU para funciones NO elemento-a-elemento

Para ello vamos a utilizar numba.cuda.jit, no requiere utilizar el typing the los argumentos

```python
from numba import cuda

# Tiene en cuenta valores que son escalares, no vectores
# El utilizar el device=True indica que la funcion SOLO puede ser llamada por
# una funcion que se ejecuta en la GPU y no en el host (CPU)
@cuda.jit(device=True)
def polar_to_cartesian(rho, theta):
    x = rho * math.cos(theta)
    y = rho * math.sin(theta)
    return x, y

# Se vectoriza la funcion
@vectorize(['float32(float32, float32, float32, float32)'], target='cuda')
def polar_distance(rho1, theta1, rho2, theta2):
    x1, y1 = polar_to_cartesian(rho1, theta1) # We can use device functions inside our GPU ufuncs
    x2, y2 = polar_to_cartesian(rho2, theta2)
    
    return ((x1 - x2)**2 + (y1 - y2)**2)**0.5
    
n = 1000000
rho1 = np.random.uniform(0.5, 1.5, size=n).astype(np.float32)
theta1 = np.random.uniform(-np.pi, np.pi, size=n).astype(np.float32)
rho2 = np.random.uniform(0.5, 1.5, size=n).astype(np.float32)
theta2 = np.random.uniform(-np.pi, np.pi, size=n).astype(np.float32)

polar_distance(rho1, theta1, rho2, theta2)
```

## Funciones Python en la GPU

Operaciones soportadas por Numba en la GPU de Python:

- `if`/`elif`/`else`
- `while` and `for` loops
- Basic math operators
- Selected functions from the `math` and `cmath` modules
- Tuples

```python
# You should not modify this cell, it contains imports and initial values needed to do work on either
# the CPU or the GPU.

import numpy as np
from numba import cuda, vectorize

# Our hidden layer will contain 1M neurons.
# When you assess your work below, this value will be automatically set to 100M.
n = 1000000

greyscales = np.floor(np.random.uniform(0, 255, n).astype(np.float32))
weights = np.random.normal(.5, .1, n).astype(np.float32)

# As you will recall, `numpy.exp` works on the CPU, but, cannot be used in GPU implmentations.
# This import will work for the CPU-only boilerplate code provided below, but
# you will need to modify this import before your GPU implementation will work.
from math import exp

# Modify these 3 function calls to run on the GPU.
@vectorize(['float32(float32)'], target = 'cuda')
def normalize(grayscales):
    return grayscales / 255

@vectorize(['float32(float32, float32)'], target = 'cuda')
def weigh(values, weights):
    return values * weights
        
@vectorize(['float32(float32)'], target = 'cuda')
def activate(values):
    return ( exp(values) - exp(-values) ) / ( exp(values) + exp(-values) )
    
# Modify the body of this function to optimize data transfers and therefore speed up performance.
# As a constraint, even after you move work to the GPU, make this function return a host array.
def create_hidden_layer(n, greyscales, weights, exp, normalize, weigh, activate):
    
    # Esto lo que permite es pasar una variable que esta en la CPU, que se 
    # almacena en la memoria RAM a la GPU reservando memoria para dicha
    # variable. En el caso de que sea un array de NumPy obtendremos una variable
    # de la clase <class 'numba.cuda.cudadrv.devicearray.DeviceNDArray'>
    greyscales_device = cuda.to_device(greyscales)
    weights_device = cuda.to_device(weights)
    
    # cuda.device_array crea un array vacio en la memoria de la GPU sin
    # copiar datos desde la CPU, lo que se parece al malloc de C.
    # Esto lo hacemos para crear un buffer en la GPU para almacenar resultados
    # de los calculos sin inicializar los datos.
    normalized_out_device = cuda.device_array(shape=(n,), dtype=np.float32) 
    weighted_out_device = cuda.device_array(shape=(n,), dtype=np.float32)  
    activated_out_device = cuda.device_array(shape=(n,), dtype=np.float32) 
		
		# Ahora llamamos a cada una de las funciones con sus respectivas entradas
		# y las variables donde almacenaremos los resultados de las operaciones
    normalize(greyscales_device, out=normalized_out_device)
    weigh(normalized_out_device, weights_device, out=weighted_out_device)
    activate(weighted_out_device, out=activated_out_device)
    
    # The assessment mechanism will expect `activated` to be a host array, so,
    # even after you refactor this code to run on the GPU, make sure to explicitly copy
    # `activated` back to the host.
    return activated_out_device.copy_to_host()
    
# You probably don't need to edit this cell, unless you change the name of any of the values being passed as
# arguments to `create_hidden_layer` below.
arguments = {"n":n,
            "greyscales": greyscales,
            "weights": weights,
            "exp": exp,
            "normalize": normalize,
            "weigh": weigh,
            "activate": activate}

# Use this cell (and feel free to create others) to self-assess your function
a = create_hidden_layer(**arguments)
print(a) 
            
```

## Diferencias entre el vectorize y cuda.jit

### Vectorize

- Es mas a alto nivel y esta pensado para operaciones elemento-a-elemento
- Se usa cuando queremos definir una funcion que se aplica a cada elemento de un array de manera automatica.
- Numba maneja la paralelizacion por nosotros.
- Se puede usar tanto en CPU (`target='cpu'`) como en GPU (`target='cuda'`).

```jsx
from numba import vectorize
import numpy as np

@vectorize(['float32(float32, float32)'], target='cuda')
def suma_elementwise(x, y):
    return x + y

# Crear arrays en la CPU
a = np.array([1, 2, 3, 4, 5], dtype=np.float32)
b = np.array([10, 20, 30, 40, 50], dtype=np.float32)

# Se ejecuta en la GPU sin necesidad de lanzar hilos manualmente
c = suma_elementwise(a, b)
print(c)  # [11. 22. 33. 44. 55.]
```

## cuda.jit

- Es más **bajo nivel**, similar a la programación con CUDA en C.
- Tú controlas explícitamente los **bloques e hilos**.
- Es más flexible que `@vectorize`, permitiendo acceso a memoria compartida, sincronización de hilos, etc.
- Mayores posibilidades de paralelización y rendimiento

```jsx
from numba import cuda
import numpy as np

@cuda.jit
def suma_kernel(x, y, out):
    idx = cuda.grid(1)  # Obtiene el índice global del hilo
    if idx < x.size:  # Evita accesos fuera de rango
        out[idx] = x[idx] + y[idx]

# Crear arrays en la CPU
a = np.array([1, 2, 3, 4, 5], dtype=np.float32)
b = np.array([10, 20, 30, 40, 50], dtype=np.float32)
c = np.zeros_like(a)  # Array para almacenar el resultado

# Copiar datos a la GPU
d_a = cuda.to_device(a)
d_b = cuda.to_device(b)
d_c = cuda.device_array_like(c)  # Array vacío en la GPU

# Configurar el número de hilos y bloques
threads_per_block = 32
blocks_per_grid = (a.size + (threads_per_block - 1)) // threads_per_block

# Lanzar el kernel
suma_kernel[blocks_per_grid, threads_per_block](d_a, d_b, d_c)

# Copiar el resultado de vuelta a la CPU
c = d_c.copy_to_host()
print(c)  # [11. 22. 33. 44. 55.]
```

## Custom CUDA Kernels en Python con Numba

Tenemos hebras → un conjunto de hebras conforman un bloque de hebras → un conjunto de bloques de hebras crean una malla (grid).

Vamos a utilizar cuda.jit. Terner en cuenta que cuda.jit no devulve nada, se utiliza un arguumento out

```python
from numba import cuda

# Note the use of an `out` array. CUDA kernels written with `@cuda.jit` do not return values,
# just like their C counterparts. Also, no explicit type signature is required with @cuda.jit
@cuda.jit
def add_kernel(x, y, out):
    
    # The actual values of the following CUDA-provided variables for thread and block indices,
    # like function parameters, are not known until the kernel is launched.
    
    # This calculation gives a unique thread index within the entire grid (see the slides above for more)
    idx = cuda.grid(1)          # 1 = one dimensional thread grid, returns a single value.
                                # This Numba-provided convenience function is equivalent to
                                # `cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x`

    # This thread will do the work on the data element with the same index as its own
    # unique index within the grid.
    out[idx] = x[idx] + y[idx]
    
import numpy as np

n = 4096
x = np.arange(n).astype(np.int32) # [0...4095] on the host
y = np.ones_like(x)               # [1...1] on the host

d_x = cuda.to_device(x) # Copy of x on the device
d_y = cuda.to_device(y) # Copy of y on the device
d_out = cuda.device_array_like(d_x) # Like np.array_like, but for device arrays

# Because of how we wrote the kernel above, we need to have a 1 thread to one data element mapping,
# therefore we define the number of threads in the grid (128*32) to equal n (4096).
threads_per_block = 128
blocks_per_grid = 32

add_kernel[blocks_per_grid, threads_per_block](d_x, d_y, d_out)
**cuda.synchronize()**
print(d_out.copy_to_host()) # Should be [1...4096]
```

Podemos conocer las especificaciones de la GPU utilizada uutilizando el siguiente codigo:

```python
# Obtener el dispositivo GPU actual
gpu = cuda.get_current_device()

# Mostrar información relevante
print(f"Nombre de la GPU: {gpu.name}")
print(f"Número de multiprocesadores: {gpu.MULTIPROCESSOR_COUNT}")
print(f"Máximo de hilos por bloque: {gpu.MAX_THREADS_PER_BLOCK}")
print(f"Máximo de bloques en cada dimensión: {gpu.MAX_GRID_DIM_X}, {gpu.MAX_GRID_DIM_Y}, {gpu.MAX_GRID_DIM_Z}")
print(f"Máximo de hilos por dimensión de bloque: {gpu.MAX_BLOCK_DIM_X}, {gpu.MAX_BLOCK_DIM_Y}, {gpu.MAX_BLOCK_DIM_Z}")
print(f"Máximo de memoria compartida por bloque: {gpu.MAX_SHARED_MEMORY_PER_BLOCK} bytes")
```

- **`gpu.MULTIPROCESSOR_COUNT`** → Número de multiprocesadores en la GPU.
- **`gpu.MAX_THREADS_PER_BLOCK`** → Número máximo de hilos por bloque.
- **`gpu.MAX_GRID_DIM_X, Y, Z`** → Máximo de bloques que puedes tener en cada dimensión del grid.
- **`gpu.MAX_BLOCK_DIM_X, Y, Z`** → Máximo de hilos por bloque en cada dimensión (`x, y, z`).
- **`gpu.MAX_SHARED_MEMORY_PER_BLOCK`** → Cantidad de memoria compartida por bloque en bytes.

<aside>
💡

In short, the latency of operations can be hidden by SMs with other meaningful work so long as there is other work to be done. **give SMs the ability to hide latency by providing them with a sufficient number of warps which can be accomplished most simply by 
executing kernels with sufficiently large grid and block dimensions.**

</aside>

Deciding the very best size for the CUDA thread grid is a complex problem, and depends on both the algorithm and the specific GPU's [compute capability](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities), but here are some very rough heuristics that we tend to follow and which can work well for getting started:

- The size of a block should be a multiple of 32 threads (the size of a warp), with typical block sizes between 128 and 512 threads per block.
- The size of the grid should ensure the full GPU is utilized where
possible. Launching a grid where the number of blocks is 2x-4x the
number of SMs on the GPU is a good starting place. Something in the
range of 20 - 100 blocks is usually a good starting point.
- The CUDA kernel launch overhead does increase with the number of
blocks, so when the input size is very large we find it best not to
launch a grid where the number of threads equals the number of input
elements, which would result in a tremendous number of blocks. Instead
we use a pattern to which we will now turn our attention for dealing
with large inputs.

### Grid Stride Loop

Let's refactor the `add_kernel` above to utilize a grid 
stride loop so that we can launch it to work on larger data sets 
flexibly while incurring the benefits of global **memory coalescing**,
 which allows parallel threads to access memory in contiguous chunks, a 
scenario which the GPU can leverage to reduce the total number of memory
 operations:

```python
from numba import cuda

@cuda.jit
def add_kernel(x, y, out):
    

    start = cuda.grid(1)
    
    # This calculation gives the total number of threads in the entire grid
    stride = cuda.gridsize(1)   # 1 = one dimensional thread grid, returns a single value.
                                # This Numba-provided convenience function is equivalent to
                                # `cuda.blockDim.x * cuda.gridDim.x`

    # This thread will start work at the data element index equal to that of its own
    # unique index in the grid, and then, will stride the number of threads in the grid each
    # iteration so long as it has not stepped out of the data's bounds. In this way, each
    # thread may work on more than one data element, and together, all threads will work on
    # every data element.
    for i in range(start, x.shape[0], stride):
        # Assuming x and y inputs are same length
        out[i] = x[i] + y[i]
        
import numpy as np

n = 100000 # This is far more elements than threads in our grid
x = np.arange(n).astype(np.int32)
y = np.ones_like(x)

d_x = cuda.to_device(x)
d_y = cuda.to_device(y)
d_out = cuda.device_array_like(d_x)

threads_per_block = 128
blocks_per_grid = 30

add_kernel[blocks_per_grid, threads_per_block](d_x, d_y, d_out)
print(d_out.copy_to_host()) # Remember, memory copy carries implicit synchronization
```

## Operaciones atomicas

### 🔥 **Race Conditions en CUDA**

Una **race condition** ocurre cuando múltiples hilos acceden a la misma ubicación de memoria y, debido a la falta de sincronización, el resultado puede ser inesperado o incorrecto.

Dos tipos de race conditions comunes en CUDA:

1. **Read-after-write (RAW) hazard**
    - Un hilo intenta leer un valor que otro hilo podría estar modificando simultáneamente.
2. **Write-after-write (WAW) hazard**
    - Dos hilos intentan escribir en la misma dirección de memoria al mismo tiempo, pero solo un valor prevalece al final.

### 🚀 **Cómo Evitar Race Conditions**

1. **Cada hilo debe escribir en una ubicación de memoria única**.
    - Esto significa que cada hilo es responsable de una parte distinta de la salida.
2. **No usar el mismo array como entrada y salida** en la misma llamada al kernel.
    - Si necesitas hacerlo, usa **double buffering** (dos buffers, alternando entre entrada y salida en cada iteración).
3. **Usar operaciones atómicas cuando sea necesario**.
    - Una **operación atómica** asegura que solo un hilo a la vez pueda modificar una variable compartida.

### 🛠 **Ejemplo: Contador Global Usando Operaciones Atómicas**

Supongamos que queremos que cada hilo incremente un contador global en la memoria de la GPU. Si lo hacemos sin operaciones atómicas, tendremos una **race condition**.

### ❌ Código Incorrecto (Sin Operaciones Atómicas)

Este código podría dar resultados incorrectos porque varios hilos pueden leer el mismo valor del contador antes de que otro lo actualice.

```python
from numba import cuda
import numpy as np

@cuda.jit
def contador_global(counter):
    idx = cuda.grid(1)  # Índice global del hilo
    counter[0] += 1  # ❌ ¡Race condition aquí!

```

Aquí, varios hilos pueden leer `counter[0]` al mismo tiempo y escribir valores incorrectos.

---

### ✅ Código Correcto (Con Operaciones Atómicas)

Ahora usamos `cuda.atomic.add()` para asegurarnos de que la operación se realice de manera segura.

```python
from numba import cuda
import numpy as np

@cuda.jit
def contador_global_atomic(counter):
    idx = cuda.grid(1)  # Índice global del hilo
    cuda.atomic.add(counter, 0, 1)  # ✅ Operación atómica, sin race condition

# Inicializamos el contador en la GPU
contador = np.zeros(1, dtype=np.int32)
d_contador = cuda.to_device(contador)

# Configuramos 128 hilos en 4 bloques
threads_per_block = 128
blocks_per_grid = 4

# Llamamos al kernel
contador_global_atomic[blocks_per_grid, threads_per_block](d_contador)

# Traemos el resultado de la GPU a la CPU
resultado = d_contador.copy_to_host()
print("Valor final del contador:", resultado[0])  # Esperado: 128 * 4 = 512

```

🔹 Aquí `cuda.atomic.add(counter, 0, 1)` asegura que solo **un hilo a la vez** pueda modificar `counter[0]`, evitando la race condition.

## Effective use of the memory subsystem

Aqui vamos a utilizar la fusión de memoria (memory coalescing)

The performance of the uncoalesced data access pattern was far worse. Now you will learn why, and how to think about data access patterns in your kernels to obtain high performing kernels.

```python
@cuda.jit
def add_experiment(a, b, out, stride, coalesced):
    i = cuda.grid(1)
    # The above line is equivalent to
    # i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if coalesced == True:
        out[i] = a[i] + b[i]
    else:
        out[i] = a[stride*i] + b[stride*i]
```

Recordar que los bloques de hebras se dividen en warps de 32 hebras, y las instrucciones son pasadas en paralelo a nivel de warp a las 32 hebras. El suubsistema de memoria intentará minimizar el numero de lineas requeridos para completar la lectura o escritura requerida por el warp.

Cuanto más contiguos sean los datos que se asigna a cada hebra del warp mayor es la capacidad de uso de la memoria y menor perdida de rendimiento.

![image.png](Numba/image%201.png)

![image.png](Numba/image%202.png)

Pero conforme la memoria requerida se vuelve menor contigua, se requeriran más lineas para ser transferidas para satisfaces las necesidades de los warps y mas datos transferidos no serán usados (rojos)

![image.png](Numba/image%203.png)

```python
@cuda.jit
def col_sums(a, sums, ds):
    # TODO: Write this kernel to store the sum of each column in matrix `a` to the `sums` vector.
    idx = cuda.grid(1)
    sum = 0.0
    
    for i in range(n):
        # Each thread will sum a row of `a`
        sum += a[i][idx]
        
    sums[idx] = sum
    
n = 16384 # matrix side size
threads_per_block = 256
blocks = int(n / threads_per_block)

a = np.ones(n*n).reshape(n, n).astype(np.float32)
# Here we set an arbitrary column to an arbitrary value to facilitate a check for correctness below.
a[:, 3] = 9
sums = np.zeros(n).astype(np.float32)

d_a = cuda.to_device(a)
d_sums = cuda.to_device(sums)

%timeit col_sums[blocks, threads_per_block](d_a, d_sums, n); cuda.synchronize()
result = d_sums.copy_to_host()
truth = a.sum(axis=0)
```

### Matrices

```python
import numpy as np
from numba import cuda

A = np.zeros((4,4)) # A 4x4 Matrix of 0's
d_A = cuda.to_device(A)

# Here we create a 2D grid with 4 blocks in a 2x2 structure, each with 4 threads in a 2x2 structure
# by using a Python tuple to signify grid and block dimensions.
blocks = (2, 2)
threads_per_block = (2, 2)

@cuda.jit
def get_2D_indices(A):
    # By passing `2`, we get the thread's unique x and y coordinates in the 2D grid
    x, y = cuda.grid(2)
    # The above is equivalent to the following 2 lines of code:
    # x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    # y = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    
    # Write the x index followed by a decimal and the y index.
    A[x][y] = x + y / 10
    
get_2D_indices[blocks, threads_per_block](d_A)

result = d_A.copy_to_host()
result
```

```python
@cuda.jit
def matrix_add(a, b, out, coalesced):
    # TODO: set x and y to index correctly such that each thread
    # accesses one element in the data.
    x, y = cuda.grid(2)
    
    if coalesced == True:
        # TODO: write the sum of one element in `a` and `b` to `out`
        # using a coalesced memory access pattern.
        out[y][x]=a[y][x]+b[y][x] # Por columna
    else:
        # TODO: write the sum of one element in `a` and `b` to `out`
        # using an uncoalesced memory access pattern.
        out[x][y]=a[x][y]+b[x][y] # Por fila
```

## Memoria compartida

The device memory we have been utilizing thus far is called **global memory** which is available to any thread or block on the device, can persist for the lifetime of the application, and is a relatively large memory space.
We will now discuss how to utilize a region of on-chip device memory called **shared memory**. Shared memory is a programmer defined cache of limited size that [depends on the GPU](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities) being used and is **shared** between all threads in a block. It is a scarce resource, cannot be accessed by threads outside of the block where it was allocated, and does not persist after a kernel finishes executing. Shared memory however has a much higher bandwidth than global memory and can be used to great effect in many kernels, especially to optimize performance.

Here are a few common use cases for shared memory:

- Caching memory read from global memory that will need to be read multiple times within a block.
- Buffering output from threads so it can be coalesced before writing it back to global memory.
- Staging data for scatter/gather operations within a block.

```python
@cuda.jit
def swap_with_shared(vector, swapped):
    # Allocate a 4 element vector containing int32 values in shared memory.
    temp = cuda.shared.array(4, dtype=types.int32)
    
    idx = cuda.grid(1)
    
    # Move an element from global memory into shared memory
    temp[idx] = vector[idx]
    
    # cuda.syncthreads will force all threads in the block to synchronize here, which is necessary because...
    cuda.syncthreads()
    #...the following operation is reading an element written to shared memory by another thread.
    
    # Move an element from shared memory back into global memory
    swapped[idx] = temp[3 - cuda.threadIdx.x] # swap elements
    
vector = np.arange(4).astype(np.int32)
swapped = np.zeros_like(vector)

# Move host memory to device (global) memory
d_vector = cuda.to_device(vector)
d_swapped = cuda.to_device(swapped)

swap_with_shared[1, 4](d_vector, d_swapped)
```

```python
@cuda.jit
def tile_transpose(a, transposed):
    # `tile_transpose` assumes it is launched with a 32x32 block dimension,
    # and that `a` is a multiple of these dimensions.
    
    # 1) Create 32x32 shared memory array.
    tile = cuda.shared.array((32, 32), numba_types.float32)
    
    # Compute offsets into global input array. Recall for coalesced access we want to map threadIdx.x increments to
    # the fastest changing index in the data, i.e. the column in our array.
    a_col = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    a_row = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    # 2) Make coalesced read from global memory into shared memory array.
    # Note the use of local thread indices for the shared memory write,
    # and global offsets for global memory read.
    tile[cuda.threadIdx.y, cuda.threadIdx.x] = a[a_row, a_col]

    # 3) Wait for all threads in the block to finish updating shared memory.
    cuda.syncthreads()

    # 4) Calculate transposed location for the shared memory array tile
    # to be written back to global memory. Note that blockIdx.y*blockDim.y 
    # and blockIdx.x* blockDim.x are swapped (because we want to write to the
    # transpose locations), but we want to keep access coalesced, so match up the
    # threadIdx.x to the fastest changing index, i.e. the column
    t_col = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.x
    t_row = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.y

    # 5) Write from shared memory (using thread indices)
    # back to global memory (using grid indices)
    # transposing each element within the shared memory array.
    transposed[t_row, t_col] = tile[cuda.threadIdx.x, cuda.threadIdx.y]
```