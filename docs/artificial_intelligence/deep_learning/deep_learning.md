---
sidebar_position: 2
authors:
  - name: Daniel Bazo Correa
description: Fundamentos del Deep Learning.
title: Deep Learning
toc_max_heading_level: 3
---

## Bibliografía

- [Alice’s Adventures in a differentiable wonderland: A primer on designing neural networks (Volume I)](https://amzn.eu/d/3oYyuHg)
- [Deep Learning for Coders with Fastai and PyTorch: AI Applications Without a PhD](https://course.fast.ai/Resources/book.html)

## 1. Introducción al _Deep Learning_

Antes de abordar el campo del _Deep Learning_, es fundamental comprender un concepto
esencial que, aunque pueda parecer evidente, no siempre resulta fácil de definir: la
**inteligencia**. Esta puede entenderse como la capacidad de procesar información y
emplearla en la toma de decisiones futuras. A partir de esta noción surge la
**Inteligencia Artificial (IA)**, disciplina cuyo objetivo es desarrollar técnicas y
algoritmos que permitan a las máquinas emular ciertos comportamientos humanos. En
términos generales, la IA busca que los sistemas puedan utilizar la información
disponible para realizar predicciones, adaptarse a distintos contextos y resolver
problemas de manera autónoma.

Dentro de la IA se encuentra un subcampo crucial: el **aprendizaje automático o _Machine
Learning_**. Su propósito es permitir que un ordenador aprenda de la experiencia sin
necesidad de recibir instrucciones explícitas. En lugar de programar paso a paso cada
acción, se diseñan algoritmos capaces de identificar patrones en los datos, de modo que
el sistema mejore su rendimiento de forma automática a medida que acumula ejemplos. Este
enfoque representa un cambio significativo respecto a la programación tradicional, ya
que el sistema aprende a generalizar a partir de los datos, en lugar de ejecutar reglas
predefinidas.

Un nivel más especializado es el **_Deep Learning_**, que utiliza **redes neuronales
artificiales** para extraer patrones complejos a partir de datos sin procesar. Estas
redes, inspiradas en la estructura y funcionamiento del cerebro humano, aprenden
representaciones jerárquicas de la información, lo que les permite captar relaciones
complejas entre variables. Gracias a esta capacidad, el _Deep Learning_ resulta
especialmente eficaz en tareas como el reconocimiento de imágenes, el procesamiento del
lenguaje natural, el análisis de audio y otros problemas que involucran grandes
volúmenes de datos no estructurados.

### 1.1. Predicción de precios de viviendas mediante regresión lineal

Para ilustrar estos conceptos, consideremos un ejemplo sencillo: estimar el precio de
una vivienda. Si representamos gráficamente el tamaño de la casa frente a su precio, se
observa generalmente una tendencia positiva: a mayor tamaño, mayor precio. Un modelo
matemático básico para capturar esta relación es la **regresión lineal**, que ajusta una
recta a los datos. Sin embargo, esta solución presenta limitaciones: por ejemplo, una
línea puede asignar valores negativos a viviendas muy pequeñas, lo cual no tiene
sentido. Para corregirlo, se introducen funciones que restringen los resultados a un
rango de valores válidos.

Este proceso se puede entender mediante el funcionamiento de una **neurona o
perceptrón**. La neurona recibe como entrada el tamaño de la vivienda, realiza un
cálculo lineal a partir de ejemplos recopilados y aplica una función que descarta
valores inválidos, generando como salida una estimación coherente del precio. No
obstante, el valor de una vivienda depende de múltiples factores adicionales, como el
número de dormitorios, la ubicación o la calidad del vecindario. Incorporar varias
características complica el modelo, ya que se incrementan las dimensiones de los datos,
y la simple regresión lineal deja de ser suficiente. En estos casos, es necesario
combinar múltiples regresiones lineales organizadas en **capas**, formando arquitecturas
más complejas.

En una arquitectura de _Deep Learning_, se distingue una **capa de entrada**, que recibe
las características iniciales, una o varias **capas ocultas**, donde se combinan y
transforman dichas características, y una **capa de salida**, que genera la predicción
final.

### 1.2. Elementos esenciales de una neurona artificial

Cada neurona asigna un **peso** a cada característica, lo que refleja su importancia
relativa en el resultado frente a las demás variables. Además, incorpora un **sesgo**,
un valor adicional que permite ajustar la función de salida y proporciona mayor
flexibilidad al modelo, modulando la propensión de la neurona a activarse o desactivarse
según los datos de entrada. Tanto los pesos como el sesgo se inicializan de forma
aleatoria y se ajustan progresivamente durante el proceso de **entrenamiento**,
optimizando el rendimiento del modelo.

El resultado de cada neurona pasa posteriormente por una **función de activación no
lineal**, un componente crucial que permite a la red capturar relaciones complejas que
van más allá de las simples combinaciones lineales y definir un rango coherente para la
salida.

### 1.3. Arquitecturas de redes y tipos de datos

El _Deep Learning_ se adapta a distintos tipos de problemas mediante arquitecturas
especializadas, lo que permite extraer información más relevante de los datos y
comprender mejor los patrones subyacentes. Entre las principales arquitecturas se
encuentran:

- **Redes neuronales densas o totalmente conectadas**, adecuadas para datos tabulares.
- **Redes convolucionales (CNN, _Convolutional Neural Networks_)**, diseñadas para
  analizar imágenes y vídeos mediante la detección de patrones espaciales.
- **Redes recurrentes (RNN, _Recurrent Neural Networks_)** y sus variantes modernas,
  idóneas para procesar secuencias como texto, series temporales o audio.
- **Modelos multimodales**, capaces de integrar simultáneamente información de
  diferentes fuentes, como texto, imágenes y sonido.

Al analizar los datos, es importante distinguir entre:

- **Datos estructurados**, organizados en tablas con filas y columnas, típicos de bases
  de datos tradicionales. En estos casos, a menudo es suficiente aplicar algoritmos de
  aprendizaje automático más simples en lugar de recurrir a _Deep Learning_.
- **Datos no estructurados**, como imágenes, grabaciones de voz o documentos de texto
  libre, que requieren arquitecturas más avanzadas para su procesamiento. El _Deep
  Learning_ sobresale en estos contextos debido a su capacidad para interpretar y
  extraer patrones complejos de grandes volúmenes de información no estructurada.

### 1.4. Factores que impulsan el desarrollo del Deep Learning

El auge del _Deep Learning_ en la última década se explica por la confluencia de tres
factores principales. En primer lugar, la **disponibilidad masiva de datos**, favorecida
por la digitalización y la conectividad global, proporciona la materia prima necesaria
para entrenar modelos complejos. En segundo lugar, los **avances en hardware
especializado**, como GPUs y TPUs, permiten entrenar modelos de gran escala en tiempos
razonables. Empresas como NVIDIA han desarrollado GPUs optimizadas para el cálculo
matricial requerido en el aprendizaje profundo, complementadas con librerías como CUDA.
Además, se observa una tendencia hacia arquitecturas diseñadas específicamente para
inteligencia artificial, como NPU y TPUs, integradas en dispositivos móviles y
embebidos, que permiten ejecutar modelos de manera eficiente, privada y sin conexión a
Internet.

El tercer factor son las **mejoras en algoritmos y técnicas de optimización**, que han
permitido abordar problemas antes inabordables. La combinación de estos factores ha
democratizado el uso del _Deep Learning_, promoviendo la aparición de startups que
liberan modelos de código abierto, parámetros de entrenamiento e incluso los datos
utilizados, facilitando así la investigación y el desarrollo de nuevas aplicaciones
basadas en inteligencia artificial.

## 2. Regresión Lineal y Regresión Logística

El entrenamiento de una neurona o de una red neuronal se fundamenta en dos procesos
esenciales: la **propagación hacia adelante (_forward propagation_)** y la **propagación
hacia atrás (_backpropagation_)**.

La propagación hacia adelante consiste en calcular la predicción del modelo a partir de
los datos de entrada. En este proceso, los datos ingresan por la capa de entrada y
atraviesan las distintas capas de la red, generando una representación que permite al
modelo estimar la salida.

Por su parte, la propagación hacia atrás se encarga de ajustar los parámetros internos
del modelo (pesos y sesgos) con el objetivo de minimizar el error de predicción. Durante
este proceso, los gradientes fluyen desde la salida hacia la entrada, permitiendo la
actualización progresiva de los parámetros en cada iteración para mejorar la precisión
del modelo.

Con estos mecanismos en mente, es útil analizar un caso clásico de aprendizaje
automático: la **clasificación binaria**, que consiste en asignar a cada ejemplo una de
dos posibles clases.

### 2.1. Detección de gatos en imágenes mediante regresión logística

Un ejemplo representativo de clasificación binaria se encuentra en la detección de
objetos en imágenes, como identificar si una imagen contiene un gato. En este escenario,
cada ejemplo se etiqueta de manera binaria, **1** si la imagen contiene un gato y **0**
si no. Aunque este problema puede ampliarse a clasificación multiclase, para fines
ilustrativos se considera únicamente la clasificación binaria.

Cada imagen se representa con dimensiones de 64×64 píxeles y formato RGB, generando tres
matrices que corresponden a los canales de color rojo, verde y azul. Cada matriz tiene
dimensiones 64×64, lo que da un total de valores por imagen de:

$$
64 \times 64 \times 3 = 12288
$$

Para introducir esta información en un modelo de red neuronal, se aplica la técnica de
**aplanamiento (_flatten_)**, que transforma las tres matrices en un único vector
columna de dimensión $12288 × 1$, manteniendo toda la información relevante de los
píxeles.

Las etiquetas indican al modelo la clase correspondiente de cada ejemplo. Por ejemplo,
una imagen denominada `gato.png` recibe la etiqueta **1**, mientras que otra que no
contenga un gato recibe **0**. Dado que cada imagen está acompañada de su etiqueta, este
escenario corresponde a lo que se denomina como **aprendizaje supervisado**.

Si se dispone de $M$ ejemplos, la matriz de características `X` tendrá dimensión
`(n, M)`, donde $n = 12288$, y el vector de etiquetas `Y` tendrá dimensión `(1, M)`,
conteniendo únicamente valores binarios.

Para abordar este problema se utiliza la **regresión logística**, un algoritmo
supervisado diseñado específicamente para tareas con etiquetas binarias (ceros y unos).
Su funcionamiento es similar al de la regresión lineal, con la diferencia clave de que
la salida se transforma mediante la **función sigmoide**, que restringe el resultado a
un valor entre 0 y 1, interpretable como probabilidad y definida como

$$
\sigma(z) = \frac{1}{1 + e^{-z}},
$$

donde

$$
z = w^T x + b.
$$

En esta ecuación, $w$ representa los **pesos**, $b$ el **sesgo** y $x$ el vector de
características de entrada. La predicción final del modelo se expresa como

$$
\hat{y} = \sigma(w^T x + b),
$$

donde $\hat{y}$ corresponde a la probabilidad de que la imagen pertenezca a la clase
positiva, es decir, que efectivamente contenga un gato. Esta representación permite
interpretar las salidas del modelo de forma probabilística y establecer umbrales para la
clasificación binaria de manera consistente y flexible.

### 2.2. Función de pérdida y función de coste

El objetivo del modelo es ajustar los parámetros $w$ y $b$ de manera que las
predicciones $\hat{y}$ se aproximen lo más posible a los valores reales $y$. Para
evaluar y guiar este ajuste se utilizan dos métricas fundamentales:

- **Función de pérdida**, que cuantifica el error en un único ejemplo individual.
- **Función de coste**, que representa el promedio de las pérdidas de todos los ejemplos
  del conjunto de entrenamiento.

En regresión logística, la función de pérdida utilizada es la **función de pérdida
logística o log-loss**, definida como

$$
\mathcal{L}(\hat{y}, y) = - \big( y \cdot \log(\hat{y}) + (1-y)\cdot \log(1-\hat{y}) \big).
$$

Esta función penaliza de manera más adecuada los errores en problemas de clasificación
binaria en comparación con el error cuadrático medio, que se define como

$$
\text{MSE} = \frac{1}{M} \sum_{i=1}^{M} (\hat{y}^{(i)} - y^{(i)})^2.
$$

La **función de coste** asociada a la regresión logística se obtiene como el promedio de
las pérdidas de todos los ejemplos, representada como

$$
J(w, b) = \frac{1}{M} \sum_{i=1}^{M} \mathcal{L}(\hat{y}^{(i)}, y^{(i)}).
$$

Esta formulación basada en log-loss evita problemas de múltiples mínimos locales y
garantiza una optimización más estable y eficiente en tareas de clasificación binaria,
proporcionando gradientes más consistentes durante el entrenamiento.

### 2.3. Descenso del gradiente

El entrenamiento de un modelo de regresión logística tiene como objetivo encontrar los
valores de $w$ y $b$ que minimicen la función de coste $J(w, b)$. Para ello se emplea el
descenso del gradiente, un algoritmo iterativo que ajusta los parámetros en la dirección
que produce la mayor reducción del error.

Si recordamos, la función de coste en regresión logística se define como

$$
J(w, b) = \frac{1}{M} \sum_{i=1}^{M} \mathcal{L}(\hat{y}^{(i)}, y^{(i)})
= -\frac{1}{M} \sum_{i=1}^{M} \Big[ y^{(i)} \log(\hat{y}^{(i)}) + (1-y^{(i)}) \log(1-\hat{y}^{(i)}) \Big],
$$

donde $M$ es el número total de ejemplos, $\hat{y}^{(i)} = \sigma(w^T x^{(i)} + b)$ es
la predicción para el ejemplo $i$, $x^{(i)}$ es el vector de características, $y^{(i)}$
es la etiqueta real y $\sigma(z)$ es la función sigmoide.

Para minimizar $J(w, b)$ se calculan las derivadas parciales respecto a cada parámetro,
lo que permite estimar la pendiente de la función de coste en un punto dado. Estas
derivadas se expresan como

$$
\frac{\partial J}{\partial w} = dw = \frac{1}{M} \sum_{i=1}^{M} (\hat{y}^{(i)} - y^{(i)}) x^{(i)},
$$

$$
\frac{\partial J}{\partial b} = db = \frac{1}{M} \sum_{i=1}^{M} (\hat{y}^{(i)} - y^{(i)}),
$$

y representan la dirección en la que deben modificarse los parámetros $w$ y $b$ para
reducir el error.

El procedimiento iterativo comienza con la inicialización de los parámetros con valores
pequeños, ya sea ceros o aleatorios. A continuación, se realiza la propagación hacia
adelante para calcular las predicciones $\hat{y}$ a partir de los datos de entrada $X$ y
se evalúa la función de pérdida $\mathcal{L}(\hat{y}, y)$ y la función de coste
$J(w, b)$ sobre el conjunto de entrenamiento. Posteriormente se aplica la propagación
hacia atrás para calcular las derivadas parciales $dw$ y $db$, que se utilizan para
actualizar los parámetros mediante la regla

$$
w := w - \alpha \cdot dw,
$$

$$
\quad b := b - \alpha \cdot db,
$$

donde $\alpha$ es la tasa de aprendizaje que regula el tamaño del paso en cada
iteración.

Cada actualización mueve los parámetros en la dirección opuesta al gradiente, asegurando
la reducción del valor de la función de coste. Este proceso se repite de manera
iterativa hasta que el modelo converge a un mínimo adecuado de $J(w, b)$, momento en el
cual las actualizaciones se vuelven insignificantes y las predicciones alcanzan la
precisión deseada.

En la práctica, estos cálculos se implementan mediante vectorización, utilizando
operaciones matriciales que permiten procesar todos los ejemplos simultáneamente. La
vectorización simplifica la implementación, reduce el tiempo de entrenamiento y
aprovecha de manera eficiente la capacidad de cálculo de las GPUs, lo que resulta
crucial en aplicaciones de Deep Learning con grandes volúmenes de datos.

### 2.4. Implementación de la regresión logística en Python

A continuación se muestra una implementación básica de regresión logística empleando
Python y la librería NumPy. Este ejemplo abarca desde la inicialización de los
parámetros hasta su actualización mediante descenso del gradiente y la generación de
predicciones finales, ofreciendo una referencia práctica para comprender el
funcionamiento y la implementación de la regresión logística en problemas de
clasificación binaria.

```python
import numpy as np
import matplotlib.pyplot as plt

# 1. Dataset de ejemplo
np.random.seed(1)
m = 200  # número de ejemplos
n = 2    # número de características

# Clase 0
X0 = np.random.randn(m//2, n) + np.array([-2, -2])
Y0 = np.zeros((m//2, 1))

# Clase 1
X1 = np.random.randn(m//2, n) + np.array([2, 2])
Y1 = np.ones((m//2, 1))

# Concatenar y transponer
X = np.vstack((X0, X1)).T
Y = np.vstack((Y0, Y1)).T

# 2. Funciones auxiliares
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def initialize_params(n):
    w = np.zeros((n, 1))
    b = 0
    return w, b

def forward_propagation(w, b, X, Y):
    m = X.shape[1]
    Z = np.dot(w.T, X) + b
    A = sigmoid(Z)
    cost = -(1/m) * np.sum(Y*np.log(A) + (1-Y)*np.log(1-A))
    return A, cost

def backward_propagation(A, X, Y):
    m = X.shape[1]
    dw = (1/m) * np.dot(X, (A - Y).T)
    db = (1/m) * np.sum(A - Y)
    return dw, db

def update_params(w, b, dw, db, learning_rate):
    w -= learning_rate * dw
    b -= learning_rate * db
    return w, b

# 3. Entrenamiento
def logistic_regression(X, Y, num_iterations=1000, learning_rate=0.1, print_cost=False):
    n = X.shape[0]
    w, b = initialize_params(n)
    costs = []

    for i in range(num_iterations):
        A, cost = forward_propagation(w, b, X, Y)
        dw, db = backward_propagation(A, X, Y)
        w, b = update_params(w, b, dw, db, learning_rate)

        if i % 100 == 0:
            costs.append(cost)
            if print_cost:
                print(f"Iteración {i}: coste = {cost:.4f}")

    return w, b, costs

# 4. Predicción
def predict(w, b, X):
    A = sigmoid(np.dot(w.T, X) + b)
    return (A > 0.5).astype(int)

# 5. Entrenar y evaluar
w, b, costs = logistic_regression(X, Y, num_iterations=1000, learning_rate=0.1, print_cost=True)
Y_pred = predict(w, b, X)
accuracy = 100 - np.mean(np.abs(Y_pred - Y)) * 100
print(f"\nExactitud del modelo: {accuracy:.2f}%")

# 6. Visualización
plt.plot(costs)
plt.xlabel("Iteraciones (x100)")
plt.ylabel("Coste")
plt.title("Reducción del coste durante el entrenamiento")
plt.show()
```

## 3. Redes Neuronales y funciones de activación

### 3.1. Generalización y sobreajuste

Un modelo que presenta un **coste bajo en el conjunto de entrenamiento** no
necesariamente constituye un buen modelo. Esta situación puede indicar la presencia de
**sobreajuste (_overfitting_)**, un fenómeno que ocurre cuando la precisión obtenida en
los datos de entrenamiento es significativamente mayor que en los conjuntos de
validación o prueba. En estos casos, el modelo no aprende patrones generales de los
datos, sino que **memoriza ejemplos específicos**, lo que reduce su capacidad de
**generalizar** a datos nuevos y limita su utilidad práctica.

El sobreajuste suele manifestarse cuando existen pocas muestras disponibles para
entrenar, cuando se emplean arquitecturas excesivamente complejas, o cuando los datos
presentan problemas de representación, tales como etiquetado incorrecto, predominancia
de ciertas clases sobre otras (desequilibrio de clases) o sesgos en el conjunto de
datos. Además, las variaciones en las distribuciones de los datos entre el entrenamiento
y el uso en producción pueden afectar la capacidad del modelo para generalizar
correctamente.

### 3.2. De neuronas a redes neuronales

Una **neurona artificial** se puede representar de manera similar a una regresión
logística: recibe entradas, las combina linealmente mediante pesos y sesgo, y aplica una
función de activación para producir una salida. No obstante, la regresión lineal
presenta limitaciones al modelar relaciones complejas en los datos. Para superar estas
limitaciones, es necesario aumentar la capacidad de representación de las neuronas.

Una **red neuronal** se construye al **apilar múltiples neuronas organizadas en capas**,
interconectadas entre sí, de manera que la información procesada por una neurona puede
transmitirse a otras neuronas de la misma capa o de capas posteriores. Este mecanismo
permite que cada neurona transfiera la representación que ha generado de los datos de
entrada a las neuronas siguientes. Las arquitecturas de redes neuronales incluyen
diferentes tipos de capas:

- La **capa de entrada** recibe las características iniciales de los datos.
- La **capa de salida** produce la predicción final.
- Las **capas ocultas**, situadas entre la capa de entrada y la de salida, transforman
  progresivamente la información. Se denominan "ocultas" porque sus valores no se
  observan directamente, sino que únicamente se percibe su efecto en la salida final.

El cálculo de la salida de una red neuronal consiste en aplicar repetidamente la
operación de combinación lineal seguida de activación. La complejidad del aprendizaje
profundo aumenta con el número de capas y conexiones, incrementando la capacidad de
representación del modelo, pero también dificultando su interpretación.

### 3.3. Funciones de activación

Las **funciones de activación** introducen no linealidad en la red neuronal, permitiendo
que el modelo aprenda relaciones complejas entre los datos. Sin funciones de activación,
una red neuronal se reduce a una combinación lineal de las entradas, comportándose de
manera similar a métodos clásicos no basados en redes neuronales. La elección de la
función de activación es fundamental y depende del tipo de capa y del problema a
resolver.

En las **capas ocultas**, se emplean funciones de activación como:

- **ReLU (Rectified Linear Unit)**: Es ampliamente utilizada en redes profundas, ya que
  acelera el entrenamiento y evita problemas de gradientes muy pequeños. No obstante,
  puede provocar **neuronas muertas**, que siempre devuelven cero. Para mitigar este
  efecto se utilizan variantes como _Leaky ReLU_, que mantiene un pequeño gradiente para
  valores negativos. Se representa como

      $$
      f(x) = \max(0, x).
      $$

- **Sigmoide**: Transforma los valores en el rango $[0,1]$. Se utiliza en redes
  recurrentes, aunque presenta el problema de **gradientes que desaparecen** en los
  extremos. Se representa como

      $$
      \sigma(x) = \frac{1}{1 + e^{-x}}
      $$

- **Tangente hiperbólica (tanh)**: Normaliza las salidas en el rango $[-1, 1]$. Suele
  preferirse frente a la sigmoide en capas ocultas porque sus activaciones tienen media
  cercana a cero, lo que facilita el entrenamiento. Tanto la sigmoide como la tangente
  hiperbólica tienden a saturarse en valores extremos, provocando gradientes muy
  pequeños que ralentizan el proceso de aprendizaje.

    $$
    \tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
    $$

En las **capas de salida**, la función de activación se selecciona según el rango de
valores esperado en la salida:

- **Clasificación binaria:** sigmoide.
- **Clasificación multiclase (mutuamente excluyentes):** _softmax_, que generaliza la
  sigmoide para más de dos clases.
- **Clasificación multietiqueta:** sigmoide, ya que una muestra puede pertenecer
  simultáneamente a varias clases.
- **Regresión:** activación lineal, permitiendo que la salida adopte cualquier valor
  real.

### 3.4. Implementación de una red neuronal con ReLU y sigmoide

El siguiente ejemplo implementa una red neuronal de dos capas para un conjunto de datos
sintético. Este código ilustra de manera práctica cómo construir, entrenar y evaluar una
red neuronal simple utilizando **ReLU** en la capa oculta y **sigmoide** en la capa de
salida para un problema de clasificación binaria. La red aprende a identificar la
relación entre las características de entrada y la clase de salida, mostrando cómo la
combinación de forward y backward propagation permite ajustar los parámetros mediante
optimización basada en gradientes.

```python
import numpy as np
import matplotlib.pyplot as plt

# --
# 1. Crear dataset sintético
# --
np.random.seed(0)
m = 200  # número de ejemplos
X = np.random.randn(2, m)  # 2 características
Y = (X[0, :] * X[1, :] > 0).astype(int).reshape(1, m)
# Clase = 1 si x1 y x2 tienen el mismo signo, si no 0

# --
# 2. Funciones auxiliares
# --
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

def compute_loss(Y, A):
    m = Y.shape[1]
    return -(1/m) * np.sum(Y*np.log(A+1e-8) + (1-Y)*np.log(1-A+1e-8))

# --
# 3. Inicializar parámetros
# --
def initialize_parameters(n_x, n_h, n_y):
    np.random.seed(1)
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}

# --
# 4. Forward propagation
# --
def forward_propagation(X, params):
    W1, b1, W2, b2 = params["W1"], params["b1"], params["W2"], params["b2"]
    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)
    cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
    return A2, cache

# --
# 5. Backpropagation
# --
def backward_propagation(X, Y, params, cache):
    m = X.shape[1]
    W2 = params["W2"]
    A1, A2, Z1 = cache["A1"], cache["A2"], cache["Z1"]

    dZ2 = A2 - Y
    dW2 = (1/m) * np.dot(dZ2, A1.T)
    db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = np.dot(W2.T, dZ2)
    dZ1 = dA1 * relu_derivative(Z1)
    dW1 = (1/m) * np.dot(dZ1, X.T)
    db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)

    return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

# --
# 6. Actualizar parámetros
# --
def update_parameters(params, grads, lr):
    params["W1"] -= lr * grads["dW1"]
    params["b1"] -= lr * grads["db1"]
    params["W2"] -= lr * grads["dW2"]
    params["b2"] -= lr * grads["db2"]
    return params

# --
# 7. Entrenamiento
# --
def model(X, Y, n_h=3, num_iterations=10000, lr=0.1, print_loss=True):
    n_x, n_y = X.shape[0], Y.shape[0]
    params = initialize_parameters(n_x, n_h, n_y)

    for i in range(num_iterations):
        A2, cache = forward_propagation(X, params)
        loss = compute_loss(Y, A2)
        grads = backward_propagation(X, Y, params, cache)
        params = update_parameters(params, grads, lr)

        if print_loss and i % 1000 == 0:
            print(f"Iteración {i}, pérdida: {loss:.4f}")

    return params

# --
# 8. Predicciones
# --
def predict(X, params):
    A2, _ = forward_propagation(X, params)
    return (A2 > 0.5).astype(int)

# --
# 9. Ejecutar el modelo
# --
params = model(X, Y, n_h=3, num_iterations=10000, lr=0.1)
Y_pred = predict(X, params)
acc = np.mean(Y_pred == Y) * 100
print(f"Precisión final: {acc:.2f}%")
```

TODO: NECESITO CORREGIR A PARTIR DE AQUI

## 4. Introducción a las Redes Neuronales Profundas (Deep Neural Networks)

## Propagación hacia delante

La **propagación hacia delante** es el proceso mediante el cual una red neuronal
transforma las entradas en salidas. En cada capa $L$, la salida se obtiene al
multiplicar los **pesos** de esa capa por las activaciones de la capa anterior $(L-1)$,
sumando el **sesgo** y aplicando posteriormente una **función de activación** que
introduce no linealidad.

En cuanto a las dimensiones:

- La **matriz de pesos** de la capa $L$ tiene dimensiones $(N_L, N_{L-1})$, donde $N_L$
  es el número de neuronas de la capa actual y $N_{L-1}$ el de la capa anterior.
- El **sesgo** de la capa $L$ tiene dimensiones $(N_L, 1)$.

Cuando se implementa la **propagación hacia atrás**, las derivadas de pesos y sesgos
mantienen estas mismas dimensiones.

## Redes profundas: motivación

La potencia de las **redes neuronales profundas** radica en su capacidad de construir
representaciones jerárquicas:

- Las primeras capas detectan características básicas (por ejemplo, bordes en imágenes).
- Capas intermedias combinan estas características para reconocer formas más complejas
  (ojos, narices).
- Capas más profundas identifican objetos completos (rostros).

Cuantas más capas posee la red, mayor es su capacidad para aprender patrones abstractos
y complejos.

## Parámetros y Hiperparámetros

- **Parámetros**: los **pesos** y **sesgos**, aprendidos automáticamente durante el
  entrenamiento.
- **Hiperparámetros**: definidos antes de entrenar, como la **tasa de aprendizaje**, el
  número de iteraciones, el número de capas ocultas, la cantidad de neuronas por capa o
  la elección de funciones de activación.

La búsqueda de hiperparámetros es un **proceso iterativo de prueba y error**, en el que
se ajustan los valores hasta encontrar la mejor configuración.

## Mejoras en redes neuronales profundas

El rendimiento de una red depende en gran medida del **ajuste de hiperparámetros**, de
las **técnicas de regularización** y de los **métodos de optimización**. El flujo de
trabajo suele seguir un ciclo: **idea → implementación → experimento → refinamiento**.

### División de conjuntos de datos

El entrenamiento de un modelo de Deep Learning requiere una adecuada gestión de los
datos. Para ello, se dividen en subconjuntos con propósitos específicos:

- **Conjunto de entrenamiento**: utilizado para ajustar los parámetros internos del
  modelo (pesos y sesgos).
- **Conjunto de validación**: compuesto por ejemplos no vistos durante el entrenamiento,
  sirve para comprobar que el modelo generaliza correctamente y no se limita a memorizar
  los datos (_sobreajuste_).
- **Conjunto de prueba (test)**: opcional pero recomendable, se reserva para una
  evaluación final y objetiva del modelo una vez entrenado.

- **Entrenamiento**: ajuste de parámetros.
- **Validación (o desarrollo)**: comparación entre modelos para seleccionar el mejor.
- **Prueba (test)**: evaluación final del rendimiento.

Distribución habitual:

- Con pocos datos: 70% entrenamiento, 30% prueba.
- Con muchos datos: 60% entrenamiento, 20% validación, 20% prueba.

Es esencial que validación y prueba provengan de la **misma distribución**.

### Sesgo y varianza

- **Sesgo alto (subajuste)**: el modelo no logra aprender la relación subyacente.
- **Varianza alta (sobreajuste)**: el modelo se adapta demasiado a los datos de
  entrenamiento y no generaliza.

Estrategias:

- Reducir sesgo → redes más grandes, entrenar más tiempo, probar arquitecturas
  alternativas.
- Reducir varianza → más datos, aplicar regularización (ej. L2, _dropout_), modificar la
  arquitectura.

## Regularización

La regularización ayuda a evitar el sobreajuste:

- **L2 (ridge)**: penaliza pesos grandes, estabilizando el aprendizaje.
- **L1 (lasso)**: favorece soluciones con muchos pesos en cero (modelos más simples).
- **Dropout**: apaga aleatoriamente neuronas durante el entrenamiento, generando
  representaciones más robustas.
- **Aumentación de datos**: crear ejemplos artificiales (rotaciones, traslaciones,
  cambios de iluminación).
- **Detención temprana (early stopping)**: detener el entrenamiento cuando el error en
  validación deja de mejorar.
- **Normalización de entradas**: escalar las características para acelerar la
  convergencia.

## Desvanecimiento y explosión de gradientes

En redes profundas, los gradientes pueden **disminuir o crecer exponencialmente**,
dificultando el entrenamiento.

Medidas de mitigación:

- **Inicialización adecuada de pesos** (Xavier, He).
- **Normalización de datos de entrada** (media 0, varianza 1).
- Uso de funciones de activación más estables (ReLU y variantes).

## Optimización

El entrenamiento con grandes volúmenes de datos requiere algoritmos de optimización
eficientes.

- **Descenso de gradiente con minilotes**: divide los datos en lotes de tamaño
  intermedio (ej. 32, 64, 128).

  - Si el lote es de un solo ejemplo, se denomina **descenso estocástico**.

### Principales variantes

- **Momentum**: suaviza el gradiente acumulando información pasada.
- **RMSprop**: ajusta la tasa de aprendizaje de manera adaptativa.
- **Adam**: combina Momentum y RMSprop, siendo uno de los más usados.
- **Decaimiento de la tasa de aprendizaje**: reducir progresivamente la tasa mejora la
  convergencia final.

## Normalización en redes neuronales

- **Batch Normalization**: normaliza activaciones dentro de cada minilote, acelerando el
  entrenamiento y estabilizando el ajuste de hiperparámetros.
- **Layer Normalization**: alternativa más usada en arquitecturas secuenciales como
  transformadores.

## Estrategia en aprendizaje automático

No todas las mejoras producen el mismo impacto. A veces añadir más datos o cambiar la
arquitectura apenas mejora el rendimiento. Por ello es clave definir **métricas claras**
que guíen las decisiones.

### Principales métricas

- **Precisión (precision)**: proporción de verdaderos positivos sobre los predichos
  positivos.
- **Recall**: proporción de verdaderos positivos sobre todos los positivos reales.
- **F1-score**: media armónica entre precisión y recall.

Se pueden combinar métricas de **optimización** (precisión) con métricas de
**satisfacción** (tiempo de ejecución, consumo de memoria).

## Comparación con el rendimiento humano

El desempeño de los modelos suele compararse con el nivel humano, lo que establece un
**techo de referencia**.

- **Sesgo evitable**: diferencia entre el error humano y el del modelo.
- **Varianza**: diferencia entre el error en entrenamiento y validación.

Reducir sesgo → modelos más grandes, mejores algoritmos, más entrenamiento. Reducir
varianza → más datos, regularización, ajuste de hiperparámetros.

## Aprendizaje por transferencia y multitarea

- **Transferencia de conocimiento**: reutilizar pesos de un modelo previamente entrenado
  en otra tarea.

  - Con pocos datos → ajustar solo las últimas capas.
  - Con muchos datos → reajustar toda la red (fine-tuning).

- **Aprendizaje multitarea**: una sola red resuelve múltiples tareas compartiendo
  representaciones internas.

  - Ejemplo: un mismo modelo para detección de peatones, señales, carriles y
    planificación de trayectorias en conducción autónoma.

## 5. Arquitecturas de Deep Learning

## 1. Redes Neuronales Convolucionales y la Visión Computacional

La visión computacional constituye uno de los campos más dinámicos y transformadores de
la inteligencia artificial. Gracias a ella se han desarrollado aplicaciones que van
desde la conducción autónoma hasta el reconocimiento facial o la clasificación
automática de imágenes. Incluso sus fundamentos han inspirado avances en dominios
aparentemente distintos, como el procesamiento del lenguaje y el reconocimiento de voz.

El principal desafío al trabajar con imágenes radica en la enorme cantidad de datos que
contienen. Una simple fotografía de 64 × 64 píxeles en color, con tres canales RGB,
representa un vector de **12.288 valores**. Alimentar directamente este volumen de datos
en una red neuronal tradicional implicaría crear capas iniciales con decenas de miles de
neuronas, lo que resulta inviable a medida que las resoluciones crecen, tanto por coste
computacional como por riesgo de sobreajuste.

La solución se encuentra en la **convolución**. Esta operación aplica pequeños filtros
(_kernels_) sobre la imagen para detectar patrones locales como bordes, esquinas o
texturas. Al desplazarse por la imagen, cada filtro genera un **mapa de
características** que refleja el grado de coincidencia con las distintas regiones.
Ejemplos clásicos son los filtros de detección de bordes verticales u horizontales, así
como los operadores Sobel o Scharr. Sin embargo, la gran ventaja de las redes
convolucionales es que estos filtros no se fijan manualmente: sus valores se aprenden
mediante retropropagación, lo que permite descubrir patrones mucho más complejos y
específicos para cada tarea.

### Conceptos clave en la convolución

El uso de convoluciones introduce varios elementos fundamentales:

- **Relleno (padding):** añade bordes artificiales para evitar la pérdida de información
  en los márgenes y mantener el tamaño de la entrada.
- **Desplazamiento (stride):** determina cuántos píxeles avanza el filtro en cada paso.
  Un stride mayor reduce el tamaño de la salida y, por ende, el número de cálculos.
- **Filtros en tres dimensiones:** en imágenes a color, los filtros no son simples
  matrices, sino cubos que recorren simultáneamente los tres canales de la imagen.

Lo notable es que el número de parámetros de una capa convolucional depende del tamaño y
número de filtros, y no del tamaño de la imagen. Por ejemplo, 10 filtros de 3 × 3 × 3
suman únicamente 280 parámetros, cifra muy reducida frente a los millones de conexiones
de una red totalmente conectada.

### Capas de agrupamiento

Tras la convolución suele aplicarse una etapa de **agrupamiento (_pooling_)**, que
reduce las dimensiones intermedias y aporta robustez frente a pequeñas variaciones. La
técnica más habitual es el **max pooling**, que selecciona el valor máximo en cada
región, priorizando la presencia de una característica por encima de su posición exacta.
También existe el **average pooling**, que utiliza el valor medio, aunque se emplea con
menor frecuencia.

### Flujo en redes convolucionales

A medida que se avanza en la red, el tamaño espacial de las representaciones disminuye y
el número de canales aumenta, lo que permite capturar patrones cada vez más abstractos.
Finalmente, se añaden capas totalmente conectadas que integran la información extraída
para generar la predicción final: clasificar, reconocer o identificar un objeto.

### Ventajas de las convoluciones

Las convoluciones resultan efectivas por dos motivos principales:

1. **Reducción drástica de parámetros**, lo que simplifica el entrenamiento.
2. **Compartición de parámetros**, ya que un patrón aprendido en una región puede
   aplicarse en cualquier otra, favoreciendo la generalización.

Gracias a estas propiedades, las redes convolucionales han revolucionado la manera en
que las máquinas interpretan imágenes, alcanzando una capacidad de análisis visual que
en algunos aspectos rivaliza con la percepción humana.

## 2. Redes Neuronales Residuales y Nuevas Arquitecturas

El aumento en profundidad de las redes trae consigo un problema: a partir de cierto
punto, en lugar de mejorar, su rendimiento se degrada. Esto ocurre por fenómenos de
**gradientes que desaparecen o explotan**, lo que impide que la red aprenda de manera
efectiva.

### Redes residuales (ResNet)

La solución llegó con las **redes residuales (ResNet)**, que introducen **conexiones de
atajo** (_skip connections_). Estas permiten transmitir activaciones de una capa a otra
más profunda, como si se tendieran puentes dentro de la red. Así, cada bloque residual
aprende no solo una transformación, sino también la diferencia (_residuo_) respecto a su
entrada. Este diseño posibilitó entrenar redes muy profundas, estableciendo un hito en
la visión computacional.

### Redes Inception

Otra innovación fue la **arquitectura Inception**, utilizada en modelos como GoogLeNet.
Su principio es aplicar en paralelo filtros de distintos tamaños (1×1, 3×3, 5×5) y una
operación de pooling, concatenando los resultados. Esto permite capturar información a
diferentes escalas. Para reducir el coste computacional, se introdujeron convoluciones
de 1×1 que actúan como cuellos de botella, disminuyendo la dimensionalidad antes de
aplicar filtros más grandes.

### MobileNet y arquitecturas ligeras

Con el auge de los dispositivos móviles surgió la necesidad de modelos más eficientes.
Así aparecieron las **MobileNet**, que se basan en **convoluciones separables en
profundidad**. El proceso se divide en:

1. **Convolución en profundidad:** cada filtro se aplica de forma independiente a un
   canal.
2. **Convolución puntual (1×1):** combina los resultados de todos los canales.

Este enfoque reduce enormemente el coste computacional. La segunda versión,
**MobileNetV2**, incorporó conexiones residuales y capas de expansión mediante filtros
1×1 para mejorar la capacidad de representación.

MobileNet permite ajustar su arquitectura según tres parámetros:

- **Ancho:** número de filtros por capa.
- **Resolución de entrada:** tamaño de la imagen procesada.
- **Profundidad:** número de capas totales.

De esta forma, la red puede adaptarse a distintos escenarios, desde aplicaciones en
tiempo real en móviles hasta entornos con gran capacidad de cómputo.

## 3. Detección de Objetos en Visión Computacional

En aplicaciones como la conducción autónoma no basta con clasificar una imagen: es
necesario identificar **qué objetos hay y dónde están**. Aquí entra la **detección de
objetos**, que combina **clasificación** y **localización** mediante recuadros
delimitadores (_bounding boxes_).

### Clasificación con localización

En el caso más simple, se entrena un modelo para detectar un único objeto por imagen,
prediciendo:

1. La probabilidad de presencia de un objeto.
2. Las coordenadas del recuadro.
3. La clase correspondiente.

### Detección de múltiples objetos

Para escenarios más complejos se emplean modelos capaces de detectar varios objetos en
una misma escena. Una estrategia consiste en dividir la imagen en una **malla de
celdas**, donde cada celda predice la clase y coordenadas de los objetos que contienen
su centro.

El algoritmo **YOLO (You Only Look Once)** implementa este enfoque aplicando la red
convolucional a toda la imagen de una sola vez, lo que permite detecciones en tiempo
real.

### Métricas y optimización

- **Intersección sobre unión (IoU):** mide la calidad de una predicción comparando el
  solapamiento entre la caja predicha y la real.
- **Supresión de no máximos (NMS):** elimina predicciones redundantes manteniendo solo
  la más confiable.
- **Cajas de anclaje (anchor boxes):** permiten a cada celda predecir múltiples objetos
  con diferentes proporciones y tamaños. Se seleccionan habitualmente mediante
  algoritmos de agrupamiento como _k-means_.

### Variantes del problema

- **Detección de puntos de referencia:** en lugar de cajas, se predicen coordenadas
  específicas, como rasgos faciales o articulaciones.
- **Métodos basados en regiones:** generan propuestas de posibles áreas de interés que
  luego son clasificadas, logrando mayor precisión aunque a costa de mayor tiempo de
  cómputo.

La detección de objetos es, por tanto, un paso clave hacia aplicaciones prácticas
avanzadas, combinando precisión y eficiencia en escenarios del mundo real.

## 4. Segmentación Semántica y la Arquitectura U-Net

La **segmentación semántica** lleva la visión por computadora a un nivel más detallado:
asignar a cada píxel una clase específica. El resultado es un mapa que representa con
precisión la forma de cada objeto, útil en áreas como la medicina, la agricultura o la
robótica.

### Convolución transpuesta

Para reconstruir la resolución original de la imagen se utiliza la **convolución
transpuesta**, que invierte el proceso de la convolución tradicional, expandiendo
progresivamente el tamaño espacial de la representación.

### U-Net

La **U-Net**, diseñada inicialmente para aplicaciones médicas, combina dos etapas:

- **Compresión:** reduce la resolución y aumenta los canales para extraer
  características abstractas.
- **Expansión:** recupera la resolución original mediante convoluciones transpuestas.

Para evitar la pérdida de detalles espaciales, se introducen **conexiones de omisión**,
que transfieren información de las capas iniciales a las correspondientes capas de
expansión. Esto permite segmentar con alta precisión bordes y contornos.

### Salidas posibles

Según el problema, la salida de la red puede variar: desde un único recuadro, hasta
múltiples coordenadas (por ejemplo, 2·n para _n_ puntos de referencia) o una máscara
completa de segmentación. En todos los casos, la segmentación semántica ofrece una
comprensión más rica y precisa de la imagen que la simple clasificación o detección.

## 5. One-Shot Learning y el Reconocimiento de Imágenes

Los modelos de visión suelen requerir grandes volúmenes de datos para aprender. Sin
embargo, en muchos casos solo se dispone de unos pocos ejemplos por clase. Este desafío
se aborda con el **One-Shot Learning** (aprendizaje con una sola muestra) o **Few-Shot
Learning** (con pocas muestras).

### Funciones de similitud y redes siamesas

La clave está en aprender un **espacio de representación** donde las imágenes similares
estén próximas y las distintas alejadas. Para medir la proximidad se utilizan métricas
como la distancia euclidiana.

Las **redes siamesas** procesan en paralelo dos imágenes con la misma red convolucional
compartiendo parámetros. El resultado son vectores de características que pueden
compararse directamente para decidir si pertenecen a la misma clase.

### Triplet Loss

El entrenamiento también puede organizarse mediante la **pérdida triple (triplet
loss)**, que utiliza un trío de imágenes:

- **Anchor:** muestra de referencia.
- **Positiva:** misma clase que el anchor.
- **Negativa:** clase distinta.

El objetivo es acercar las imágenes positivas al anchor y alejar las negativas, creando
representaciones robustas y discriminativas.

### Aprendizaje contrastivo y autosupervisado

El **aprendizaje autosupervisado** ha potenciado aún más estas técnicas mediante la
**pérdida contrastiva**, que aproxima pares de imágenes similares y aleja las distintas.
Esto puede hacerse sin etiquetas, generando versiones transformadas de la misma imagen
como pares positivos. Cuando se dispone de etiquetas, se utilizan imágenes de la misma
clase como positivas y de clases diferentes como negativas.

### Aplicaciones

El One-Shot Learning se aplica en múltiples áreas: reconocimiento facial, clasificación
de enfermedades raras en medicina o identificación de especies poco comunes. En lugar de
entrenar un clasificador rígido, el modelo aprende un espacio de representaciones donde
las distancias codifican similitud, lo que permite generalizar con muy pocos datos.

# 6. Modelos Secuenciales y Redes Recurrentes

Muchos problemas en inteligencia artificial implican datos **secuenciales**, es decir,
información organizada en un orden temporal o lógico. Ejemplos incluyen reconocimiento
de voz, generación de música, análisis de sentimientos en texto, secuencias de ADN o
traducción automática. Para abordarlos, se utilizan **modelos secuenciales**, que
procesan los datos considerando su orden y dependencia temporal.

## 6.1 Representación de secuencias

En tareas de lenguaje natural, las palabras deben transformarse en representaciones
comprensibles para un modelo. Este proceso, llamado **tokenización**, convierte cada
palabra en un índice único dentro de un diccionario y, posteriormente, en un vector que
codifica su información.

Se emplean tokens especiales para palabras desconocidas y, en generación de texto, un
token de **fin de secuencia** para indicar el cierre de la frase. Las longitudes de las
secuencias de entrada y salida pueden diferir, por lo que se manejan variables
específicas que reflejan estos tamaños.

## 6.2 Redes Neuronales Recurrentes (RNN)

Las **RNNs** introducen la capacidad de “recordar” información previa, reutilizando la
salida de un paso anterior como entrada en el siguiente. En cada instante, la red
combina la entrada actual con el estado oculto anterior para producir un nuevo estado y
una salida. Los parámetros se comparten a lo largo de la secuencia, permitiendo que la
predicción en un momento dado considere toda la información previa.

### Limitaciones

En la práctica, las RNN enfrentan dos problemas principales:

1. **Desvanecimiento de gradientes:** los gradientes se vuelven extremadamente pequeños,
   dificultando el aprendizaje de dependencias largas.
2. **Explosión de gradientes:** los gradientes crecen descontroladamente, afectando la
   estabilidad del entrenamiento.

## 6.3 Extensiones de las RNN

Para superar estas limitaciones, se desarrollaron variantes más robustas:

- **RNN bidireccionales:** procesan la secuencia en ambas direcciones, integrando
  información pasada y futura.
- **LSTM (Long Short-Term Memory):** incorporan una celda de memoria y puertas que
  controlan qué información se guarda, se olvida y se utiliza, capturando dependencias a
  largo plazo.
- **GRU (Gated Recurrent Unit):** versión simplificada de las LSTM, eficiente en
  recursos y con rendimiento comparable en muchos casos.

## 6.4 Modelos de lenguaje y predicción de secuencias

Los **modelos de lenguaje** asignan probabilidades a secuencias de palabras, prediciendo
la siguiente palabra dada una historia de texto. Se entrenan con grandes corpus para
aprender no solo asociaciones entre palabras, sino también patrones gramaticales y
contextuales. Estas redes habilitan aplicaciones como asistentes virtuales, análisis de
emociones y descifrado de secuencias genéticas.

## 7. Representación de Palabras y la Revolución de los Transformers

En el **procesamiento de lenguaje natural (NLP)**, una idea clave es la de los **word
embeddings**, vectores que representan palabras en un espacio continuo donde las
relaciones semánticas se reflejan en la geometría. Este enfoque supera al **one-hot
encoding**, que carecía de información semántica, y permite capturar similitudes,
analogías y relaciones complejas entre palabras.

### 7.1 Aprendizaje de embeddings

- **Word2Vec:** aprende a predecir palabras a partir de su contexto mediante ventanas de
  texto, utilizando técnicas como _negative sampling_ para reforzar relaciones
  relevantes.
- **GloVe:** combina coocurrencias globales de palabras con factorización de matrices,
  integrando información local y estadística global del corpus.

Estas representaciones pueden preentrenarse en grandes corpus y transferirse a tareas
específicas, aunque también reflejan **sesgos presentes en los datos**, que pueden
mitigarse mediante técnicas de neutralización.

## 7.2 Mecanismo de atención

La **atención** permite a los modelos centrarse en las partes relevantes de la entrada
según la tarea. Se implementa mediante tres vectores:

- **Query (Q):** lo que se busca.
- **Key (K):** la información disponible.
- **Value (V):** contenido asociado.

El modelo compara Q con K, asigna pesos y genera representaciones contextuales a partir
de V.

## 7.3 Transformers

Los **Transformers**, introducidos en _Attention is All You Need_, eliminan las RNN y
permiten procesar secuencias de forma paralela. Su arquitectura se compone de:

- **Encoder:** procesa la secuencia de entrada.
- **Decoder:** genera la secuencia de salida.

Cada bloque combina **autoatención** y redes totalmente conectadas. Dado que no procesan
secuencias de manera temporal, se incorporan **positional encodings** para conservar
información de orden.

El **multi-head attention** permite observar relaciones desde múltiples perspectivas
simultáneamente, enriqueciendo la representación. Los Transformers escalan
eficientemente, capturan dependencias complejas y se han extendido más allá del NLP a
visión computacional, bioinformática y aprendizaje por refuerzo.

# 8. Redes Neuronales de Grafos

Los grafos son estructuras flexibles y poderosas para representar información compleja,
compuestas por **nodos** (o vértices) y **aristas** (o conexiones) que describen las
relaciones entre los elementos. Esta formalización simple permite modelar fenómenos muy
diversos, desde redes sociales y moléculas hasta sistemas de telecomunicaciones,
imágenes y texto.

Las **Redes Neuronales de Grafos (GNN, Graph Neural Networks)** están diseñadas para
procesar directamente estas estructuras, extrayendo representaciones cada vez más ricas
de los nodos y del grafo en su conjunto, de manera similar a cómo las redes
convolucionales trabajan sobre imágenes.

## 8.1 Representación de nodos y flujo de información

Cada nodo de un grafo se representa mediante un **vector de características**. Durante
sucesivas iteraciones, este vector se actualiza combinando la información propia del
nodo con la de sus vecinos, enriqueciendo así su representación con el contexto del
grafo.

Dado que los grafos no poseen un orden natural de nodos o conexiones, las operaciones de
agregación (como suma o promedio) deben ser **conmutativas**, garantizando que el
resultado sea independiente del orden en que se procesan los vecinos.

Con cada iteración, los nodos adquieren representaciones que integran tanto sus
propiedades individuales como las de su entorno inmediato.

## 8.2 Representación de la estructura del grafo

La topología de un grafo puede representarse mediante:

- **Matriz de adyacencia:** indica la presencia o ausencia de aristas entre nodos. Es
  sencilla, pero su eficiencia depende del orden de los nodos y puede ser costosa en
  grafos grandes.
- **Listas de adyacencia:** enumeran explícitamente las conexiones de cada nodo,
  ofreciendo mayor flexibilidad y eficiencia.

En la práctica, estas representaciones se traducen en tensores que almacenan tanto las
características de los nodos como las relaciones que los unen.

## 8.3 Tareas sobre grafos

Las GNN permiten abordar problemas a diferentes niveles:

- **Nivel de grafo:** predicción de propiedades globales, como la clasificación de
  moléculas o la determinación del sentimiento de un texto completo.
- **Nivel de nodo:** identificación de roles o categorías de nodos, útil en segmentación
  de imágenes o detección de usuarios influyentes en redes sociales.
- **Nivel de arista:** predicción de existencia o valor de conexiones, como en
  recomendaciones de amistad o enlaces en grafos de conocimiento.

## 8.4 Arquitecturas y variantes

Entre las arquitecturas más destacadas se incluyen:

- **Graph Convolutional Networks (GCN):** cada nodo se actualiza a partir de la
  información de sus vecinos, de manera análoga a las convoluciones en imágenes.
- **Graph Attention Networks (GAT):** incorporan un mecanismo de atención que permite
  ponderar la importancia relativa de cada vecino, mejorando la capacidad de la red para
  diferenciar relaciones críticas de otras menos relevantes.

El flujo de información en una GNN se basa en el **intercambio de mensajes** entre
nodos. En grafos muy grandes, este proceso puede resultar costoso; por ello, se
introducen mecanismos como el **nodo maestro (masternode)**, que centraliza la
propagación de información global sin necesidad de mantener conexiones exhaustivas.

## 8.5 Aplicaciones

Los grafos ofrecen un marco unificado para múltiples dominios:

- **Visión por computadora:** una imagen puede representarse como nodos (píxeles o
  superpíxeles) conectados según proximidad o similitud.
- **Lenguaje natural:** palabras de una oración o documento pueden organizarse como
  nodos en grafos secuenciales o semánticos.
- **Biología y química:** moléculas y proteínas se describen naturalmente como grafos de
  átomos y enlaces.

En todos estos contextos, las GNN optimizan las representaciones de nodos, aristas y del
grafo completo, preservando la estructura intrínseca y capturando patrones complejos que
serían difíciles de detectar con modelos tradicionales.

$$
$$
