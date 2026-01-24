---
authors:
Daniel Bazo Correa
description: Fundamentos del Machine Learning.
title: Machine Learning
---

## 1. Introducción

### 1.1. Definición

<p align="center">
  <img src="https://www.techspot.com/articles-info/2048/images/2020-07-07-image.jpg"/>
  <br />
  <em>Ilustración sobre los conjuntos que engloba la inteligencia artificial. [Link](https://www.techspot.com/articles-info/2048/images/2020-07-07-image.jpg)</em>
</p>

El **aprendizaje automático** es una rama de la inteligencia artificial que se centra en
el desarrollo y uso de algoritmos, también denominados **modelos**, capaces de
identificar y comprender patrones en los datos de entrada con el objetivo de optimizar
una métrica establecida.

A diferencia de los enfoques tradicionales de programación, donde las reglas se definen
explícitamente, en el aprendizaje automático los algoritmos ajustan sus parámetros
automáticamente para mejorar su desempeño en función de los datos.

### 1.2. Técnicas

<p align="center">
  <img src="https://www.sharpsightlabs.com/wp-content/uploads/2021/04/regression-vs-classification_simple-comparison-image_v3.png"/>
  <br />
  <em>Clasificación vs Regresión. [Link](https://www.sharpsightlabs.com/wp-content/uploads/2021/04/regression-vs-classification_simple-comparison-image_v3.png)</em>
</p>

Entre las técnicas más utilizadas se encuentran la **clasificación** y la **regresión**.
La clasificación permite asignar etiquetas o categorías a los datos en función de sus
características comunes. Un ejemplo de clasificación es la identificación del tipo de
planta a partir de atributos como el ancho y la altura de sus hojas. Por otro lado, la
regresión se emplea para realizar predicciones numéricas, como la estimación del precio
de una vivienda en función de sus características.

La elección de la técnica adecuada depende de la naturaleza del problema. Un enfoque
común consiste en evaluar múltiples algoritmos viables y compararlos para determinar
cuál ofrece el mejor rendimiento. Esta comparación se basa en métricas de desempeño
obtenidas a partir de los datos.

**El proceso de entrenamiento de los modelos requiere dividir el conjunto de datos en
distintas partes**: una para el **entrenamiento** del modelo, otra para la
**evaluación** de su desempeño y, en algunos casos, una tercera partición para
**validar** su capacidad de generalización antes de su implementación en entornos
reales. Durante este proceso, el algoritmo analiza las relaciones entre las
características de los datos, identifica patrones y genera predicciones que se comparan
con los valores reales. La diferencia entre las predicciones y las observaciones se mide
mediante una métrica de error, lo que permite ajustar el modelo en cada iteración o
**época**, es decir, cada vez que el algoritmo analiza completamente el conjunto de
datos.

<p align="center">
  <img src="https://miro.medium.com/max/1125/1*_7OPgojau8hkiPUiHoGK_w.png"/>
  <br />
  <em>Ejemplo de subajuste, ajuste adecuado y sobreajuste. [Link](https://miro.medium.com/max/1125/1*_7OPgojau8hkiPUiHoGK_w.png)</em>
</p>

Un modelo puede presentar **sobreajuste** (**_overfitting_**) cuando se ajusta demasiado
a los datos de entrenamiento, logrando un alto rendimiento en estos pero fallando en
datos nuevos. Este problema se conoce como el **compromiso entre sesgo y varianza**
(**_bias-variance tradeoff_**), y su mitigación es esencial para obtener modelos que
generalicen correctamente.

La imposibilidad de un algoritmo de aprendizaje automático de capturar la relación real
de los datos, se conoce como sesgo y la diferencia de ajuste entre el CD entrenamiento y
otras, como el de validación y lo pruebas se conoce como varianza lo ideal es tener un
bajo sesgo para modelar con mayor exactitud, la distribución de los datos en la baja
varianza, para que el resultado de las predicciones sea consistente para diferentes
conjuntos de datos.

### 1.3. Tipos de datos

#### 1.3.1. Variables dependientes e independientes

En un conjunto de datos, cada atributo que varía entre muestras se denomina
**variable**. Si una variable depende de otra, se considera **dependiente**, en caso
contrario, se clasifica como **independiente**. Las variables independientes, también
llamadas **características** (**_features_**), son las utilizadas en el entrenamiento
del modelo para predecir la variable dependiente.

#### 1.3.2. Datos continuos y discretos

<p align="center">
  <img src="https://agencyanalytics.com/_next/image?url=https%3A%2F%2Fimages.ctfassets.net%2Fdfcvkz6j859j%2F6k4gJrY1mvlPUxf7WZhqdp%2F9f2e800789b81fa6fe751fabf50e9069%2FDiscrete-vs-Continuous-Data-Supporting-Graphics-1.png&w=3840&q=75"/>
  <br />
  <em>Datos discretos vs datos continuos. [Link](https://agencyanalytics.com/_next/image?url=https%3A%2F%2Fimages.ctfassets.net%2Fdfcvkz6j859j%2F6k4gJrY1mvlPUxf7WZhqdp%2F9f2e800789b81fa6fe751fabf50e9069%2FDiscrete-vs-Continuous-Data-Supporting-Graphics-1.png&w=3840&q=75)</em>
</p>

Los datos pueden clasificarse en **continuos** o **discretos**. Los valores continuos
pueden tomar cualquier número dentro de un rango, como la altura de una persona, ya que
pueden existir valores intermedios con una precisión arbitraria. En contraste, los
valores discretos solo pueden asumir ciertos valores específicos, como la cantidad de
páginas de un libro, donde no existen valores intermedios entre un número entero y otro.

## 2. Estrategias para la selección y validación de datos

Los datos son un elemento esencial en los algoritmos de aprendizaje automático. Sin una
selección adecuada, es posible obtener relaciones no significativas o incluso
perjudiciales.

No todos los datos o métricas son útiles, por lo que es fundamental ajustarse al
problema, asegurar la coherencia dentro de la misma distribución y minimizar la
presencia de valores atípicos. Una correcta selección de los datos permite desarrollar
modelos más robustos. Para ello, se emplea la **validación cruzada**.

### 2.1. Validación cruzada

<p align="center">
  <img src="https://www.sharpsightlabs.com/wp-content/uploads/2024/02/cross-validation-explained_FEATURED-IMAGE.png"/>
  <br />
  <em>Esquema de funcionamiento de la validación cruzada. [Link](https://www.sharpsightlabs.com/wp-content/uploads/2024/02/cross-validation-explained_FEATURED-IMAGE.png)</em>
</p>

La selección de muestras para el entrenamiento y validación de un modelo puede resultar
compleja, ya que una elección inadecuada puede generar sesgos en el modelo.

Por ejemplo, en conjuntos de datos con dependencia temporal, como el tráfico de una red
a lo largo del día, la distribución de las muestras en el conjunto de datos puede
influir en el desempeño del modelo. Si los datos se registran en orden cronológico y las
primeras muestras corresponden a la mañana, mientras que las últimas a la noche,
seleccionar las primeras muestras para entrenamiento y las últimas para prueba, podría
generar un modelo que no capture correctamente patrones generales.

Para evitar este problema, se recomienda introducir aleatoriedad en la selección de las
muestras y definir un porcentaje para cada partición del conjunto de datos.

note

Es fundamental establecer una **semilla aleatoria** antes de cualquier proceso que
requiera aleatorización, garantizando así la reproducibilidad de los resultados.

Por ejemplo, el siguiente código establece semillas para las bibliotecas más utilizadas
en Python para aprendizaje automático y profundo, garantizando la reproducibilidad de
los experimentos:

```py linenums="1"
import random
import numpy as np
import tensorflow as tf
import torch
import sklearn.utils

# Valor de la semilla
SEED = 42

# Establecer semilla en Python (random)
random.seed(SEED)

# Establecer semilla en NumPy
np.random.seed(SEED)

# Establecer semilla en TensorFlow
tf.random.set_seed(SEED)

# Establecer semilla en PyTorch
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)  # Para GPUs
torch.cuda.manual_seed_all(SEED)  # Para múltiples GPUs
torch.backends.cudnn.deterministic = True  # Para reproducibilidad en CUDA
torch.backends.cudnn.benchmark = False

# Establecer semilla en Scikit-learn
sklearn.utils.check_random_state(SEED)
```

Otra estrategia para la selección de datos es la **validación cruzada**, la cual
consiste en dividir el conjunto de datos en múltiples partes y realizar iteraciones en
las que se alternan los subconjuntos destinados a entrenamiento y prueba. Si se opta por
una validación cruzada de 5 particiones (**_5-fold cross-validation_**), el modelo se
entrena y evalúa cinco veces, cada vez utilizando un subconjunto distinto para prueba y
los demás para entrenamiento. Posteriormente, los resultados obtenidos en cada iteración
se promedian para obtener una evaluación más robusta del modelo.

El proceso de validación cruzada se basa en la partición de los datos en subconjuntos
denominados **_folds_**. En cada iteración, se entrena el modelo con algunos _folds_ y
se evalúa con el _fold_ restante. Este procedimiento se repite hasta que cada _fold_
haya sido utilizado tanto para entrenamiento como para prueba. Finalmente, los
resultados se promedian utilizando métricas como la precisión o el error del modelo.

Una ventaja de la validación cruzada es la reducción del problema conocido como **_data
leakage_**, que ocurre cuando características utilizadas en el entrenamiento también
están presentes en la fase de prueba, generando una evaluación artificialmente optimista
del modelo.

La **validación cruzada** es una técnica fundamental en aprendizaje automático para
evaluar y comparar diferentes modelos. Su objetivo es estimar el rendimiento de un
modelo en datos no vistos y seleccionar el algoritmo más adecuado. El proceso consiste
en dividir el conjunto de datos en varios subconjuntos llamados **folds**, generalmente
de tamaño similar, y entrenar el modelo de manera iterativa:

1. Se separa el conjunto de datos en (k) folds (por ejemplo, cinco o diez).
2. En cada iteración, se utiliza un fold como conjunto de prueba y los restantes como
   conjunto de entrenamiento.
3. El modelo se entrena con los folds de entrenamiento y se evalúa con el fold de
   prueba.
4. Este procedimiento se repite hasta que todos los folds hayan sido utilizados como
   conjunto de prueba una vez.
5. Finalmente, se promedian las métricas de evaluación obtenidas en cada iteración, como
   precisión, error, sensibilidad, etc.

El caso más común es el **k-Fold Cross Validation**, donde (k) suele ser cinco o diez,
dependiendo del tamaño del conjunto de datos y de la complejidad del modelo.

## 3. Conceptos de estadística

### 3.1. Distribuciones

Antes de realizar predicciones, es fundamental recopilar datos. En muchas ocasiones,
esta recopilación genera histogramas, que permiten visualizar la distribución de los
datos.

Un histograma se compone de dos ejes principales: el eje x, donde se representan los
datos agrupados en categorías, y el eje y, que indica la frecuencia de cada categoría,
es decir, el número de muestras que pertenecen a cada grupo. Las divisiones en el eje x
para agrupar los datos en rangos similares se conocen como **_bins_** o contenedores.

El uso de histogramas facilita la identificación de tendencias en los datos. En casos
donde los valores pueden solaparse, los _bins_ ayudan a agrupar puntos de datos dentro
de un intervalo definido. De este modo, se generan distribuciones que permiten analizar
el comportamiento de los datos.

note

La elección del número de _bins_ es crucial, ya que debe reflejar correctamente la
distribución de los datos. Este tipo de histogramas resulta especialmente útil en
algoritmos como **Naïve Bayes**, donde se generan distribuciones de probabilidad en cada
iteración, permitiendo obtener valores como medias e intervalos de confianza.

El conjunto completo de datos recopilados se denomina **población** y se representa con
la letra $N$. Un subconjunto de la población se denomina **muestra** y se representa con
la letra $n$.

La probabilidad de que un dato pertenezca a una determinada parte del histograma se
calcula dividiendo el número de muestras en esa sección entre el número total de
muestras en la población.

note

La confianza en los resultados depende del tamaño de la muestra: cuanto mayor sea el
número de muestras, mayor será la confianza en la estimación. Donde la confianza
representa el grado de incertidumbre asociado a una probabilidad.

### 3.2. Características de la probabilidad

La probabilidad está normalizada en un rango de 0 a 1, donde 0 indica imposibilidad y 1
certeza absoluta. Cuando todos los resultados posibles tienen la misma probabilidad, se
habla de **equiprobabilidad**. Además, la suma de todas las probabilidades en un sistema
debe ser 1.

Cuando el número de datos disponibles es insuficiente, las estimaciones de probabilidad
pueden no ser precisas. No obstante, recopilar más datos puede resultar costoso en
términos de tiempo, esfuerzo y dinero. Para mitigar esta limitación, se emplean
**distribuciones de probabilidad**, que pueden ser **discretas** (cuando los datos toman
valores específicos y finitos) o **continuas** (cuando los datos pueden tomar cualquier
valor dentro de un rango determinado).

A continuación, se presentan algunas de las distribuciones más comunes.

#### 3.2.1. Distribución binomial (discreta)

Cuando se trabaja con datos discretos y se requiere calcular probabilidades en eventos
independientes con solo dos posibles resultados, **éxito** o **fracaso** (representados
por 1 y 0, respectivamente), se trata de un **problema binario**.

Para modelar este tipo de situaciones, se utiliza la **distribución binomial**, que
permite calcular la probabilidad de obtener una determinada cantidad de éxitos en una
secuencia de ensayos independientes. La distribución binomial se expresa mediante la
siguiente fórmula:

$$
P(X = k | n, p) = \binom{n}{k} \cdot p^k \cdot (1 - p)^{n - k},
$$

donde:

- $X$ representa el número de éxitos en los ensayos.
- $n$ es el número total de ensayos.
- $p$ es la probabilidad de éxito en un único ensayo.
- $k$ es el número de éxitos deseados.
- $\binom{n}{k}$ es el coeficiente binomial, que calcula de cuántas formas se pueden
  obtener $k$ éxitos en $n$ ensayos, sin importar el orden. Se calcula mediante la
  siguiente fórmula:

$$
\binom{n}{k} = \frac{n!}{k! \cdot (n-k)!}.
$$

Esta distribución es útil en situaciones donde se realizan múltiples intentos
independientes de un mismo experimento y se desea conocer la probabilidad de obtener un
número específico de éxitos.

???+ example "Ejemplo"

Supongamos que se lanza una moneda equilibrada (equiprobable, la probabilidad de tener
cara es la misa que de tener cruz) 5 veces y se quiere calcular la probabilidad de
obtener exactamente 3 caras.

Se definen los parámetros:

- $n = 5$ (número total de lanzamientos).
- $p = 0.5$ (probabilidad de obtener cara en un solo lanzamiento).
- $k = 3$ (número de caras que se desean obtener).

Aplicando la fórmula de la distribución binomial:

$$
P(X = 3 | n=5, p=0.5) = \binom{5}{3} \cdot (0.5)^3 \cdot (1 - 0.5)^{5 - 3}
$$

Calculamos el coeficiente binomial:

$$
\binom{5}{3} = \frac{5!}{3! \cdot (5-3)!} = \frac{5!}{3! \cdot 2!} = 10
$$

Sustituyendo en la ecuación:

$$
P(X = 3) = 10 \cdot (0.5)^3 \cdot (0.5)^2 = 0.3125
$$

Por lo tanto, la probabilidad de obtener exactamente 3 caras en 5 lanzamientos de una
moneda equilibrada es del 31.25%.

#### 3.2.2. Distribución de Poisson (discreta)

La **distribución de Poisson** se utiliza para modelar la probabilidad de que ocurra un
número determinado de eventos en un intervalo de tiempo o espacio, siempre que los
eventos ocurran de manera independiente y a una tasa promedio constante. Algunos
ejemplos de aplicación incluyen: el número de llamadas recibidas en una central
telefónica durante una hora, el número de accidentes en una intersección en un día, o la
cantidad de errores tipográficos en una página de texto.

La distribución de Poisson se expresa mediante la siguiente fórmula:

$$
P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!},
$$

donde:

- $X$ es el número de eventos que ocurren en un intervalo específico.
- $\lambda$ es el número promedio de eventos en dicho intervalo.
- $k$ es el número de eventos cuya probabilidad se desea calcular.
- $e$ es la base del logaritmo natural.

Esta distribución es especialmente útil cuando se estudian eventos raros o poco
frecuentes en un período de tiempo determinado.

???+ example "Ejemplo"

Supongamos que una central telefónica recibe en promedio 10 llamadas por hora y se desea
calcular la probabilidad de que en una hora lleguen exactamente 7 llamadas.

Se definen los parámetros:

- $\lambda = 10$ (promedio de llamadas por hora).
- $k = 7$ (número específico de llamadas que se desea calcular).

Aplicamos la fórmula de la distribución de Poisson:

$$
P(X = 7) = \frac{10^7 e^{-10}}{7!} \approx 0.0902
$$

Por lo tanto, la probabilidad de recibir exactamente 7 llamadas en una hora es del
9.02%.

#### 3.2.3. Distribución Normal o Gaussiana (continua)

La distribución normal, también denominada distribución gaussiana, se representa
mediante una curva en forma de campana. En esta distribución, el eje $$y$$ indica la
**verosimilitud** (**_likelihood_**) de observar un determinado valor en el eje $$x$$.

note Verosimilitud vs. Probabilidad

Aunque son conceptos relacionados, la verosimilitud y la probabilidad tienen diferencias
clave:

- **Probabilidad**: Representa la posibilidad de que ocurra un evento dado un modelo y
  sus parámetros. Se expresa como $$P(D|\theta)$$, donde $$D$$ son los datos y
  $$\theta$$ los parámetros del modelo. Responde a la pregunta: _Dado que los parámetros
  del modelo son conocidos, ¿qué tan probable es observar ciertos datos?_
- **Verosimilitud**: Mide qué tan bien un conjunto de parámetros explica un conjunto de
  datos observados. Se denota como $$L(\theta | D)$$ y representa la plausibilidad de
  los parámetros $$\theta$$ dados los datos $$D$$. Responde a la pregunta: _Dado que los
  datos han sido observados, ¿qué tan plausible es que provengan de un modelo con
  ciertos parámetros?_

Mientras que la probabilidad se emplea para predecir eventos futuros basándose en un
modelo conocido, la verosimilitud se usa para evaluar qué tan bien un modelo con ciertos
parámetros explica los datos observados. Para obtener probabilidades a partir de la
verosimilitud, se puede utilizar el Teorema de Bayes.

La distribución normal es **simétrica** respecto a su **media** ($$\mu$$), lo que
implica que el valor más verosímil es precisamente la media. La forma de la curva normal
está determinada por dos parámetros: la media ($$\mu$$) y la desviación típica
($$\sigma$$).

- **Una curva alta y estrecha** indica que los datos están más concentrados alrededor de
  la media, lo que corresponde a una **baja varianza**.
- **Una curva baja y ancha** sugiere una mayor dispersión de los datos, es decir,
  **mayor varianza**.

La **desviación típica** ($$\sigma$$) mide la dispersión de los datos respecto a la
media, mientras que la **varianza** ($$\sigma^2$$) es el cuadrado de la desviación
típica. La varianza se puede calcular de las siguientes dos maneras:

- **Varianza muestral**:

  $$
  s^2 = \frac{\sum_{i=1}^{n} (x_i - \bar{x})^2}{n-1}.
  $$

- **Varianza poblacional**:

  $$
  \sigma^2 = \frac{\sum_{i=1}^{N} (x_i - \mu)^2}{N}.
  $$

Donde:

- $$x_i$$ son los valores de la muestra o población.
- $$\bar{x}$$ es la media muestral.
- $$\mu$$ es la media poblacional.
- $$n$$ es el tamaño de la muestra.
- $$N$$ es el tamaño de la población.

La distribución normal es fundamental en estadística y aprendizaje automático debido a
su presencia en numerosos fenómenos naturales y conjuntos de datos del mundo real.

##### 3.2.3.1. Función de Densidad de Probabilidad

La **función de densidad de probabilidad** (_Probability Distribution Function_, PDF) de
la distribución normal se define como:

$$
f(X|\mu, \sigma) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x_i - \mu)^2}{2\sigma^2}}.
$$

En las distribuciones continuas, el cálculo de probabilidades requiere la integración de
la función de densidad de probabilidad (PDF). Esta integración permite obtener el área
bajo la curva entre dos puntos, lo que representa la probabilidad acumulada en dicho
intervalo. Dado que el área total bajo la curva es igual a 1, el área acumulada hasta la
media en una distribución normal es de 0.5.

Es importante destacar que la probabilidad exacta en un único punto es igual a 0. Esto
se debe a que, gráficamente, un punto no tiene ancho y, por lo tanto, no contribuye con
área bajo la curva. En consecuencia, solo es posible calcular probabilidades en
intervalos.

La **función de distribución acumulada** (_Cumulative Distribution Function_, CDF)
expresa la probabilidad acumulada hasta un determinado valor. Matemáticamente,
representa el área bajo la curva de la función de densidad desde $$-\infty$$ hasta dicho
punto.

##### 3.2.3.2. Propiedades de la Función de Densidad de Probabilidad

Sea $F$ la función de distribución acumulada (CDF) y $\mathbb{R}$ el conjunto de números
reales, entonces se cumple que $F: \mathbb{R} \to [0,1]$, lo que significa que el rango
de valores de la función de distribución está comprendido entre 0 y 1.

note

Se usa mayúscula para $F(x)$ porque se refiere a la función matemática que mapea los
valores de la variable aleatoria $X$ a la probabilidad acumulada, distinguiéndola de la
función de densidad de probabilidad que se representa en minúscula, como $f(x)$.

Algunas de sus propiedades fundamentales son:

- $F(x) = P(A_x) = P(X \leq x)$, donde $A_x$ representa el evento $X \leq x$.
- $P(X \leq x) = F(x)$, lo que corresponde a la función de distribución acumulada (CDF).
- $P(X > x) = 1 - P(X \leq x) = 1 - F(x)$, es decir, la probabilidad complementaria a
  $P(X \leq x)$.
- $P(a < X \leq b) = F(b) - F(a)$, para calcular la probabilidad de que $X$ esté entre
  dos valores $a$ y $b$.
- $P(X \geq a) = P(X > a) + P(X = a)$, una forma de descomponer la probabilidad de que
  $X$ sea mayor o igual que $a$.
- $P(X = a) = F(a) - \lim\_{h \to 0^+} F(a - h) = F(a) - P(X \leq a)$, que calcula la
  probabilidad de que $X$ tome el valor $a$.

Estas propiedades permiten calcular probabilidades acumuladas y facilitan el análisis de
distribuciones de probabilidad continuas.

???+ example "Ejemplo"

Se desea calcular la probabilidad de que un valor se encuentre en el intervalo
$[142.5,
155.7]$ en una distribución normal $N(\mu=155.7, \sigma=6.6)$.

La probabilidad se obtiene a partir de la función de distribución acumulada (CDF):

$P(a < X \leq b) = P(142.5 < X \leq 155.7) = F(155.7) - F(142.5)$

$P(X \leq 155.7) - P(X \leq 142.5) \approx 0.5 - 0.02275 \approx 0.4772$

Implementación en Python:

```py linenums="1"
from statistics import NormalDist

# Se calcula la función de distribución acumulada (CDF) en los puntos de interés
cdf_p1 = NormalDist(155.7, 6.6).cdf(155.7)
# cdf_p1 = 0.5, debido a la simetría de la distribución normal

cdf_p2 = NormalDist(155.7, 6.6).cdf(142.5)
# cdf_p2 ≈ 0.02275

# Se obtiene la probabilidad del intervalo restando las probabilidades acumuladas
diff = cdf_p1 - cdf_p2
# diff ≈ 0.4772 = 47.72%
```

Por lo tanto, la probabilidad de que un valor de esta distribución normal se encuentre
en el intervalo $[142.5, 155.7]$ es aproximadamente del 47.72%.

#### 3.2.4. Distribución Exponencial (continua)

La distribución exponencial se emplea para modelar el tiempo transcurrido entre eventos
en un proceso de Poisson, donde los eventos ocurren de manera independiente y con una
tasa constante. Se utiliza en el análisis de tiempos de espera, confiabilidad de
sistemas y modelado de fallos en ingeniería.

La función de densidad de probabilidad (PDF) está definida como:

$$
f(x; \lambda) = \lambda e^{-\lambda x}, \quad x \geq 0, \, \lambda > 0,
$$

donde $\lambda$ indica la frecuencia con la que ocurren los eventos.

La función de distribución acumulada (CDF) se expresa como:

$$
F(x) = 1 - e^{-\lambda x}, \quad x \geq 0.
$$

La media de la distribución exponencial equivale a la **esperanza matemática** $E[X]$.

note

La **esperanza matemática**, denotada como $E[X]$, es lo que comúnmente llamamos la
**media** o el **valor esperado** de una variable aleatoria. Sin embargo, la
interpretación de la esperanza matemática puede variar dependiendo del tipo de variable
aleatoria y el contexto en el que se utilice.

Para una variable aleatoria discreta $X$, cuya función de masa de probabilidad es
$P(X = x_i)$, la esperanza matemática se calcula mediante la siguiente fórmula:

$$
E[X] = \sum_{i} x_i \cdot P(X = x_i)
$$

En este caso, el valor esperado se obtiene sumando el producto de cada valor posible de
$X$ y su probabilidad correspondiente.

Para una variable aleatoria continua $X$, cuya función de densidad de probabilidad es
$f(x)$, la esperanza matemática se calcula utilizando la integral de la siguiente
manera:

$$
E[X] = \int_{-\infty}^{\infty} x \cdot f(x) \,dx
$$

Aquí, el valor esperado se obtiene integrando el producto de cada valor de $X$ y su
densidad de probabilidad asociada.

En teoría de probabilidad, la esperanza matemática se considera la media "teórica" de la
distribución. **En distribuciones simétricas con un único pico, como la distribución
normal, la esperanza matemática coincide con el centro de la distribución. Sin embargo,
en distribuciones asimétricas, la esperanza matemática puede no coincidir con la mediana
o la moda**. Por ejemplo, en una distribución sesgada a la derecha, como la distribución
exponencial, la esperanza matemática es mayor que la mediana, lo que indica que los
valores más altos de la variable aleatoria tienen una probabilidad significativa de
ocurrir.

En la distribución exponencial, se obtiene:

$$
\mu = E[X] = \frac{1}{\lambda}.
$$

La varianza se expresa como:

$$
\sigma^2 = \frac{1}{\lambda^2}.
$$

#### 3.2.5. Distribución Uniforme (continua)

La distribución uniforme se caracteriza porque todos los valores dentro de un intervalo
$[a, b]$ tienen la misma probabilidad de ocurrir. Se emplea en la generación de números
aleatorios, simulaciones y situaciones en las que no hay preferencia por ningún valor
específico dentro de un rango determinado.

La función de densidad de probabilidad (PDF) para una distribución uniforme continua es:

$$
f(x) = \begin{cases} \frac{1}{b-a}, & a \leq x \leq b \\ 0, & \text{en otro caso}
\end{cases}
$$

La función de distribución acumulada (CDF) está dada por:

$$
F(x) = \begin{cases} 0, & x < a \\ \frac{x-a}{b-a}, & a \leq x \leq b \\ 1, & x > b
\end{cases}
$$

La media de la distribución uniforme es:

$$
\mu = E[X] = \frac{a + b}{2}.
$$

Y su varianza se expresa como:

$$
\sigma^2 = \frac{(b-a)^2}{12}.
$$

tip ¿De dónde sale el 12 de la varianza de la distribución uniforme?

La varianza de una variable aleatoria continua $X$ se define como:

$$
\text{Var}(X) = E[X^2] - (E[X])^2.
$$

Para una distribución uniforme continua $U(a, b)$, la esperanza, la cual coincide con la
media, se obtiene con la fórmula:

$$
E[X] = \frac{a+b}{2}.
$$

Por tanto, el cálculo de $E[X^2]$ se calcula como:

$$
E[X^2] = \int_a^b x^2 \cdot f(x) \, dx = \int_a^b x^2 \cdot \frac{1}{b-a} \, dx.
$$

Resolviendo la integral:

$$
E[X^2] = \frac{1}{b-a} \int_a^b x^2 \, dx = \frac{1}{b-a} \cdot \left[ \frac{x^3}{3} \right]_{a}^{b}.
$$

Evaluando de $a$ a $b$:

$$
E[X^2] = \frac{1}{b-a} \left[ \frac{b^3}{3} - \frac{a^3}{3} \right] = \frac{b^3 -
a^3}{3(b-a)}.
$$

Ahora, usando la fórmula de la varianza y sustituyendo los valores obtenemos:

$$
\text{Var}(X) = E[X^2] - (E[X])^2 = \frac{b^3 - a^3}{3(b-a)} - \left(\frac{a+b}{2} \right)^2.
$$

Aplicando las identidades algebraicas siguientes:

$$
(a+b)^{2}=a^2+b^2+2ab,
$$

$$
b^3 - a^3 = (b-a)(b^2 + ab + a^2),
$$

Finalmente, después de desarrollar la expresión y simplificar, se obtiene la expresión:

$$
\text{Var}(X) = \frac{(b-a)^2}{12}
$$

### 3.3. Evaluación del error

Los modelos de aprendizaje automático requieren datos de entrenamiento para establecer
relaciones entre las variables y construir una función que se aproxime a la distribución
de los datos. Un aspecto fundamental en este proceso es la evaluación del desempeño del
modelo, lo cual se realiza mediante métricas estadísticas.

#### 3.3.1. Suma de los Cuadrados de los Residuales (SSR)

<p align="center">
  <img src="https://images.squarespace-cdn.com/content/v1/5acbdd3a25bf024c12f4c8b4/1600368657769-5BJU5FK86VZ6UXZGRC1M/Mean+Squared+Error.png"/>
  <br />
  <em>Ejemplo de SSR. [Link](https://images.squarespace-cdn.com/content/v1/5acbdd3a25bf024c12f4c8b4/1600368657769-5BJU5FK86VZ6UXZGRC1M/Mean+Squared+Error.png)</em>
</p>

La **Suma de los Cuadrados de los Residuales (_Sum Square Residuals_, SSR)** mide la
diferencia entre las predicciones del modelo y los valores reales. Se calcula sumando el
cuadrado de estas diferencias, lo que permite evaluar qué tan buena es la predicción del
modelo. Un valor bajo de SSR indica un mejor ajuste.

Matemáticamente, la SSR se expresa como:

$$
SSR = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2,
$$

donde:

- $y_i$ es el valor real.
- $\hat{y}_i$ es el valor estimado por el modelo.
- $n$ es el número total de observaciones.

Sin embargo, la SSR depende del número de datos, lo que puede dificultar la comparación
entre modelos. Para abordar este problema, se emplea el **Error Cuadrático Medio
(MSE)**.

#### 3.3.2. Error Cuadrático Medio (MSE)

El **Error Cuadrático Medio (MSE)** se obtiene dividiendo la SSR entre el número total
de muestras. Su objetivo es promediar la magnitud del error para normalizarlo con
respecto al tamaño del conjunto de datos. Se define como:

$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2.
$$

A pesar de que el MSE proporciona una medida más interpretable del error, sigue
dependiendo de la escala de los datos. Para eliminar esta dependencia, se emplea el
**Coeficiente de Determinación ($R^2$)**.

#### 3.3.3. Coeficiente de Determinación

El **Coeficiente de Determinación ($R^2$)** mide la capacidad del modelo para replicar
los resultados observados y la proporción de variabilidad explicada por el modelo en
comparación con la media de los datos. Se expresa como:

$$
R^2 = 1 - \frac{\text{SSR}}{SST} = 1 - \frac{\text{SSR}(\text{respecto al modelo})}{\text{SSR}(\text{respecto a la media})}
$$

donde:

- $\text{SST}$ es la **Suma Total de los Cuadrados**, que representa la variabilidad
  total de los datos en torno a la media.

**El coeficiente $R^2$ varía entre 0 y 1, donde un valor cercano a 1 indica que el
modelo explica bien la varianza de los datos, lo que sugiere un buen ajuste**. En
cambio, un valor cercano a 0 sugiere que el modelo apenas mejora la predicción en
comparación con la media. Si $R^2$ es negativo, el modelo tiene un mal ajuste y predice
peor que la media.

El coeficiente $R^2$ se emplea en problemas de regresión sobre datos continuos.

note

El coeficiente $R^2$ equivale al cuadrado del coeficiente de correlación de Pearson solo
en el caso de la regresión lineal simple.

#### 3.3.4. Coeficiente de Correlación de Pearson

<p align="center">
  <img src="https://external-content.duckduckgo.com/iu/?u=http%3A%2F%2Fwww.statisticshowto.com%2Fwp-content%2Fuploads%2F2012%2F10%2Fpearson-2-small.png&f=1&nofb=1&ipt=25bc8844d74e829cb2103b12684b70568fc8a54b572ffa6ac17a40d3e106789d&ipo=images"/>
  <br />
  <em>Ejemplo de la correlación para una nuve de puntos. [Link](http://www.statisticshowto.com/wp-content/uploads/2012/10/pearson-2-small.png)</em>
</p>

El **Coeficiente de Correlación de Pearson** mide la relación lineal entre dos variables
cuantitativas y continuas. Se define como:

$$
r = \frac{\text{cov}(X,Y)}{\sigma_X \sigma_Y}
$$

donde:

- $\text{cov}(X,Y)$ es la **covarianza** entre las variables $X$ e $Y$.
- $\sigma_X$ y $\sigma_Y$ son las desviaciones típicas de $X$ e $Y$, respectivamente.

La **covarianza** indica la relación entre dos variables:

- Si la covarianza es **positiva**, un aumento en $X$ se asocia con un aumento en $ Y$
  (relación directa).
- Si la covarianza es **negativa**, un aumento en $X$ se asocia con una disminución en
  $Y$ (relación inversa).
- Una covarianza cercana a **0** sugiere que no existe relación lineal entre las
  variables.

Dado que la covarianza depende de la escala de las variables, se normaliza mediante el
coeficiente de correlación de Pearson, que toma valores entre -1 y 1, donde:

- **1**: correlación positiva perfecta.
- **-1**: correlación negativa perfecta.
- **0**: ausencia de correlación lineal.

Este coeficiente permite evaluar la intensidad y dirección de la relación lineal entre
las variables sin depender de su escala.

#### 3.3.5. P-values

los piba luz son números entre cero y uno que cuantifican como de seguros podemos estar
de que una opción A siendo mejor sea diferente de una opción B viva luz cercano a cero
significa que a es distinto bebé se suele usar con un umbral de 0.5 para indicar que a
es distinto debe alerta puede darse el caso de obtener un valor pequeño de Tiwayo cuando
no existe diferencia, es decir, cuando hay un falso positivo un umbral de 0.0 cinco
significa que el 5 % de los experimentos generará un valor de pi Balliu menor a 0.0
cinco. Si necesitamos estar seguros, podemos usar umbrales más bajos, por ejemplo para
medicina que se podría utilizar umbrales como a 0.0 0001 lo que indica un falso positivo
cada 100.000 experimentos. El umbral es el umbral de significancia que es la
probabilidad máxima aceptada de cometer un falso positivo y se y es la inversa de la
frecuencia la idea de intentar determinar si una opción no es igual o no a una opción B
se llama prueba de hipótesis y lo que queremos ver es ya es igual a ver para ver si es
una hipótesis nula por tanto el P valor mide la probabilidad de que el resultado
observado ocurra. Si no hay diferencias lo que es una hipótesis nula verdadera, pero no
mide la magnitud de la diferencia.

## 4. Modelos clásicos

Una vez comprendido el concepto de modelo de aprendizaje automático, donde se utilizan
datos para modelar su distribución, analizar relaciones y extraer conocimiento, es
posible aplicar estos modelos para realizar tareas como clasificación de nuevos datos,
predicción de valores y otras aplicaciones. A continuación, se presentan algunos de los
métodos más utilizados.

tip

A pesar del auge de los modelos de lenguaje basados en arquitecturas de **aprendizaje
profundo (_Deep Learning_)**, su aplicación sigue siendo limitada en ciertos contextos
debido a la gran cantidad de datos y capacidad de cómputo que requieren, así como a la
necesidad de explicabilidad en sectores específicos. Por ello, los métodos tradicionales
siguen desempeñando un papel fundamental, especialmente en el análisis de datos
**tabulares**, los cuales representan la mayoría de los datos empresariales.

Es recomendable iniciar con modelos más sencillos para comprender los resultados y
evaluar su utilidad en función de los objetivos del análisis. A partir de esta base, y
considerando factores como el tiempo y los recursos disponibles, se puede optar por
soluciones más complejas que ofrezcan un mayor retorno de inversión (ROI).

### 4.1. Regresión lineal

En los modelos lineales, lo que hacemos es ajustar una linea, utilizando least square
para hacer un fit de los datos, vamos que nos quedamos con la linea donde el valor de
SSR sea minimo, mejor ajuste modelo a los datos. Calculamos la R2 y tambien podemos
calcular los p-values para R2. Calcular promedi tamaño, obtener linea promedio si lo
aplanamos todos podemos visualizar el residuo para ver como se ajusta la linea ajustada
a los datos y los valores reales para ver la variabiliadad, podemos calcular la varianza
que seria la SS media que es el punto X menos la media todo al cuadrado y dividido por
el numero de puntos, podemos calcular el promedio que es el tamaño para obtener la niña
promedio. Luego también podemos realizar el cálculo de la varianza que es el SS promedio
dividido entre N que indica el grado de dispersión que tenemos luego la R cuadrado que
es la barra la media de la varianza menos la varianza del ajuste dividido de la media de
la varianza. Por ejemplo, si la R cuadrado es 0.6 yo significa que es el 60 %, que
decimos que X por ejemplo no pues explica el 60 % de la variación de las muestras.

En los modelos lineales podemos introducir más parámetros no solo dos pero si esos
parámetros adicionales empeoran el ajuste del modelo a los datos, dichos parámetros
serán multiplicados por el modelo por cero o para no tenerlos en cuenta.

### 4.2. DesceQue sea estocástico implica que se elige de forma aleatoria un punto por paso ya que hacerlo para todo sería costoso computacional mente aunque realmente se toma un subconjunto llamado lote. Esto también permite reducir la posibilidad de quedarnos en un mínimo local.nso del gradiente

Es un proceso reiterativo para minimizar una función de pérdida, un dato o coste que es
la media respecto a todos los puntos para ello que se hace es el punto de partida
aleatoria de la función. Luego se calcula la derivada que es el gradiente de la función
en ese punto, con la derivada que podemos conocer la pendiente de la función de dicho
punto, si la derivada es positiva la función aumenta en dicha dirección y el algoritmo
se moverá en la dirección opuesta y si es negativo vamos en esa dirección. Que sea
estocástico implica que se elige de forma aleatoria un punto por paso ya que hacerlo
para todo sería costoso computacional mente aunque realmente se toma un subconjunto
llamado lote. Esto también permite reducir la posibilidad de quedarnos en un mínimo
local.

### 4.3. Regresión logística

### 4.4. Naive Bayes

### 4.5. Árboles de Decisión

#### 4.5.X. Random Forest

Random Forest es una técnica de ensamblado basada en árboles de decisión que mejora la
capacidad de generalización de estos últimos. Aunque los árboles de decisión clásicos
son fácilmente interpretables y eficientes en el ajuste a los datos de entrenamiento,
presentan una alta varianza que los hace poco robustos frente a nuevas muestras. Random
Forest soluciona esta limitación mediante un enfoque basado en el aprendizaje conjunto
de múltiples árboles de decisión.

El proceso de construcción de un modelo Random Forest se compone de tres etapas
fundamentales:

1. **Generación de conjuntos de entrenamiento mediante bootstrap**: A partir del
   conjunto de datos original, se crean múltiples subconjuntos de entrenamiento mediante
   muestreo aleatorio con reemplazo. Este procedimiento se conoce como _bootstrap
   sampling_. Como consecuencia, algunas observaciones pueden repetirse dentro de un
   subconjunto, mientras que otras no serán seleccionadas.

2. **Construcción de árboles de decisión**: Cada subconjunto generado se utiliza para
   entrenar un árbol de decisión independiente. A diferencia del procedimiento habitual,
   en cada división del árbol se selecciona aleatoriamente un subconjunto de
   características (_features_) en lugar de utilizar todas. Esta estrategia introduce
   diversidad entre los árboles y reduce la correlación entre ellos, lo que mejora el
   rendimiento general del modelo.

3. **Agregación de predicciones (bagging)**: El término _bagging_ (de _bootstrap
   aggregating_) hace referencia a la combinación de múltiples modelos entrenados sobre
   diferentes subconjuntos de datos. En Random Forest, esto se implementa promediando
   (para regresión) o votando (para clasificación) las predicciones generadas por cada
   árbol.

Durante el entrenamiento, algunas muestras no se utilizan en la construcción de un árbol
determinado. Estas observaciones, conocidas como _out-of-bag samples_, se emplean para
evaluar el rendimiento del modelo de manera interna, sin necesidad de un conjunto de
validación adicional. Al calcular el porcentaje de muestras _out-of-bag_ clasificadas
incorrectamente por el conjunto de árboles, se obtiene el llamado _out-of-bag error_,
que actúa como una estimación fiable del error de generalización.

Por último, el número de características consideradas en cada división puede ajustarse
como hiperparámetro del modelo. Este control permite optimizar el equilibrio entre sesgo
y varianza, mejorando la precisión y robustez del Random Forest frente a los árboles de
decisión individuales.

### 4.6. Máquina de Vectores de Soporte

### 4.7. XGBoost

# XGBoost Regressor

- Suele usar por defecto **loo → MSE**

## Funciona:

1. **Toma un grado de la función**  
   **toma la raíz**  
   **2da de pérdida**

   $$\text{MSE} = \frac{1}{n} \sum\_{i=1}^{n} (y_i - \hat{y}\_i)^2$$

2. **Calcula la desviación estándar**

   $$
   \text{desv} = \sqrt{\frac{1}{n} \sum\_{i=1}^{n} (y_i - 2\hat{y}\_i +
   \hat{y}\_i)^2}
   $$

3. **Calcula la desviación cuadrática**  
   $$\text{desv}^2 = \frac{1}{n} \sum\_{i=1}^{n} (y - 2\hat{y})^2$$

4. **Simplifica**  
   $$\text{Similarity} = \sqrt{G^2}$$

5. **Regula la desviación cuadrática**  
   $$\text{regulación cuadrática}$$

6. **Hace que la función de pérdida**  
   $$\text{H} = \text{desv} \cdot \text{regulación cuadrática}$$

7. **Mide que la función de pérdida**  
   $$\text{Mide que la función de pérdida}$$

8. **Mide que la función de desviación cuadrática**  
    $$\text{Mide que la función de desviación cuadrática}$$ Otra guía de la "ganancia"
   que es la diferencia de  
   similaridad entre modos difíciles divisionales.

gana = Similarity, izq + Similarity der - Similarity, H  
Se diga la división con mayor ganancia.

---

### Por cuartiles. martiles

→ padecas obtener un rango de valores → obteniendo por ejemplo un valor bajo - medio y
alto, representados como el cuartil P  
0.1, 0.5, 0.9, respectivamente,  
ocupando un rango del 80% de 0.9 a 0.4.

$$
L_q (y, \hat{y}) = \begin{cases} q \cdot (y - \hat{y}) & \text{si } y > \hat{y} \\
(1 - q) \cdot (y - \hat{y}) & \text{si } y < \hat{y} \\ \text{quartil} \end{cases}
$$

→ función de pérdida.

---

\frac{\partial L_q}{\partial y} =

\begin{cases} q \cdot (-1) = -q & \text{si } y > \hat{y} \\ (1 - q) \cdot (1 - \hat{y})
= 1 - q & \text{si } y < \hat{y} \end{cases}$$

→ Tanto la máxima casi como contable pues no es suave, la segunda  
derivada.

→ Sign. mundo el mismo paso de la similaridad y  
ganancia.

→ 45 (a (0.5) = mediana → unido borde de protección.

1. valor del modelo: modelo del árbol: \( \alpha \)

- \( HSE \rightarrow w = -\frac{\sum g_i}{\sum h_i + \lambda} \)

- \( \text{cuantité del valor target de ese modelo} \)

- \( \text{cuantité de ajuste} \)

- \( \text{modelo} = \times 6 \text{Regresion ( objective = "reg: quantidade")} \)

- \( \text{cuantité de ajuste - de ajuste} \)

- \( \text{cuantité de ajuste - de ajuste - de ajuste} \)

- \( \text{cuantité - de ajuste - de ajuste - de ajuste} \)

- \( \text{Entrada: un modelo, por cantidad} \)
-

## 5. Algoritmos de agrupación

### 5.1. Tipos de algoritmos de agrupación

#### 5.1.1. Métodos basados en particiones

un ejemplo de método de cluster bien basado en particiones es K-means, lo que hace es
una inicialización de cada puntos suponiendo el centro y de de los clusters luego
calcula la distancia del resto de puntos del conjunto de datos al centro y de y realiza
una asociación al centro y luego los datos asociados a un centro idea se promedian para
calcular la nueva posición del centro vida y este proceso se repite hasta un número de
pasos determinado o cuando no ya no se producen cambios en el centro.

El centro del cluster se representa con la letra mu si Aika igualados pues un centro y
de muy uno y otro será mudos lo que se hace es para cada índice que empieza desde uno
hasta M donde M sería la cantidad de cluster que vamos a considerar pues tengo el centro
idea del cluster y que es el índice del uno hasta acá del centro y del cluster cercano
al punto XV que es lo que hace es una asignación de puntos al centro edad y luego lo que
calculamos es la mínima distancia que existe entre el punto y y la media del cluster K y
se aplica la distancia euclídeo luego para cada valor de ca hasta llegar a acá se
calcula la media beca que es la media de puntos asignados al cluster y luego se aplica
la función de coste que es promediar la distancia euclidiana entre todas las muestras
del conjunto de datos entonces en cada iteración de Camins la función de coste debería
reducir y con ello podemos comparar el valor actual con el anterior para saber si el
modelo converge o no. El número de grupo busca tiene que ser menor a M que es el número
de muestras para buscar la mejor inicialización podemos inicializar Camins de forma
aleatoria una vez convergida cada iteración. Se calcula la función de coste de cada
iteración y nos quedamos con la iteración con menor coste, lo que permite evitar mínimos
no óptimos algunos métodos para elegir K pues es el método del codo por ejemplo que es
el punto donde crece más la función de coste es decir todo lo que haces es medir la
función de coste para un número de clusters y te quedas con ese punto donde la
convergencia es mayor, que es donde se produce un codo.

#### 5.1.2. Métodos basados en jerarquías

Ejemplo: Clustering jerárquico aglomerativo

#### 5.1.3. Métodos basados en densidad

Ejemplo: DBSCAN

#### 5.1.4. Métodos basados en modelos

Ejemplo: Gaussian Mixture Models (GMM)

#### 5.1.5. Métodos basados en grafos

Ejemplo: Spectral Clustering

### 5.2. Mecanismos en la elección de grupos

## 6. Métodos de comparación de modelos

### 6.1. Clasificación

#### 6.1.1. Matrices de confusión

La **matriz de confusión** es una herramienta clave para evaluar la capacidad de un
modelo de clasificación. Relaciona los valores predichos por el modelo con los valores
reales, organizándolos en cuatro categorías:

- **True Positives (TP)**: instancias positivas correctamente clasificadas.
- **False Positives (FP)**: instancias negativas clasificadas incorrectamente como
  positivas.
- **True Negatives (TN)**: instancias negativas correctamente clasificadas.
- **False Negatives (FN)**: instancias positivas clasificadas incorrectamente como
  negativas.

La diagonal principal de la matriz (TP y TN) refleja la tasa de aciertos del modelo;
valores más elevados indican un mejor desempeño.

#### 6.1.2.ROC y AUC

Rock permite recopilar información sobre la sensibilidad, lo que se conoce como el
recall y la especificidad, lo que se conoce como especificidad. Si tenemos una gráfica
en un plano en dos dimensiones con rangos de valores comprendidos entre cero y uno en
ambos de sus ejes el eje X y llegáis y trazamos una línea recta que es lineal. Esa línea
recta es la separación que existe entre un clasificador aleatorio, es decir, que tiene
una proporción Gual de falsos positivos que de verdaderos positivos donde el donde los
valores que están por encima de esa diagonal principal son modelos mejores más cercanos
a la perfección y por debajo de esa diagonal, pues es un rendimiento malo aleatorio.

El eje x es el False Positive Rate y el eje y es el True positive rate, que un modelo
sea mejor que otro dependerá del ratio de falsos positivos o verdaderos positivos
dependiendo de lo que sea más importante.

AUC, area under the curve, mide el área bajo la curva de ROC a mayor valor mejor. sE
UTILIZA PARA COMPARAR MODELOS, CON DIFERENTES roc.

Tambien podemos sustituir FPR por precision, que es la proporcion de resultados
positivos correctamente clasificados. Mas utilizado cuando el conjunto de datos no esta
balanceado, ya que es una medida mas equilibrada.

### 6.2. Regresión

## 7. Métodos para la reducción de la dimensionalidad

### 7.1. PCA

### 7.2. T-SNE

### 7.3. UMAP

### 7.4. Auto Encoders

- **Autoencoders**

$$g\_{\phi}(z|x) $$ : estimar la probabilidad posterior, probabilística encoder.

$$p\_{\theta}(x|z) $$ : probabilidad de generar la muestra de datos, verdadera dada el
código latente, deseos probabilístico decoder.

$$g\_{\phi}(.) $$ : encoder, parauentrizado con \(\phi\).

$$g\_{\theta}(.) $$ : decoder, parauentrizado con \(\theta\).

Encoder:

$$
\begin{bmatrix} x \\ y*{\phi}(x) \end{bmatrix} \rightarrow \begin{bmatrix} z \\
y*{\theta}(z) \end{bmatrix} \rightarrow \begin{bmatrix} p*{\theta}(z) \\ y*{\phi}(x)
\end{bmatrix}
$$

$$1 \times x \times x^T $$

- Tienden al overfitting. → Agregamos ruido gaussiano o eliminamos partes de la imagen
  de forma cstocástica. →  
  DropOut (DropBlock ZLD) → Alora SpatialDropout

- Spane: Autoencoder, penúteu/fuerzan al modelo a tener un uo reducido de neuronas
  activadas al mismo tiempo. → k-spane AE, la espaciad solo se mantiene en las
  activaciones k más activas. el resto se ponen a 0.

- **Contractive**  
  Autoencoder: penalizada a la representación, el ser muy sensible a los datos de
  entrada.

$$J*f(x) = \sum*{i,j} \left( \frac{\partial h_j(x)}{\partial x_i} \right)^2$$

- La sensibilidad se rinde con la máxima flexibilidad de la matriz Jacobiana de las
  activaciones del encoder con respecto a la entrada.
-

## 8. Métodos para la imputación de datos

La imputación de datos es una técnica fundamental en la preparación de datos,
especialmente cuando se enfrentan valores faltantes en un conjunto. Dependiendo del tipo
de variable (numérica o categórica), se aplican diferentes estrategias para completar
los valores ausentes de manera coherente y eficiente.

### 8.1. Imputación simple

Para variables **numéricas**, se emplean habitualmente medidas de tendencia central como
la **media** o la **mediana**. No obstante, la **mediana** es preferida en contextos
reales debido a su mayor robustez frente a valores atípicos o fuera de distribución. La
decisión entre usar media o mediana puede fundamentarse en un análisis estadístico
preliminar, como el estudio de la función de distribución acumulada (CDF) y el **rango
intercuartílico (IQR)**, que corresponde a la diferencia entre el percentil 75 y el
percentil 25. Esta evaluación permite identificar valores anómalos y decidir si deben
eliminarse o si la imputación debe ajustarse a una medida más robusta como la mediana.

Para variables **categóricas**, la imputación más común se realiza mediante la **moda**,
es decir, el valor más frecuente en la columna correspondiente. Cabe destacar que estas
imputaciones se aplican **por columna**, es decir, por cada característica (_feature_)
del conjunto de datos.

### 8.2. Imputación basada en vecinos

Una estrategia más avanzada es el uso de métodos basados en los **vecinos más
cercanos**, como el algoritmo **k-Nearest Neighbors (k-NN)**. Este enfoque consiste en
identificar, para una muestra con valores faltantes, las muestras más similares
(vecinas) utilizando métricas de distancia, como la distancia euclidiana. Una vez
determinadas las _k_ muestras más cercanas, el valor faltante se imputa en función de
las características de esas vecinas, por ejemplo, mediante la media, la mediana o la
moda de los valores presentes en ese grupo. Esta técnica permite imputar valores de
forma contextualizada, mejorando la precisión respecto a métodos globales.

### 8.3. Imputación con modelos predictivos

#### 8.3.1. MissForest

Un enfoque aún más sofisticado es **MissForest**, que emplea algoritmos de aprendizaje
automático como **Random Forest** para imputar valores faltantes. El proceso consiste
en:

1. Realizar una imputación inicial de los valores faltantes utilizando técnicas simples
   (media, mediana o moda según el tipo de variable).
2. Entrenar un modelo Random Forest con las características completas para predecir los
   valores ausentes de cada característica incompleta.
3. Actualizar los valores imputados con las predicciones obtenidas.
4. Repetir iterativamente el proceso hasta que se alcanza la convergencia o un número
   máximo de iteraciones.

MissForest es especialmente útil en contextos donde las relaciones entre variables son
complejas y no lineales, ofreciendo un balance entre precisión y robustez.

En resumen, la selección del método de imputación más adecuado depende de la naturaleza
de los datos, del patrón de ausencia y del nivel de precisión requerido en el análisis
posterior.

## X. Sensibilidad y especificidad

Dos métricas derivadas de la matriz de confusión permiten evaluar la capacidad del
modelo para identificar correctamente las clases positivas y negativas:

- **Sensibilidad (Recall)**: mide la proporción de verdaderos positivos respecto al
  total de instancias realmente positivas. Se calcula como:

$$\text{Sensibilidad} = \frac{TP}{TP + FN}$$

Un valor alto indica que el modelo identifica correctamente la mayoría de las instancias
positivas.

- **Especificidad**: mide la proporción de verdaderos negativos respecto al total de
  instancias realmente negativas. Se calcula como:

$$\text{Especificidad} = \frac{TN}{TN + FP}$$

Un valor alto indica que el modelo discrimina adecuadamente las instancias negativas.
Ambas métricas se pueden expresar en porcentaje multiplicando por cien, lo que facilita
la interpretación de los resultados.

Por ejemplo, si tenemos un algoritmo de **regresión logística** donde estamos
prediciendo: **tiene enfermedad cardíaca** y **no tiene enfermedad cardíaca**, con una
**matriz de confusión de 2 × 2**, donde el primer valor es **145**, luego **25**, que
corresponderían a la primera fila (primera y segunda columna), y luego **30** y **100**
para los elementos de la primera y segunda columna de la segunda fila, tendríamos la
siguiente matriz:

|                        | Predice enfermedad | Predice no enfermedad |
| ---------------------- | ------------------ | --------------------- |
| **Enfermedad real**    | 145 (TP)           | 25 (FN)               |
| **No enfermedad real** | 30 (FP)            | 100 (TN)              |

El **recall** (o sensibilidad) se calcula como:

$$
Recall = \frac{TP}{TP + FN}
$$

Sustituyendo los valores:

$$
Recall = \frac{145}{145 + 25} = \frac{145}{170} = 0.8529 \; (85.29\%)
$$

La **especificidad** se calcula como:

$$
Specificity = \frac{TN}{TN + FP}
$$

Sustituyendo los valores:

$$
Specificity = \frac{100}{100 + 30} = \frac{100}{130} = 0.7692 \; (76.92\%)
$$

De ahí tendríamos que el **85.29%** de los pacientes que presentan enfermedad han sido
clasificados correctamente y el **76.92%** de los pacientes que no presentan enfermedad
han sido clasificados correctamente. Esto nos permitiría compararlos directamente con
las métricas obtenidas de otros modelos como un **árbol de decisión**, donde por ejemplo
podríamos elegir la regresión logística si detectar pacientes **sin enfermedad** es más
importante, o elegir el árbol de decisión si detectar pacientes **con enfermedad** es
más importante. Esto al final es una **matriz de confusión**, que para matrices de
confusión de mayor tamaño se interpreta de la misma forma, pero calculando la
**sensibilidad (recall)** y la **especificidad** para cada categoría. Supongamos una
matriz de confusión con **3 clases A, B y C**, con los siguientes valores numéricos
inventados:

| Real \ Predicho | A   | B   | C   |
| --------------- | --- | --- | --- |
| **A**           | 50  | 5   | 10  |
| **B**           | 8   | 45  | 7   |
| **C**           | 6   | 9   | 40  |

En esta matriz:

- Los valores de la **diagonal principal** representan las clasificaciones correctas.
- Las filas representan la **clase real**.
- Las columnas representan la **clase predicha**.

---

### Cálculo del Recall (Sensibilidad)

El recall mide qué proporción de elementos de una clase real ha sido correctamente
clasificada.

#### Recall de la clase A

$$
Recall_A = \frac{A_{AA}}{A_{AA} + A_{BA} + A_{CA}}
$$

Sustituyendo valores:

$$
Recall_A = \frac{50}{50 + 8 + 6} = \frac{50}{64} = 0.7813 \; (78.13\%)
$$

#### Recall de la clase B

$$
Recall_B = \frac{A_{BB}}{A_{BB} + A_{AB} + A_{CB}}
$$

$$
Recall_B = \frac{45}{45 + 5 + 9} = \frac{45}{59} = 0.7627 \; (76.27\%)
$$

#### Recall de la clase C

$$
Recall_C = \frac{A_{CC}}{A_{CC} + A_{AC} + A_{BC}}
$$

$$
Recall_C = \frac{40}{40 + 10 + 7} = \frac{40}{57} = 0.7018 \; (70.18\%)
$$

---

### Cálculo de la Especificidad

La **especificidad** mide qué proporción de elementos que **no pertenecen** a una clase
han sido correctamente clasificados como no pertenecientes a dicha clase.

La fórmula general es:

$$
Specificity_X = \frac{TN_X}{TN_X + FP_X}
$$

---

#### Especificidad de la clase A

- **FP_A**: valores predichos como A pero que no son A  
  $FP_A = 8 + 6 = 14$
- **TN_A**: todos los valores que no están ni en la fila A ni en la columna A

$$
TN_A = 45 + 7 + 9 + 40 = 101
$$

$$
Specificity_A = \frac{101}{101 + 14} = \frac{101}{115} = 0.8783 \; (87.83\%)
$$

---

#### Especificidad de la clase B

- **FP_B**: $5 + 9 = 14$
- **TN_B**:

$$
TN_B = 50 + 10 + 6 + 40 = 106
$$

$$
Specificity_B = \frac{106}{106 + 14} = \frac{106}{120} = 0.8833 \; (88.33\%)
$$

---

#### Especificidad de la clase C

- **FP_C**: $10 + 7 = 17$
- **TN_C**:

$$
TN_C = 50 + 5 + 8 + 45 = 108
$$

$$
Specificity_C = \frac{108}{108 + 17} = \frac{108}{125} = 0.8640 \; (86.40\%)
$$

---

En resumen, para matrices de confusión multiclase, el **recall** se calcula utilizando
la **columna de la clase de interés**, mientras que la **especificidad** se obtiene
considerando todos los valores que no pertenecen a dicha clase, lo que permite evaluar
el rendimiento del modelo para cada categoría de forma individual.

Al final, el **recall** es el elemento de la **diagonal principal** de la columna $X$
dividido entre la suma del elemento diagonal de la columna $X$ más el resto de los
elementos de esa columna. Esto nos permite obtener los **verdaderos positivos**, que son
los valores predichos como $X$ y que realmente pertenecen a $X$.

Para obtener los **verdaderos negativos**, se deben considerar todas las columnas
diferentes de $X$ y todas las filas diferentes de $X$. Para los **falsos positivos**, se
observa únicamente la fila de $X$ y se excluye la columna de $X$, ya que representan los
valores predichos como $X$ pero que en realidad pertenecen a otra clase.

## X. Sistermas de deteccion de anomalias

la premisa está en que se entrenan contratos no anómalos estos sistemas de anomalías.
Por ejemplo podemos utilizar este imágenes basados en la densidad que consiste en
calcular la probabilidad de que un dato esté o sea visto en el conjunto de datos donde
se tiene el centro de un conjunto de datos y se va a calcula la baja, probabilidad o
alta probabilidad de que está basado en la distancia con respecto al centro entre dos
características por ejemplo y lo que se hace es establecer un umbral de probabilidad.
También podemos aplicar métodos basados en distribuciones normales o cauciones, donde lo
que hacemos es modelar la media de la varianza para cada parámetro a partir de una
distribución Gaussiana.

También sería bueno incluir datos anómalos en el set de validación. Aquí va muy bien
utilizar validación cruzada.

Detector de anomalías con Versus el aprendizaje supervisado el detector de anomalías
muchos tipos de posibles anomalías que desconocemos y que son diferentes a los vistos.
No hacemos la suposición de que los datos nuevos que nos llegan están dentro de la misma
distribución un clasificado supervisado por ejemplo tiene muchos positivos y negativos y
se espera que las muestras a futuro sigan una distribución similar, pero puede darse el
caso de que las características no tengan una forma gaussiana entonces hay que aplicar
transformaciones.

Algunas transformaciones escoger los datos de entrada y aplicar el logaritmo de los
datos en logaritmo +1 constante C o el organismo de una función exponencial de los datos
donde por ejemplo se llame pues son valores que deberíamos probar para hacer que
nuestros datos tengan una distribución normal existe lo que se conoce como flujos de
normalizadores, que son modelos generativos invertibles que transforma en una
distribución de datos compleja a una conocida como la distribución normal preservando la
dimensión alidada de los datos. Con ellos podemos detectar anomalías.

# Flow normalización

**Enfoque**: en el modelado de imágenes.

→ Flujo de normalización: son modelos linealidad.

Se considera un modelo generativo pero a diferencia de un VAE o GAN, este apriete la
función de densidad de probabilidad de un dato \( x \). Modela la distribución \( p(x)
\) real.

**Objetivo**: minimizar el negativo log-likelihood.

$$X \sim p(x)$$

$$f(x) : x \rightarrow z$$

**Función**: que mapea \( x \) a \( z \).

**Donde**: \( f \) es biyectiva, es decir:

$$f : x \rightarrow z$$

\( x \) ha de tener el misma altura que \( x \).

**Especie**: latente de igual dimensión.

$$X \sim p(x)$$

$$f^{-1}(x)$$

**Fluxo**:

$$X \sim p(x)$$

**Inversa**:

$$f^{-1}(x)$$

**Cerra pondencia**:

**Para ello, utilizamos:**

**Cerra regla del cambio de variables**:

**Dada una distribución (prior) \( p(z) \), \( p(z) \), \( p(z) \), Gaussian, y una
función invertible \( f \), podemos determinar**:

**\( p(x) \)** Por definición, el área bajo la curva para una  
función de densidad de probabilidad es 1.

$$\int*{x_1}^{x_2} P(x) \, dx = \int*{z_1}^{z_2} P(z) \, dz = 1$$

$$P(x) \, dx = \int\_{z_1}^{z_2} P(z) \frac{dz}{dx}$$

$$P(x) = \int\_{z_1}^{z_2} P(z) \frac{dz}{\frac{dP}{dx}}$$

Probabilidad de \( x \) en el espacio.

Para un factorable minimizando el logaritmo.

$$\log P(x) = \log P_2^{(x)} + \log \left| \frac{dP}{dx} \right|$$

Reducimos hacer la función \( P \) más compleja,  
con el modelo de la distribución de  
probabilidad de \( x \), pero más complejo de  
obtener \( P_2^{(z)} \). Pero, podemos unir múltiples  
funciones \( P_2^{(z)} \) invertibles aprendibles para obtener

$$z_0 \xrightarrow{(z_0)} z_k = x$$

Una función con una distribución Gaussiana,  
a la que le aplicamos nuevas funciones  
invertibles.

## X. Sistemas de recomendacion

podríamos tener un conjunto de usuarios con puntuaciones de películas, pero puede que
ciertos usuarios no hayan visto todas las películas entonces podríamos estimar la
puntuación de las películas no vistas utilizando algoritmos como la colaboración, el
filtrado por colaborativo por filtrado basado en el contenido.

# Bayesian Neural Networks

**Bayesian Neural Networks (BNNs)** represent a paradigm that integrates Bayesian
inference into deep learning models. Unlike traditional neural networks, where
parameters (weights and biases) are fixed values determined through optimization
algorithms like backpropagation and gradient descent, BNNs model these parameters as
probability distributions. This conceptual shift allows capturing the inherent
uncertainty in both the model's parameters and its predictions, offering a more
comprehensive understanding of the model's limitations and reliability.

![image](https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fjonascleveland.com%2Fwp-content%2Fuploads%2F2023%2F07%2FBayesian-Network-vs-Neural-Network.png&f=1&nofb=1&ipt=79ec39d4258da81fe61c9d9395d92f984259b951150c23451b0892cd578e92e4)

## Theoretical Foundations of Bayesian Inference

Bayesian inference is based on **Bayes' Theorem**, which provides a mathematical
framework for updating beliefs about a model when new observations become available. To
understand this concept, it's helpful to consider the process of human learning:
initially, we have prior knowledge about a phenomenon, and when we observe new data, we
update that knowledge to gain a more accurate understanding.

Bayes' Theorem is mathematically expressed as:

$$
P(\theta | D) = \frac{P(D | \theta) P(\theta)}{P(D)}.
$$

This equation can be interpreted as a rule for updating knowledge, where each component
represents a specific aspect of the learning process:

- **$P(\theta)$ – Prior Knowledge (Prior Distribution)**: Represents the initial beliefs
  about the model parameters before observing the data. For example, if we want to
  predict a person's height, the prior might state that most heights lie between 1.50
  and 2.00 meters, with an average around 1.70 meters.
- **$P(D | \theta)$ – Data Compatibility (Likelihood)**: Measures how likely the
  observed data are given a specific set of parameters. Continuing the previous example,
  if the model parameters suggest an average height of 1.80 meters, the likelihood
  evaluates how compatible the observed heights are with that prediction.
- **$P(D)$ – Normalization (Evidence)**: Acts as a normalization factor ensuring that
  the posterior distribution sums to one (since probabilities must lie between 0 and 1),
  satisfying the properties of a valid probability distribution. This term represents
  the total probability of observing the data under all possible parameter values.
- **$P(\theta | D)$ – Updated Knowledge (Posterior Distribution)**: This is the final
  result of the Bayesian process: the updated beliefs about the parameters after
  considering both the prior knowledge and the observed data. The posterior distribution
  combines prior information with empirical evidence to provide a more informed estimate
  of the parameters.

## Probabilistic Parameter Modeling in BNNs

In a BNN, each weight and bias is represented by a **probability distribution**,
typically a normal distribution with mean 0 and standard deviation 1, denoted as
$\mathcal{N}(0, 1)$. The training process does not aim to estimate a single value for
each parameter but rather to adjust the **posterior distribution** that best explains
the observed data.

This approach requires parameterizing the distributions through the mean and standard
deviation, updating them iteratively during training. The goal is to learn a **posterior
distribution $P(\theta | D)$** over the parameters $\theta$ given the data $D$, where:

- The **prior distribution** $P(\theta)$ typically assumes a standard Gaussian form,
  representing prior knowledge about the parameters.
- The **posterior distribution** $P(\theta | D)$ is adjusted during training and can
  differ significantly from the prior, shifting to reflect the knowledge gained from the
  data.

## Approximation Methods for the Posterior Distribution

Since exact computation of the posterior distribution is computationally intractable in
most practical cases, approximate inference techniques are employed:

- **Variational Inference**: Approximates the posterior distribution with a simpler
  distribution $q(\theta)$, optimizing the Kullback-Leibler (KL) divergence between
  $q(\theta)$ and $P(\theta | D)$. This method offers computational efficiency and
  scalability for large models, making it the most common choice in practical
  applications.

- **Markov Chain Monte Carlo (MCMC)**: Sampling-based methods that approximate the
  posterior by generating multiple samples. Although computationally more expensive,
  they provide more accurate approximations of the posterior and are useful when
  precision is prioritized over efficiency.

### ELBO Loss Function

Optimization in Bayesian Neural Networks is fundamentally based on maximizing the
Evidence Lower Bound (ELBO):

$$
\mathcal{L} = \mathbb{E}_{q(\theta)}[\log P(D | \theta)] - KL(q(\theta) || P(\theta)).
$$

This objective function balances two critical components that are essential for Bayesian
learning. The first component, known as the likelihood term
$\mathbb{E}_{q(\theta)}[\log P(D | \theta)]$, maximizes the probability of the observed
data under the approximate distribution $q(\theta)$. This component ensures that the
model maintains a good fit to the training data by encouraging the approximate posterior
to assign high probability to parameter values that explain the observed data well.

The second component, referred to as the regularization term
$KL(q(\theta) || P(\theta))$, minimizes the Kullback-Leibler divergence between the
approximate posterior distribution $q(\theta)$ and the prior distribution $P(\theta)$.
This component acts as a regularizing force that prevents overfitting by maintaining the
posterior distribution close to the prior when data is insufficient or ambiguous.

The KL divergence is formulated differently depending on the type of distribution. For
discrete distributions, the divergence is calculated as:

$$
KL(P || Q) = \sum_{x} P(x) \log \frac{P(x)}{Q(x)}.
$$

For continuous distributions, the divergence is expressed as an integral over the
parameter space:

$$
KL(P || Q) = \int_{-\infty}^{\infty} p(x) \log \frac{p(x)}{q(x)} dx.
$$

This duality in formulation allows the Bayesian framework to be applied in both discrete
and continuous spaces, providing flexibility in modeling different types of parametric
uncertainty. The continuous formulation is particularly relevant for BNNs, where the
parameters typically follow continuous distributions such as Gaussians, enabling the
framework to capture smooth variations in parameter uncertainty across the continuous
parameter space.

## Inference and Uncertainty Quantification

During the inference phase, a BNN generates predictions by repeatedly sampling from the
weight distribution. This process typically involves multiple independent inferences
(commonly between 50 and 1000 repetitions) for the same input, producing a set of
predictions that allows:

- Calculating the **mean** of the predictions as the final estimate.
- Determining the **variance** or standard deviation as a quantitative measure of the
  **associated uncertainty**.

This ability to quantify uncertainty is the main advantage of BNNs, providing insight
into the reliability of each individual prediction.

## Applications and Comparative Advantages

### Application Domains

BNNs are particularly valuable in contexts where uncertainty quantification is critical:

- **Biochemistry and drug discovery**: Risk and reliability assessment of new molecules.
- **Medical diagnosis**: Probabilistic estimation of critical diagnoses where
  uncertainty must be explicit.
- **Finance**: Risk assessment based on probabilistic predictions.
- **Robotics and reinforcement learning**: Adapting to dynamic environments under
  uncertainty.
- **Telecommunications**: Dynamic adjustment of network parameters considering
  environmental variability.

### Advantages over Deterministic Models

BNNs offer several advantages over traditional neural networks:

- **Formal uncertainty quantification**: Enables understanding of the model's
  limitations on new inputs, providing crucial information for decision-making in
  critical domains.
- **Effective regularization**: Prior distributions and KL divergence terms act as
  natural regularization mechanisms, significantly reducing the risk of overfitting.
- **Improved performance with limited data**: Prior knowledge acts as a guide when
  available data is scarce, improving the model's generalization.
- **Greater interpretability**: Facilitates analysis of prediction reliability and
  provides additional tools for informed decision-making, especially important in
  high-risk applications.

## Integration with Probabilistic Programming

BNNs naturally integrate with **probabilistic programming**, a paradigm that allows
complex statistical models to be described using declarative code. This integration
significantly broadens their applicability and facilitates implementation in systems
where explicit modeling of uncertainty is essential.

The combination provides a unified framework for developing applications that require
both the representational power of neural networks and the uncertainty modeling
capabilities of Bayesian inference.

## Mixture DEnsity mODELS

- **MIXTURE DENSITY MODEL (MDN)**

Obtener a la salida del modelo \( P(y|x) \), la salida del modelo será la descripción de
la distribución que modela los datos un target dado los datos.

Es una mezcla de distribuciones, guardadas.

Cada distribución:

- \( M_i \) media → centro
- \( \sigma_i^2 \) varianza → ancho
- \( w_i \) pesos → importancia/sequencia

**Likelihood**

\( KPIS \downarrow \)

- \( Z \downarrow \)
- \( N \downarrow \)

\( MDN \)

- \( \left\{ \begin{array}{l} M_i \\ \sigma_i^2 \\ w_i \end{array} \right\} \)
- \( \sum w_i = 1 \)

- Distribución
- \( \sum w_i = 1 \)

- Posterior Density

- Tenemos en realidad múltiples distribuciones:
  - Se deja como caja

$$x \sim MDN$$

$$x \in C$$

$$
MDN = \left\{ \begin{array}{l} M_1, \sigma_1^2, \mu_1 \\ M_2, \sigma_2^2, \mu_2 \\
\vdots \\ M_n, \sigma_n^2, \mu_n \end{array} \right\}
$$

- \( \text{MDN} \sim \text{MDN} \sim \text{MDN} \sim \text{SDM} \sim \text{SDM} \sim
  \text{SDM} \sim \)
- \( \text{MDN} \sim \text{MDN} \sim \)

- \( \text{MDN} \sim \text{MDN} \sim \)
- \( \text{MDN} \sim \text{MDN} \sim \sigma_1^2, \sigma_2^2, \mu_1, \mu_2 \)

- \( \text{MDN} \sim \text{MDN} \)

- \( \text{MDN} \sim \text{MDN} \)
- \( \text{MDN} \sim \text{MDN} \)
- **Sofmax**

- **Mixture Model**
  - \( W_i \sim \text{Distribución} \)
  - \( \text{MDN} \sim \text{MDN} \sim \text{Distribución} \)
  - \( \text{MD} \sim \text{MD} \sim \text{MD} \sim \text{MD} \)

$$
- \log(p(y|x)) = - \log \left( \sum\_{j=0}^{m} \exp(\log(n_j) + \log(p_j(x|y)))
\right)
$$

$$P(x) = P(N(\mu, \sigma^2)(x)) \quad \text{torch_distribution.Normal}(\mu, \sigma^2)$$

$$P(x) = P(N(\mu, \sigma^2)(x))$$

$$N(\mu, \sigma^2) \equiv \text{torch.distributions.Normal}(\mu, \sigma^2) = m$$

$$P(N(x)) \equiv m \cdot \log \text{prob}(x)$$

$$
\log(N(x)) \equiv \text{torch.log}(N(x)) \to \text{torch.log}(N(x) + 1 \in
\mathbb{R})
$$

$$\log(\text{torch. logsumexp}(N(x) + 1 \in \mathbb{R}))$$

$$\text{torch.logsumexp}(N(x) + 1 \in \mathbb{R}) = m \cdot \log \text{prob}(x)$$

$$\text{torch.logsumexp}(N(x) + 1 \in \mathbb{R})$$

$$\text{torch.logsumexp}(N(x) + 1 \in \mathbb{R})$$

Casi este tipo de modelos, podemos considerar que la incertidumbre de las predicciones
obtenidas,  
**Monte Carlo**  
**Monte Carlo**

- **Epistemia**: lo que el modelo no sabe
- **Reducible**: con más datos / complejidad
- **Aleatoria**: variabilidad en el entorno  
  (p.ej.)  
  Esto es lo que modelo  
  **principalmente**  
  **supremado**

- **Paramentrización**: del  
  **modelo**  
  **input**

---

**Modelo**

- **Epistemia** →  
  \( y = f(x) \)  
  (Relación directa con el modelo)

- **Alcatena** →  
  \( x = 1 \)  
  (Relación directa con lo dado)

---

En vez de predecir un valor, unico como la regresión obtenemos la \( y \) o que define a
cada distribución. Con ello, podemos estimar la incertidumbre.

$$
MON \quad \xrightarrow{N} \quad X (input) \rightarrow [NN] \rightarrow H (hidden,
reproductión)
$$

$$\xrightarrow{mixtura Model} $$  
modela la PDF de \( P(X) \)

---

**Vamos a tener una guanadora que que con ello podemos aproximar a la distribución de la
PDF con cierta precisión**  
**siendo una mezcla de coeficientes y parámetros.**

---

**MON**  
**Parametrización una**  
**Mixture Model**  
**GMM (Gaussian Mixture)**

- Para la varianza usar la función de activación ELU:

  $$
  A(z) = \begin{cases} z & z > 0
  \\ \alpha (e^z - 1) & z < 0 \end{cases}$$ modificada
  ELU(z) + 1 + 1e^{-15}
  $$

- Evita que la función de operación crezca mucho y suprida los datos con alta variancia.
  → Inestable

- ELU mantiene el comportamiento espacioso al reducir a un comportamiento lineal para
  valores más altos.

$$ELU(z) + 1 \rightarrow \text{Movemos ELU a la zona de los positivos}$$

$$ELU(z) + 1 + 1e^{-15} \rightarrow \text{El (optimo) para dar estabilidad}$$

- Para evitar cobro en los modelos y que ignore alguna componente (y, o, ...), teniendo
  mayor relevancia una distribución sobre otra podemos usar:

- **Weigtir reglamentario**: \( t_1, t_2 \)
- **Bias inutilización a partir de pre-calcular el centro de cada gausiana**

- **Sufitir**: \( \text{Sofmax} \rightarrow \text{Gumbel Sofmax} \), \( \text{Ja que
  puede generar distribuciónes más agresivas (llegó a aparecer, poner un 0, en
  componentes sin importancia) Sofmax es una suave}$$

# Bateria: Intervención

Ya tenemos el modelo, informamos un nuevo dato con el fin de conocer la componente
(distribución) a la que pertenece. Queremos:

$$p(h|x) \propto p(h) \cdot p(x|h) \quad (h, o z)$$

proba dado x  
probado dado h  
de tener h  
(embedding)  
de tener x

Si aparecen: Nada, podría ser:

1. Logaritmo de un valor cercano a cero.
2. Eficiencia con denominador.
3. Exponential de un valor muy grande, que de NaN.

??? info "Solución":

1. Gradiente chipping
2. Weight regularization
3. Batalla Normal output layer

Intervalles de calibración

Rango de valores que con cierta probabilidad (calibración) cuente el verdadero valor de
un parámetro desconocido. P.ej. un intervalo de calibración del 95% indica que al
repetir un experimento varias veces aproximadamente el 95% de los intervalos calculados
contienen el valor verdadero.

# Anuncio 2

- Tipicamente se anume que todos los valores (inputs) son igual de probables y que son
  continuos.
- La razón de valores no conocemos, por lo que no podemos calcular los valores de los
  datos.
- La razón de valores no conocemos, por lo que no podemos calcular los datos.
- La razón de valores no conocemos, por lo que no podemos hacer la PDF?

---

# Procedimiento 2

- Podemos obtener la media, varianza, percentiles.
- Intervalos de confianza.
- Si tenemos distribuciones en train,  
  comparaciones visuales,  
  comparativas de percatiles.

---

- **KL**
- **Perteque**
