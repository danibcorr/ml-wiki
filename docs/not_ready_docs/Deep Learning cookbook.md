# Deep Learning cookbook

Realizado por Daniel Bazo Correa.

# Índice

# 1. Conceptos teóricos

## Por qué Deep Learning

- Simplicity: Deep learning removes the need for feature engineering, replacing complex, brittle, engineering-heavy pipelines with simple, end-to-end trainable models that are typically built using only five or six different tensor operations.
- Scalability—Deep learning is highly amenable to parallelization on GPUs or TPUs,
so it can take full advantage of Moore’s law. In addition, deep learning models
are trained by iterating over small batches of data, allowing them to be trained on
datasets of arbitrary size. (The only bottleneck is the amount of parallel computational power available, which, thanks to Moore’s law, is a fast-moving barrier.)
- Versatility and reusability—Unlike many prior machine learning approaches, deep
learning models can be trained on additional data without restarting from scratch, making them viable for continuous online learning—an important property for very large production models. Furthermore, trained deep learning models are repurposable and thus reusable: for instance, it’s possible to take a deep learning model trained for image classification and drop it into a videoprocessing pipeline. This allows us to reinvest previous work into increasingly
complex and powerful models. This also makes deep learning applicable to
fairly small datasets.

## Formato datos en tensor

- Vector data—Rank-2 tensors of shape (samples, features), where each sample is a vector of numerical attributes (“features”)
- Timeseries data or sequence data—Rank-3 tensors of shape (samples, timesteps, features), where each sample is a sequence (of length timesteps) of feature vectors.
    
    ![Untitled](Deep%20Learning%20cookbook/Untitled.png)
    
- Images—Rank-4 tensors of shape (samples, height, width, channels), where each sample is a 2D grid of pixels, and each pixel is represented by a vector of values (“channels”)
- Video—Rank-5 tensors of shape (samples, frames, height, width, channels), where each sample is a sequence (of length frames) of images

## Normalización de los datos

The model might be able to automatically adapt to such heterogeneous
data, but it would definitely make learning more difficult. A widespread best practice
for dealing with such data is to do feature-wise normalization: for each feature in the
input data (a column in the input data matrix), we subtract the mean of the feature
and divide by the standard deviation, so that the feature is centered around 0 and has
a unit standard deviation

## Validaciones del entrenamiento

### K-fold validation

To evaluate our model while we keep adjusting its parameters (such as the number of
epochs used for training), we could split the data into a training set and a validation
set, as we did in the previous examples. But because we have so few data points, the
validation set would end up being very small (for instance, about 100 examples). As a
consequence, the validation scores might change a lot depending on which data
points we chose for validation and which we chose for training: the validation scores
Listing 4.25 Model definition
Because we need to instantiate
the same model multiple times,
we use a function to construct it.
116 CHAPTER 4 Getting started with neural networks: Classification and regression
might have a high variance with regard to the validation split. This would prevent us
from reliably evaluating our model.
The best practice in such situations is to use K-fold cross-validation

![Untitled](Deep%20Learning%20cookbook/Untitled%201.png)

It consists of splitting the available data into K partitions (typically K = 4 or 5), instantiating K identical models, and training each one on K – 1 partitions while evaluating
on the remaining partition. The validation score for the model used is then the average of the K validation scores obtained. In terms of code, this is straightforward

```python
k = 4
num_val_samples = len(train_data) // k
num_epochs = 100
all_scores = [] 
for i in range(k):
 print(f"Processing fold #{i}")
 val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
 val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
 partial_train_data = np.concatenate(
 [train_data[:i * num_val_samples],
 train_data[(i + 1) * num_val_samples:]],
 axis=0)
 partial_train_targets = np.concatenate(
 [train_targets[:i * num_val_samples],
 train_targets[(i + 1) * num_val_samples:]],
 axis=0)
 model = build_model() 
model.fit(partial_train_data, partial_train_targets,
 epochs=num_epochs, batch_size=16, verbose=0)
 val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
all_scores.append(val_mae)

>>> all_scores
[2.112449, 3.0801501, 2.6483836, 2.4275346]
>>> np.mean(all_scores)
2.5671294
```

The different runs do indeed show rather different validation scores, from 2.1 to 3.1.
The average (2.6) is a much more reliable metric than any single score—that’s the
entire point of K-fold cross-validation.

## Ambiguous features

Not all data noise comes from inaccuracies—even perfectly clean and neatly labeled
data can be noisy when the problem involves uncertainty and ambiguity. In classification tasks, it is often the case that some regions of the input feature space are associated with multiple classes at the same time

## Overfitting

Likewise, machine learning models trained on datasets that include rare feature
values are highly susceptible to overfitting

## Consideraciones en los datos

The arrow of time—If you’re trying to predict the future given the past (for example, tomorrow’s weather, stock movements, and so on), you should not randomly shuffle your data before splitting it, because doing so will create a
temporal leak: your model will effectively be trained on data from the future. In
such situations, you should always make sure all data in your test set is posterior
to the data in the training set.
 Redundancy in your data—If some data points in your data appear twice (fairly
common with real-world data), then shuffling the data and splitting it into a
training set and a validation set will result in redundancy between the training
and validation sets. In effect, you’ll be testing on part of your training data,
which is the worst thing you can do! Make sure your training set and validation
set are disjoint.

## Regularización

### L1 y L2

A simple model in this context is a model where the distribution of parameter values
has less entropy (or a model with fewer parameters, as you saw in the previous section). Thus, a common way to mitigate overfitting is to put constraints on the complexity of a model by forcing its weights to take only small values, which makes the
distribution of weight values more regular. This is called weight regularization, and it’s
done by adding to the loss function of the model a cost associated with having large
weights. This cost comes in two flavors:

- L1 regularization—The cost added is proportional to the absolute value of the
weight coefficients (the L1 norm of the weights).
- L2 regularization—The cost added is proportional to the square of the value of the
weight coefficients (the L2 norm of the weights). L2 regularization is also called
weight decay in the context of neural networks. Don’t let the different name confuse you: weight decay is mathematically the same as L2 regularization.

In Keras, weight regularization is added by passing weight regularizer instances to layers
as keyword arguments. Let’s add L2 weight regularization to our initial movie-review
classification model.

```python
from tensorflow.keras import regularizers
model = keras.Sequential([
 layers.Dense(16,
 kernel_regularizer=regularizers.l2(0.002),
 activation="relu"),
 layers.Dense(16,
 kernel_regularizer=regularizers.l2(0.002),
 activation="relu"),
 layers.Dense(1, activation="sigmoid")
])
model.compile(optimizer="rmsprop",
 loss="binary_crossentropy",
 metrics=["accuracy"])
history_l2_reg = model.fit(
 train_data, train_labels,
 epochs=20, batch_size=512, validation_split=0.4)
```

In the preceding listing, l2(0.002) means every coefficient in the weight matrix of
the layer will add 0.002 * weight_coefficient_value ** 2 to the total loss of the
model. Note that because this penalty is only added at training time, the loss for this
model will be much higher at training than at test time.

### Dropout

The core idea is that introducing noise in the output values of a layer can break up happenstance patterns that aren’t significant

![Untitled](Deep%20Learning%20cookbook/Untitled%202.png)

## Precisiones

Si las etiquetas son valores numericos podemos utilizas valores enteros sin signo como uint8 para reducir consumo de memoria. También podemos utilizar precisión mixta, Es el uso de tipos de punto flotante de 16 y 32 bits en un modelo durante el entrenamiento para que se ejecute más rápido y use menos memoria. Al mantener ciertas partes del modelo en los tipos de 32 bits para la estabilidad numérica, el modelo tendrá un menor tiempo de paso y se entrenará igual de bien en términos de las métricas de evaluación como la precisión. [Esta guía describe cómo usar la API de precisión mixta de Keras para acelerar sus modelos1](https://edgeservices.bing.com/edgesvc/chat?udsframed=1&form=SHORUN&clientscopes=chat,noheader,udsedgeshop,channelstable,&shellsig=fdee925ef17d6d1840504b627b7c1ac1c3f8a412&setlang=es&darkschemeovr=1#sjevt%7CDiscover.Chat.SydneyClickPageCitation%7Cadpclick%7C0%7C96ecd3e1-4e18-4788-b144-b60f60c2f8db%7C%7B%22sourceAttributions%22%3A%7B%22providerDisplayName%22%3A%22This%20guide...%22%2C%22pageType%22%3A%22html%22%2C%22pageIndex%22%3A1%2C%22relatedPageUrl%22%3A%22https%253A%252F%252Fwww.tensorflow.org%252Fguide%252Fmixed_precision%22%2C%22lineIndex%22%3A1%2C%22highlightText%22%3A%22This%20guide%20describes%20how%20to%20use%20the%20Keras%20mixed%20precision%20API%20to%20speed%20up%20your%20models.%22%2C%22snippets%22%3A%5B%5D%7D%7D). [Usar esta API puede mejorar el rendimiento en más de 3 veces en las GPU modernas, 60% en los TPU y más de 2 veces en las últimas CPU de Intel2](https://edgeservices.bing.com/edgesvc/chat?udsframed=1&form=SHORUN&clientscopes=chat,noheader,udsedgeshop,channelstable,&shellsig=fdee925ef17d6d1840504b627b7c1ac1c3f8a412&setlang=es&darkschemeovr=1#sjevt%7CDiscover.Chat.SydneyClickPageCitation%7Cadpclick%7C1%7C96ecd3e1-4e18-4788-b144-b60f60c2f8db%7C%7B%22sourceAttributions%22%3A%7B%22providerDisplayName%22%3A%22Using%20this...%22%2C%22pageType%22%3A%22html%22%2C%22pageIndex%22%3A1%2C%22relatedPageUrl%22%3A%22https%253A%252F%252Fwww.tensorflow.org%252Fguide%252Fmixed_precision%22%2C%22lineIndex%22%3A1%2C%22highlightText%22%3A%22Using%20this%20API%20can%20improve%20performance%20by%20more%20than%203%20times%20on%20modern%20GPUs%2C%2060%25%20on%20TPUs%20and%20more%20than%202%20times%20on%20latest%20Intel%20CPUs.%22%2C%22snippets%22%3A%5B%5D%7D%7D).

Por ejemplo en numpy podemos establecer que por defecto se utilice float32 asi:

```python
np.set_default_dtype(np.float32)
```

# 2. Extracción de características, feature extraction y feature engineering

# 3. Producción

## 3.1. Flujo de trabajo universal para Deep Learning

El flujo de trabajo está estructurado en tres partes:

1. **Definir la tarea:** Comprender el dominio del problema y la lógica comercial subyacente en lo que el cliente solicitó. Recopilar un conjunto de datos, comprender qué representa y elegir cómo medir el éxito en la tarea.
2. **Desarrollar un modelo:** Preparar tus datos para que puedan ser procesados por un modelo de aprendizaje automático, seleccionar un protocolo de evaluación del modelo y un punto de referencia simple para superar, entrenar un primer modelo con capacidad de generalización y que pueda sobreajustar, y luego regularizar y ajustar tu modelo hasta lograr el mejor rendimiento posible de generalización.
3. **Desplegar el modelo:** Presentar tu trabajo a las partes interesadas, enviar el modelo a un servidor web, una aplicación móvil, una página web o un dispositivo integrado, supervisar el rendimiento del modelo en la práctica y comenzar a recopilar los datos que necesitarás para construir el modelo de la próxima generación.

### 3.1.1. Definir la tarea

No puedes hacer un buen trabajo sin una comprensión profunda del contexto de lo que estás haciendo. Para ello, se deben seguir los siguientes pasos:

**Enmarcar el problema**

- ¿Cuáles serán tus datos de entrada? ¿Qué estás tratando de predecir?
- ¿Qué tipo de tarea de aprendizaje automático enfrentas? ¿Es clasificación binaria? ¿Clasificación multiclase? Regresión escalar? Regresión vectorial? ¿Clasificación multiclase y multietiqueta ¿Segmentación de imágenes? ¿Ranking? Puede ser que el aprendizaje automático ni siquiera sea la mejor manera de dar sentido a los datos, y debas usar algo más, como un análisis estadístico convencional.
- ¿Cómo se ven las soluciones existentes? Asegúrate de entender qué sistemas ya están en funcionamiento y cómo funcionan.
- ¿Hay restricciones particulares con las que necesitarás lidiar? Por ejemplo, podrías descubrir que la aplicación para la cual estás construyendo un sistema de detección de spam está estrictamente cifrada de extremo a extremo, por lo que el modelo de detección de spam deberá residir en el teléfono del usuario final y debe entrenarse con un conjunto de datos externo, restricciones de latencia, etc.

Hasta que tengas un modelo funcional, estas son solo hipótesis esperando ser validadas o invalidadas. No todos los problemas se pueden resolver con el aprendizaje automático.

**Recopilar un conjunto de datos**

Un buen conjunto de datos es un activo que merece cuidado e inversión. Si tienes 50 horas adicionales para gastar en un proyecto, es probable que la forma más efectiva de asignarlas sea recopilando más datos en lugar de buscar mejoras incrementales en el modelado.

**Invertir en una buena infraestructura de anotación de datos**
Tu proceso de anotación de datos determinará la calidad de tus objetivos, que a su vez determinarán la calidad de tu modelo. Considera cuidadosamente las opciones que tienes disponibles:

- ¿Deberías anotar los datos tú mismo?
- ¿Deberías usar una plataforma de crowdsourcing como Mechanical Turk para recopilar etiquetas?
- ¿Deberías usar los servicios de una empresa especializada en etiquetado de datos?

Externalizar puede ahorrarte tiempo y dinero, pero también quita control. El uso de algo como Mechanical Turk probablemente sea económico y se escale bien, pero las anotaciones pueden resultar bastante ruidosas.

Para elegir la mejor opción, considera las restricciones con las que estás trabajando:

- ¿Los anotadores de datos necesitan ser expertos en la materia, o cualquiera puede etiquetar los datos?

Si la anotación de datos requiere conocimientos especializados, ¿puedes capacitar a personas para hacerlo? Si no, ¿cómo puedes acceder a expertos relevantes?

- ¿Entiendes cómo los expertos crean las anotaciones? Si no lo haces, tendrás que tratar tu conjunto de datos como una caja negra, y no podrás realizar ingeniería de características manual; esto no es crítico, pero puede ser limitante.

Los modelos de aprendizaje automático solo pueden dar sentido a entradas que sean similares a lo que han visto antes. Por lo tanto, es crucial que los datos utilizados para el entrenamiento sean representativos de los datos de producción. Si es posible, recopila datos directamente del entorno donde se utilizará tu modelo. 

Un fenómeno relacionado del que debes estar al tanto es el cambio de concepto. Encontrarás cambios de concepto en casi todos los problemas del mundo real, especialmente aquellos que tratan con datos generados por el usuario. El cambio de concepto ocurre cuando las propiedades de los datos de producción cambian con el tiempo, lo que provoca que la precisión del modelo disminuya gradualmente. Un motor de recomendación de música entrenado en el año 2013 puede no ser muy efectivo hoy.

**Comprender tus datos**

- Si tus datos incluyen imágenes o texto en lenguaje natural, echa un vistazo a algunas muestras (y sus etiquetas) directamente.
- Si tus datos contienen características numéricas, es una buena idea trazar el histograma de los valores de las características para tener una idea del rango de valores y la frecuencia de los diferentes valores.
- ¿Faltan algunos valores para algunas características?
- Si tu tarea es un problema de clasificación, imprime el número de instancias de cada clase en tus datos. ¿Las clases están representadas aproximadamente de manera equitativa? Si no es así, deberás tener esto en cuenta.

**Elegir una medida de éxito**

Para controlar algo, necesitas poder observarlo. Para lograr el éxito en un proyecto, primero debes definir qué significa éxito. ¿Precisión? Precisión y exhaustividad?¿Tasa de retención de clientes?.

### 3.1.2. Desarrollar un modelo

Una vez que sepas cómo medir tu progreso, puedes comenzar con el desarrollo del modelo. Las cosas más difíciles en el aprendizaje automático son enmarcar problemas y recopilar, anotar y limpiar datos.

**Preparar los datos**

Muchas técnicas de preprocesamiento son específicas del dominio.

En general, no es seguro alimentar a una red neuronal datos que toman valores relativamente grandes (por ejemplo, enteros de varios dígitos, que son mucho mayores que los valores iniciales tomados por los pesos de una red) o datos heterogéneos (por ejemplo, datos donde una característica está en el rango de 0 a 1 y otra está en el rango de 100 a 200). Hacerlo puede desencadenar actualizaciones de gradiente grandes que evitarán que la red converja. Para facilitar el aprendizaje para tu red, tus datos deben tener las siguientes características:

- Tomar valores pequeños: típicamente, la mayoría de los valores deberían estar en el rango de 0 a 1.
- Ser homogéneos: todas las características deberían tomar valores en aproximadamente el mismo rango.

**Curación del conjunto de datos**

Si tus datos permiten interpolar suavemente entre muestras, podrás entrenar un modelo de aprendizaje profundo que generalice. El aprendizaje profundo es ajuste de curvas. Invertir más esfuerzo y dinero en la recopilación de datos casi siempre produce un rendimiento mucho mayor en comparación con gastar lo mismo en desarrollar un modelo mejor.

- Asegúrate de tener suficientes datos. Más datos generarán un modelo mejor.
- Minimiza los errores de etiquetado: visualiza tus entradas para detectar anomalías y revisa tus etiquetas.
- Limpia tus datos y maneja los valores faltantes.
- Si tienes muchas características y no estás seguro de cuáles son realmente útiles, realiza selección de características.

Una forma especialmente importante de mejorar el potencial de generalización de tus datos es la ingeniería de características.

**Ingeniería de características, feature engineering**

La ingeniería de características es el proceso de utilizar tu conocimiento sobre los datos y el algoritmo de aprendizaje automático en cuestión para mejorar el rendimiento del algoritmo aplicando transformaciones codificadas (no aprendidas) a los datos antes de que ingresen al modelo.

Esa es la esencia de la ingeniería de características: hacer un problema más fácil expresándolo de una manera más simple. Haz que la variedad latente sea más suave, más simple, mejor organizada. Hacerlo generalmente requiere entender el problema en profundidad.

Afortunadamente, el aprendizaje profundo moderno elimina la necesidad de la mayoría de la ingeniería de características, ya que las redes neuronales pueden extraer automáticamente características útiles de los datos sin procesar. ¿Significa esto que no tienes que preocuparte por la ingeniería de características siempre que estés utilizando redes neuronales profundas? No, por dos razones:

- Las características buenas aún te permiten resolver problemas de manera más elegante utilizando menos recursos. Por ejemplo, sería absurdo resolver el problema de leer un reloj utilizando una red neuronal convolucional.
- Las características buenas te permiten resolver un problema con mucha menos cantidad de datos. La capacidad de los modelos de aprendizaje profundo para aprender características por sí mismos depende de tener muchos datos de entrenamiento disponibles; si solo tienes algunas muestras, el valor informativo de sus características se vuelve crítico.

**Elegir un protocolo de evaluación**

- Mantener un conjunto de validación de retención: esta es la mejor opción cuando tienes muchos datos.
- Realizar validación cruzada K-fold: esta es la elección correcta cuando tienes muy pocas muestras para que la validación de retención sea confiable.
- Realizar validación cruzada K-fold iterada: esto es para realizar una evaluación de modelo altamente precisa cuando hay pocos datos.

**Superar un punto de referencia**

- Ingeniería de características: filtra características no informativas (selección de características) y utiliza tu conocimiento del problema para desarrollar nuevas características que probablemente sean útiles.
- Selección de prioridades de arquitectura correctas: ¿Qué tipo de arquitectura de modelo utilizarás? ¿Una red densamente conectada, una convolucional, una red neuronal recurrente, un Transformer? ¿Es el aprendizaje profundo incluso un enfoque adecuado para la tarea, o deberías usar algo más?
- Seleccionar una configuración de entrenamiento lo suficientemente buena: ¿Qué función de pérdida debes usar? ¿Qué tamaño de lote y tasa de aprendizaje?

### 3.1.3. Desplegar el modelo

**Explicar tu trabajo a las partes interesadas y establecer expectativas**

Las expectativas de los no especialistas hacia los sistemas de IA a menudo son poco realistas.

Para abordar esto, deberías considerar mostrar algunos ejemplos de los modos de falla de tu modelo (por ejemplo, muestra cómo se ven las muestras clasificadas incorrectamente, especialmente aquellas para las cuales la clasificación errónea parece sorprendente).

También podrían esperar un rendimiento a nivel humano, especialmente para procesos que anteriormente eran manejados por personas. La mayoría de los modelos de aprendizaje automático, porque están entrenados para aproximar etiquetas generadas por humanos, no llegan ni cerca.

Evita usar declaraciones abstractas como "El modelo tiene un 98% de precisión" (que la mayoría de las personas redondean mentalmente al 100%), y prefiere hablar, por ejemplo, sobre las tasas de falsos negativos y falsos positivos. Podrías decir, "Con esta configuración, el modelo de detección de fraude tendría una tasa de falsos negativos del 5% y una tasa de falsos positivos del 2.5%. Cada día, se marcarían como fraudulentas, en promedio, 200 transacciones válidas y se perderían en promedio 14 transacciones fraudulentas. Se detectarían correctamente en promedio 266 transacciones fraudulentas". Relaciona claramente las métricas de rendimiento del modelo con los objetivos comerciales.

También debes asegurarte de discutir con las partes interesadas la elección de parámetros clave de lanzamiento, por ejemplo, el umbral de probabilidad en el cual se debe marcar una transacción. Tales decisiones implican compensaciones que solo se pueden manejar con una comprensión profunda del contexto empresarial.

**Optimización del modelo en inferencia**

La optimización de tu modelo para la inferencia es especialmente importante al implementarlo en un entorno con restricciones estrictas en la potencia y la memoria disponibles, o para aplicaciones con requisitos de baja latencia.

Existen dos técnicas de optimización populares que puedes aplicar:

- **Poda de pesos (Weight pruning):** No todos los coeficientes en un tensor de pesos contribuyen por igual a las predicciones. Es posible reducir considerablemente el número de parámetros en las capas de tu modelo manteniendo solo los más significativos. Esto disminuye la huella de memoria y cómputo de tu modelo, con un pequeño costo en las métricas de rendimiento. Al decidir cuánta poda aplicar, tienes control sobre el equilibrio entre tamaño y precisión.
- **Cuantificación de pesos (Weight quantization):** Los modelos de aprendizaje profundo se entrenan con pesos de precisión única de punto flotante (float32). Sin embargo, es posible cuantificar los pesos a enteros con signo de 8 bits (int8) para obtener un modelo solo de inferencia que sea un cuarto del tamaño pero que se mantenga cerca de la precisión del modelo original.

## Acumulación de gradientes

Using gradient accumulation loops over your forward and backward pass (the number of steps in the loop being the number of gradient accumulation steps). A for loop over the model is less efficient than feeding more data to the model, as you’re not taking advantage of the parallelization your hardware can offer.

The only reason to use gradient accumulation steps is when your whole batch size does not fit on one GPU, so you pay a price in terms of speed to overcome a memory issue.

# Self supervised learning

El principal problema viene con la cantidad de datos que se requieren en los sistemas supervisados, la cantidad de tiempo que requiere etiquetar esos datos de manera manual, llevar sesgos a los datos (un etiquetador puede anotar algo como una etiqueta A mientras que otro puede anotarlo como B). La gran mayoría de los datos que tenemos son no etiquetados, por lo que obtener abstracciones que permiten obtener generalizaciones de los datos y con ello poder ir teniendo conocimiento de los datos. Este es el proceso de aprendizaje natural de las personas, coger idea y luego extrapolar el aprendizaje.

Muy relacionado con el aprendizaje multimodal de las personas. El aprendizaje requiere conocimiento previo, necesitamos redundancia de las señales.

Cuando se aprenden asociaciones en parejas de N eventos, necesitamos almacenar N^2 posibles probabilidades de un suceso C procedente de un suceso U. Si son indepedientes, podemos calcular las probabilidades por separado, donde la probabilidad total sería el producto de probabilidades.

Algunas ideas:

- Inpainting, reordenar puzzles (dar patches de imagenes no ordenadarlas y que sepa encontrar el orden correcto), que aprenda a colorear imagenes en escala de grises, predicciones de rotacion.
- Discriminación, contrastive learning, triple loss, etc
- Learning object keypoints
- Pseudo labelling - entrenar un modelo supervisado en datos etiquetados, predecir datos no etiquetados, re-entrenar el modelo con las pseaudo etiquetas.