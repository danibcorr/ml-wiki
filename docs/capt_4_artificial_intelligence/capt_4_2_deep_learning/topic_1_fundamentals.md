---
sidebar_position: 1
authors:
  - name: Daniel Bazo Correa
description: Fundamentos del Deep Learning.
title: Deep Learning
toc_max_heading_level: 3
---

## Bibliografía

- [Alice’s Adventures in a differentiable wonderland: A primer on designing neural networks (Volume I)](https://amzn.eu/d/3oYyuHg)
- [Deep Learning for Coders with Fastai and PyTorch: AI Applications Without a PhD](https://course.fast.ai/Resources/book.html)
- [Standford](https://youtube.com/playlist?list=PLoROMvodv4rNjRoawgt72BBNwL2V7doGI&si=TXQ-EA7J7sAwfKEQ).

## 1. Introducción al aprendizaje profundo

Antes de abordar el estudio del **aprendizaje profundo (_Deep Learning_)**, resulta
esencial comprender el concepto de inteligencia, una noción que, aunque aparentemente
simple, presenta una complejidad notable cuando se intenta definir con precisión. En
términos generales, la inteligencia puede entenderse como la **capacidad de procesar
información y utilizarla para tomar decisiones orientadas al logro de objetivos
específicos**.

Este concepto constituye el fundamento del campo de la **Inteligencia Artificial (IA)**,
disciplina que se dedica al desarrollo de técnicas y algoritmos capaces de reproducir
ciertos aspectos del comportamiento humano. La IA busca emular la inteligencia mediante
sistemas computacionales, permitiendo que las máquinas procesen información, se adapten
a diversos contextos y realicen predicciones para resolver problemas de manera autónoma,
minimizando la intervención humana.

Dentro de la IA se encuentra el **aprendizaje automático (_Machine Learning_)**, cuyo
propósito es permitir que las máquinas **aprendan a partir de la experiencia**, sin
necesidad de recibir instrucciones explícitas para cada tarea. En lugar de programar
manualmente cada paso del proceso, se diseñan algoritmos que **identifican patrones en
los datos**, ajustando sus parámetros internos con el objetivo de mejorar
progresivamente su rendimiento a medida que acumulan ejemplos. Este proceso de
aprendizaje se guía mediante una **función objetivo**, la cual mide el grado de
aproximación del sistema a la meta deseada.

El investigador Andrej Karpathy ha descrito este paradigma como “software 2.0”, en
contraposición al enfoque tradicional de programación. En el “software 1.0”, el
programador define de forma explícita las reglas y procedimientos que el sistema debe
ejecutar. En cambio, en el “software 2.0”, el programador proporciona **ejemplos,
recompensas o etiquetas** que guían el proceso de optimización del algoritmo,
permitiendo que el propio sistema descubra de manera implícita las reglas necesarias
para cumplir la tarea. Este cambio de paradigma marca una transición desde la
programación manual hacia el aprendizaje basado en datos, donde el sistema adquiere la
capacidad de generalizar más allá de los ejemplos proporcionados durante el
entrenamiento.

El aprendizaje profundo representa una evolución dentro del aprendizaje automático. Su
principal característica radica en el uso de **redes neuronales artificiales** como
núcleo del proceso de aprendizaje. Estas redes, inspiradas en la estructura y el
funcionamiento del cerebro biológico humano, están compuestas por múltiples capas de
procesamiento que permiten **aprender representaciones jerárquicas de la información**.
Gracias a esta arquitectura, el aprendizaje profundo puede capturar relaciones complejas
entre variables, lo que le permite reconocer patrones altamente complejos en los datos.
Como resultado, el aprendizaje profundo ha demostrado un rendimiento excepcional en
tareas que antes se consideraban exclusivas del razonamiento humano, tales como el
reconocimiento de imágenes, el procesamiento del lenguaje natural, el análisis de audio
y la interpretación de grandes volúmenes de datos no estructurados.

### 1.1. Escalabilidad y leyes de crecimiento

Un aspecto esencial en la evolución del aprendizaje profundo es el estudio de las
**leyes de escalado neuronal (_Neural Scaling Laws_)**, las cuales describen
comportamientos empíricamente observables en el rendimiento de los modelos a medida que
se incrementan los recursos disponibles. Estas leyes establecen que, al aumentar de
forma sistemática el tamaño de los conjuntos de datos, la capacidad computacional y el
número de parámetros de un modelo, se obtiene una mejora predecible y sostenida en la
precisión y eficiencia de las predicciones.

Este fenómeno ha guiado gran parte de la estrategia de desarrollo en la industria
tecnológica contemporánea. Empresas líderes como Google, Meta, OpenAI y otras
organizaciones han adoptado el principio del escalado como un eje fundamental de su
investigación y desarrollo, apostando por la creación de modelos cada vez más grandes y
sofisticados. La aplicación práctica de estas leyes ha dado lugar a la construcción de
redes neuronales más profundas y con un mayor número de neuronas, lo que ha impulsado la
aparición de los denominados **modelos de gran escala**, entre los que destacan los
**modelos de lenguaje de gran tamaño (_Large Language Models, LLMs_)**. Estos modelos
han demostrado una capacidad notable para generalizar conocimientos, generar texto
coherente, responder preguntas complejas y adaptarse a una amplia variedad de tareas
cognitivas. Estos modelos no solo procesan y reproducen patrones lingüísticos, sino que
exhiben capacidades emergentes, como el razonamiento en cadena, la resolución de
problemas matemáticos o la comprensión de instrucciones complejas, que no fueron
programadas explícitamente durante su entrenamiento.

En paralelo, existe una tendencia creciente en la investigación que busca optimizar la
eficiencia computacional sin sacrificar la calidad del modelo. Esta línea de trabajo
resulta relevante en entornos donde los recursos son limitados, como los dispositivos
móviles, los sistemas embebidos o las plataformas de Internet de las Cosas (_Internet of
Things_, IoT).

Para abordar estas limitaciones, se desarrollan múltiples estrategias, entre ellas:

- **Arquitecturas especializadas:** Diseños de redes más ligeras y eficientes, adaptadas
  a las restricciones de hardware.
- **Optimización a nivel de hardware:** Uso de unidades de procesamiento específicas,
  como _Graphics Processing Units_ (GPU), _Tensor Processing Units_ (TPU) o _Neural
  Processing Unit_ (NPU), capaces de acelerar las operaciones matriciales y reducir el
  consumo de energía.
- **Compilación a lenguajes de bajo nivel:** Traducción del modelo a representaciones
  más próximas al hardware para mejorar el rendimiento.

En conjunto, estas estrategias permiten democratizar el acceso y uso del aprendizaje
profundo, posibilitando su ejecución incluso en equipos de consumo general. De este
modo, el campo avanza no solo hacia modelos más grandes y potentes, sino también hacia
sistemas más eficientes, accesibles y sostenibles desde el punto de vista energético y
económico.

### 1.2. Memoria implícita y modelos fundacionales

Las redes neuronales artificiales poseen la capacidad de aproximar distribuciones de
probabilidad a partir de los datos de entrada. En esencia, su propósito es construir una
función parametrizada que permita comprender, representar y generalizar el
comportamiento de los datos observados.

En los modelos actuales, esta capacidad alcanza niveles en los que la red puede llegar a
memorizar parte de los datos de entrenamiento. Aunque las arquitecturas contemporáneas
no suelen incorporar mecanismos explícitos de memoria, como una base de datos interna o
una estructura dedicada al almacenamiento, la información queda codificada en los
propios parámetros del modelo. Este fenómeno se manifiesta en la activación selectiva de
neuronas ante determinados contextos, lo que sugiere que la red conserva rastros de
información previa y los utiliza para procesar nuevas entradas.

Aunque esta memoria no sea explícita, existen líneas de investigación que buscan
extender o complementar este comportamiento con mecanismos dedicados. En algunos casos,
se exploran estructuras que incorporan memoria persistente, como las redes recurrentes o
los _Transformers_ con mecanismos de atención. En otros, se utilizan recursos
_hardware_, como la memoria caché o el almacenamiento intermedio en disco, para
gestionar información temporal durante los procesos de entrenamiento e inferencia. Estas
aproximaciones buscan aumentar la capacidad de los modelos para manejar secuencias
largas, retener información contextual de manera más eficiente y facilitar un
aprendizaje más continuo.

La existencia de esta memoria implícita plantea, además, una distinción fundamental
entre los **datos dentro de distribución (_in-distribution_)** y los **datos fuera de
distribución (_out-of-distribution_)**. Los primeros se refieren a ejemplos similares a
los utilizados durante el entrenamiento, en los cuales el modelo optimiza su función
objetivo hasta alcanzar la **convergencia**, es decir, hasta que los ajustes en los
parámetros dejan de producir mejoras significativas en el desempeño. Los segundos, en
cambio, corresponden a entradas que difieren significativamente del conjunto de
entrenamiento, lo que puede provocar fallos, respuestas erróneas o predicciones con alta
incertidumbre. Por ejemplo, un modelo entrenado exclusivamente para reconocer perros no
es capaz de identificar correctamente un gato, ya que este pertenece a una distribución
distinta de patrones visuales y características. Sin embargo, debido a que los conjuntos
de datos empleados actualmente son cada vez más amplios, diversos y heterogéneos, esta
separación entre ambos tipos de datos tiende a desdibujarse. La capacidad de los modelos
para generalizar más allá de su distribución de entrenamiento constituye un área activa
de investigación.

En este contexto, el estudio de la **capacidad de generalización** de los modelos
adquiere un papel central. Los avances recientes han explorado estrategias que permiten
mejorar la inferencia y la estimación de incertidumbre. Entre estas estrategias destacan
el uso de técnicas de cálculo en tiempo de inferencia (_test-time computation_), que
permiten al modelo dedicar más recursos computacionales a problemas complejos en el
momento de la predicción, las redes neuronales bayesianas, que incorporan distribuciones
de probabilidad sobre los parámetros en lugar de valores fijos, y técnicas como _Monte
Carlo Dropout_, que simula múltiples predicciones mediante la desactivación aleatoria de
neuronas durante la inferencia. Estas aproximaciones posibilitan la creación de
**intervalos de confianza** para las predicciones, otorgando a los modelos una mayor
robustez frente a datos desconocidos y una capacidad para expresar el grado de seguridad
de sus respuestas.

Paralelamente, existe el fenómeno del **olvido catastrófico (_catastrophic
forgetting_)**, que describe la tendencia de las redes neuronales a perder información
previamente aprendida cuando incorporan nuevo conocimiento. Este problema representa uno
de los mayores desafíos del **aprendizaje continuo (_continual learning_)**, un
paradigma en el que se busca que el modelo sea capaz de actualizarse de manera
progresiva sin olvidar su conocimiento previo. La solución a este desafío requiere el
desarrollo de mecanismos que equilibren la plasticidad (la capacidad de aprender nueva
información) con la estabilidad (la preservación del conocimiento existente).

La evolución de estas ideas conduce al desarrollo de los **modelos fundacionales
(_foundation models_)**, que se conciben como sistemas de aprendizaje generalista
capaces de adaptarse a múltiples dominios y tareas. Estos modelos no están diseñados
para una tarea específica, sino que aprenden representaciones amplias y abstractas del
mundo que pueden reutilizarse en diversos contextos. A partir de una base preentrenada
sobre grandes volúmenes de datos, es posible **ajustarlos finamente (_fine-tuning_)**
para resolver tareas concretas sin necesidad de entrenarlos desde cero.

### 1.3. El aprendizaje como problema de optimización

El proceso de aprendizaje en redes neuronales debe entenderse, desde una perspectiva
formal, como un problema de optimización matemática. En este marco, un modelo se define
a partir de un conjunto de parámetros ajustables que determinan su comportamiento. Estos
parámetros representan el conocimiento adquirido durante el entrenamiento y se
actualizan progresivamente con el objetivo de **minimizar una función que mide el error
del modelo** respecto a los datos observados.

Las redes neuronales se consideran **modelos diferenciables** porque su mecanismo de
aprendizaje se basa en la capacidad de **calcular derivadas parciales** de una **función
de coste** (también denominada **función de pérdida**) con respecto a sus parámetros.
Esta función cuantifica la discrepancia entre las predicciones generadas por el modelo y
los valores reales, actuando como una medida de su rendimiento. Los parámetros
aprendibles son, por tanto, aquellas variables internas que se modifican iterativamente
para reducir dicha discrepancia y mejorar la capacidad predictiva del sistema.

El proceso de aprendizaje es **iterativo y dinámico**. Consiste en un ciclo continuo de
cálculo, actualización y evaluación que se repite hasta alcanzar un criterio de parada
determinado. Este criterio puede definirse en función del número de iteraciones, de la
estabilidad alcanzada por la función de coste o de la satisfacción de una métrica de
desempeño preestablecida. En la práctica, este procedimiento se implementa mediante
algoritmos de optimización, entre los que destaca el **descenso del gradiente**, que
ajusta los parámetros en la dirección que más reduce la pérdida. Existen además
variantes adaptativas, que mejoran la eficiencia del proceso y aceleran la convergencia
en arquitecturas complejas.

Una herramienta fundamental que posibilita este proceso es la **diferenciación
automática**, la cual permite calcular de manera eficiente las derivadas necesarias para
actualizar los parámetros del modelo. Gracias a esta técnica, es posible entrenar redes
profundas sin requerir una derivación manual de las expresiones analíticas. La
diferenciación automática constituye, por tanto, uno de los pilares que han hecho viable
la expansión moderna del aprendizaje profundo.

No obstante, el carácter diferenciable del modelo impone ciertas restricciones sobre los
tipos de datos que pueden procesarse directamente. Las derivadas sólo son aplicables a
funciones continuas, por lo que representaciones discretas (como caracteres, palabras o
números enteros) no pueden utilizarse tal cual en los cálculos diferenciales. Para
hacerlos compatibles, los datos deben transformarse en representaciones numéricas
continuas, generalmente en forma de **vectores o tensores**, que permitan aplicar las
operaciones matemáticas requeridas durante el entrenamiento.

Este proceso de conversión se denomina **_embedding_**, y su función no se limita
únicamente a permitir el procesamiento diferencial, sino también a **capturar las
relaciones semánticas, estructurales y contextuales entre los elementos de los datos**.
Por ejemplo, en el caso del lenguaje natural, los _embeddings_ permiten representar
palabras o frases de modo que aquellas con significados similares se encuentren próximas
en el **espacio vectorial**, que constituye el espacio matemático multidimensional
creado por el propio modelo. Este espacio permite al sistema establecer y mapear las
relaciones semánticas entre los datos de manera cuantitativa, facilitando operaciones
como la comparación de similitudes, la búsqueda de analogías o la agrupación de
conceptos relacionados. De este modo, los _embeddings_ transforman información simbólica
en representaciones geométricas que preservan y codifican el significado subyacente de
los datos originales.

A medida que el modelo optimiza su función de coste, desarrolla internamente una forma
de **entender y codificar la información** que refleja la estructura subyacente de los
datos. Cuanto mejor sea la capacidad del modelo para comprimir la información sin perder
significado, más eficaz será su desempeño. La compresión eficiente implica que el modelo
ha aprendido a distinguir entre la información relevante y la irrelevante, capturando
sólo aquellos patrones que resultan esenciales para la tarea. Este principio de
compresión es, en última instancia, una manifestación del aprendizaje mismo: la
habilidad de mapear, abstraer y recuperar información compleja sin necesidad de
conservar todos los detalles explícitos.

### 1.4. Arquitecturas y tipos de datos

El aprendizaje profundo se adapta a diferentes problemas mediante el uso de
arquitecturas especializadas, diseñadas para extraer información relevante según la
naturaleza y estructura del tipo de datos analizados. Cada arquitectura incorpora
componentes y operaciones específicas que explotan las características intrínsecas de
los datos, permitiendo al modelo capturar patrones de manera más eficiente y efectiva.
Entre las principales arquitecturas destacan:

- **Redes neuronales densas o totalmente conectadas (_Fully Connected Networks_, FCN)**:
  Constituyen la arquitectura más básica y general, en la que cada neurona de una capa
  está conectada con todas las neuronas de la capa siguiente. Estas redes pueden
  procesar, por lo general, cualquier tipo de datos, siempre que estos se presenten en
  forma vectorial unidimensional, es decir, aplanados (_flattened_). Aunque versátiles,
  presentan limitaciones al trabajar con datos de alta dimensionalidad o con estructuras
  espaciales o temporales complejas, debido al elevado número de parámetros que
  requieren y a su incapacidad para explotar eficientemente dichas estructuras.

- **Redes convolucionales (_Convolutional Neural Networks_, CNN)**: Diseñadas
  específicamente para el procesamiento de datos que poseen estructura espacial o
  espacio-temporal, como imágenes y vídeos. Las CNN utilizan operaciones de convolución
  que aplican filtros deslizantes sobre los datos de entrada, detectando patrones
  locales como bordes, texturas o formas geométricas en las primeras capas, y
  progresivamente características más abstractas y complejas en capas más profundas.
  Esta arquitectura explota la localidad espacial y la invariancia traslacional,
  reduciendo significativamente el número de parámetros en comparación con redes densas
  equivalentes, y facilitando la generalización del modelo a diferentes posiciones
  dentro de la imagen.

- **Redes recurrentes (_Recurrent Neural Networks_, RNN)** y sus variantes modernas,
  como las LSTM (_Long Short-Term Memory_) y GRU (_Gated Recurrent Units_): Empleadas en
  el tratamiento de secuencias, donde el orden temporal de los datos es primordial.
  Estas arquitecturas son especialmente adecuadas para procesar texto, series
  temporales, señales de audio o cualquier tipo de datos secuenciales. Las RNN
  incorporan conexiones recurrentes que permiten a la red mantener un estado interno o
  memoria que captura información de elementos anteriores de la secuencia, posibilitando
  la modelización de dependencias temporales.

- **Modelos basados en _Transformers_**: Representan una evolución significativa en el
  procesamiento de secuencias, basándose en mecanismos de atención que permiten al
  modelo ponderar la importancia de diferentes elementos de la entrada de manera
  dinámica y contextual. Los _Transformers_ han demostrado ser altamente efectivos para
  tareas de procesamiento de lenguaje natural y han sido adoptados también en otros
  dominios como la visión por computador.

- **Modelos multimodales**: Capaces de integrar y procesar información proveniente de
  distintas fuentes o modalidades, como texto, imágenes, audio y vídeo. Estos modelos se
  basan en la idea de representar todos los datos de entrada, independientemente de su
  formato original, como **representaciones embebidas** (_embeddings_) en un espacio
  vectorial común. Este espacio, creado y aprendido por el modelo durante el
  entrenamiento, permite establecer relaciones semánticas entre elementos de diferentes
  modalidades, facilitando que conceptos similares (expresados en formatos distintos) se
  encuentren próximos en dicho espacio. Este proceso de conversión se conoce actualmente
  como **tokenización**, y consiste en la creación de **_tokens_**, representaciones
  vectoriales aprendibles y entendibles por el modelo que encapsulan unidades
  significativas de información. Un único modelo final puede entonces procesar estos
  _tokens_ de manera unificada, independientemente de su origen modal, permitiendo
  tareas complejas como la generación de descripciones textuales a partir de imágenes,
  la búsqueda multimodal o la traducción entre diferentes tipos de contenido.

En este contexto, resulta necesario distinguir entre diferentes tipos de datos según su
estructura y formato:

- **Datos estructurados**: Organizados en tablas de filas y columnas, donde cada fila
  representa una observación o ejemplo, y cada columna corresponde a una característica
  o variable con un significado bien definido. Este formato es característico de las
  bases de datos relacionales tradicionales y de las hojas de cálculo. Para este tipo de
  datos, suelen bastar algoritmos de aprendizaje automático clásicos, como árboles de
  decisión o regresión logística, que pueden alcanzar rendimientos competitivos sin
  requerir la complejidad arquitectónica del aprendizaje profundo. No obstante, las
  redes neuronales también pueden aplicarse a datos estructurados, especialmente cuando
  existen interacciones complejas entre variables o cuando se combinan con datos no
  estructurados en modelos híbridos.

- **Datos no estructurados**: Carecen de una organización tabular predefinida y
  presentan formatos heterogéneos y complejos. Ejemplos incluyen imágenes, grabaciones
  de voz, documentos en lenguaje natural, vídeos o señales biomédicas. Estos datos
  requieren arquitecturas avanzadas de aprendizaje profundo para su procesamiento
  efectivo, pues contienen patrones intrincados, relaciones jerárquicas y dependencias
  contextuales que no pueden ser fácilmente capturadas por algoritmos tradicionales. El
  _Deep Learning_ se muestra especialmente eficaz en estos casos, permitiendo extraer
  automáticamente representaciones significativas y patrones complejos a partir de
  grandes volúmenes de información, sin necesidad de ingeniería manual de
  características. Perfecto. A continuación te presento la **versión extendida
  completa** del módulo **2. Conceptos básicos de matemáticas**, con todas las secciones
  integradas, cohesionadas y redactadas con el mismo tono formal, técnico y didáctico.
  Se han añadido las subsecciones 2.4 a 2.9 siguiendo la línea académica del material
  original, reforzando la narrativa y manteniendo la progresión natural hacia los
  fundamentos de _Deep Learning_.

## 2. Conceptos básicos de matemáticas

### 2.1. Tensores como estructura fundamental

En el ámbito del aprendizaje profundo, los **tensores** constituyen la estructura de
datos esencial sobre la cual se construye y ejecuta la totalidad del proceso de cómputo.
Un tensor puede definirse formalmente como una colección ordenada de elementos numéricos
organizados en un espacio de $N$ dimensiones, que permite representar, almacenar y
manipular información de manera eficiente dentro de un modelo de red neuronal.

Su principal ventaja radica en su compatibilidad con sistemas de cómputo masivamente
paralelos, como las unidades de procesamiento gráfico (GPU) o las unidades de
procesamiento tensorial (TPU). Estas arquitecturas están diseñadas para ejecutar de
forma simultánea miles de operaciones matemáticas, lo cual resulta indispensable para el
entrenamiento y la inferencia en redes neuronales de gran escala, donde la eficiencia
computacional y el manejo óptimo de los recursos son factores determinantes.

Cada tensor se describe a partir de dos componentes fundamentales: el tipo de datos que
contiene y la precisión numérica empleada en los cálculos. Los valores almacenados
suelen ser numéricos, representados comúnmente como enteros o números en coma flotante.
En la práctica, los modelos de aprendizaje profundo suelen utilizar tensores de 32 bits
(precisión simple), aunque es frecuente aplicar técnicas de **cuantización** que reducen
la precisión a 16, 8 o incluso 4 bits, especialmente una vez completada la fase de
entrenamiento. Estas reducciones, sin embargo, dependen de las capacidades del hardware,
ya que no todas las arquitecturas soportan operaciones de baja precisión con la misma
eficiencia o estabilidad numérica.

Bibliotecas especializadas como **PyTorch**, **TensorFlow** o **Keras** facilitan estos
procesos mediante instrucciones de alto nivel. La elección del nivel de precisión
implica un compromiso entre exactitud y eficiencia. En aplicaciones donde los errores
mínimos son tolerables, como la clasificación de imágenes comunes, puede optarse por una
menor precisión para reducir el consumo energético y acelerar el entrenamiento. En
cambio, en entornos donde la seguridad y la fiabilidad son crítica, se requiere una
precisión numérica más alta que garantice la estabilidad y exactitud de los resultados.
Por tanto, existe una relación directa entre la precisión numérica, el error acumulado y
el costo computacional, de modo que optimizar este equilibrio constituye uno de los
aspectos clave del diseño de modelos eficientes.

Desde el punto de vista operativo, los tensores funcionan de manera análoga a los
**_arrays_** de los lenguajes de programación tradicionales, permitiendo realizar
operaciones como indexación, segmentación o extracción de subconjuntos de datos. Estas
operaciones son esenciales, ya que posibilitan el procesamiento de partes específicas de
un conjunto de información sin necesidad de manipular el tensor completo.

La dimensionalidad es una de las características más importantes de los tensores, pues
determina la forma en que los datos se estructuran internamente. Según su número de
dimensiones, pueden clasificarse del siguiente modo:

- Un **escalar** corresponde a un tensor de dimensión cero y representa un único valor
  numérico.
- Un **vector** es un tensor unidimensional que almacena una secuencia ordenada de
  valores.
- Una **matriz** constituye un tensor bidimensional que organiza los datos en filas y
  columnas.
- Los **tensores de orden superior**, con tres o más dimensiones, permiten representar
  estructuras de datos más complejas, como secuencias temporales, imágenes, vídeos o
  volúmenes tridimensionales.

Un ejemplo ilustrativo lo constituye una imagen en color de 84 × 84 píxeles con tres
canales (rojo, verde y azul) procesada en lotes durante el entrenamiento. En este caso,
la representación corresponde a un tensor de rango 4, cuyas dimensiones reflejan: el
número de ejemplos en el lote, la altura y la anchura de la imagen, y el número de
canales de color.

En el ámbito del aprendizaje profundo, el tensor constituye la unidad de procesamiento
fundamental dentro de las bibliotecas de cálculo numérico. La mayoría de los modelos se
construyen mediante la composición de funciones elementales, tales como sumas,
multiplicaciones y transformaciones no lineales. Estas operaciones permiten representar
relaciones complejas entre los datos y, por tanto, son esenciales para el funcionamiento
de los modelos de inteligencia artificial.

Toda la información que un modelo procesa se expresa en forma de tensores, los cuales
pueden entenderse como **_arrays_ multidimensionales** que abarcan desde los escalares
(orden 0) y vectores (orden 1), hasta matrices (orden 2) y estructuras de orden
superior. En consecuencia, muchas de las operaciones habituales aplicables sobre
_arrays_ pueden ejecutarse directamente sobre tensores, lo que facilita la manipulación
de los datos.

Para ilustrar estos conceptos se empleará **PyTorch**, una biblioteca de código abierto
para _Deep Learning_ reconocida por su flexibilidad, su ecosistema de herramientas
complementarias y su amplia adopción tanto en el ámbito académico como en el industrial.
PyTorch permite definir, entrenar y desplegar modelos de redes neuronales de manera
eficiente, ofreciendo una interfaz altamente integrada con el lenguaje de programación
Python, lo que la hace especialmente accesible para investigadores y desarrolladores.

Aunque existen otras alternativas consolidadas, como **TensorFlow**, **JAX** y
**Keras**, PyTorch destaca por su creciente popularidad y por su estrecha vinculación
con la Linux Foundation, lo que garantiza un desarrollo sostenido y un soporte
comunitario cada vez mayor. Además, múltiples proyectos de terceros, como **Ray**,
utilizado para la creación de sistemas distribuidos de entrenamiento de modelos, también
forman parte del ecosistema de la Linux Foundation. Este entorno colaborativo impulsa la
innovación y asegura un soporte activo tanto por parte de empresas tecnológicas
reconocidas como de la comunidad de código abierto.

Una de las principales ventajas de PyTorch es su sintaxis intuitiva y expresiva, que
sigue de forma natural los principios del estilo “**pythónico**”, es decir, un diseño
limpio y legible que favorece la comprensión del código.

Independientemente de la biblioteca elegida, los principios matemáticos y conceptuales
que sustentan el aprendizaje profundo son los mismos. Las diferencias radican
principalmente en la sintaxis y en las implementaciones específicas de cada entorno,
pero la base teórica y las operaciones fundamentales definidas sobre tensores permanecen
invariantes.

### 2.2. Operaciones vectoriales

Los vectores constituyen tensores unidimensionales, habitualmente representados como
$x \sim (d)$, donde $d$ indica la dimensión del tensor. En el contexto del álgebra
lineal, y también en el ámbito de la inteligencia artificial, resulta fundamental
distinguir entre vectores columna y vectores fila, denotados respectivamente por $x$ y
$x^\top$. En esta notación, el superíndice del segundo símbolo representa la
**transpuesta** del vector, operación que intercambia filas por columnas. Un vector
columna puede considerarse como un tensor bidimensional de forma $(d, 1)$, mientras que
un vector fila posee forma $(1, d)$.

Esta distinción es particularmente relevante en entornos de programación, donde las
operaciones entre tensores deben cumplir las reglas de **_broadcasting_**, las cuales
determinan cómo se alinean las dimensiones durante las operaciones aritméticas. En
PyTorch, cuando las dimensiones de los tensores son incompatibles, puede ser necesario
utilizar funciones como `squeeze()`, `unsqueeze()` o `view()` para ajustar su
estructura. Dado que la biblioteca se actualiza con frecuencia, resulta imposible
abarcar todas las posibles modificaciones y nuevas funcionalidades. Por ello, se
recomienda consultar la documentación oficial o realizar búsquedas específicas sobre las
funciones mencionadas para obtener información actualizada.

Si se disponen dos vectores del mismo tamaño, $x$ y $y$, es posible combinarlos
linealmente mediante coeficientes escalares $a$ y $b$, generando un nuevo vector $z$,
tal que:

$$
z = a x + b y.
$$

Desde una perspectiva geométrica, en un espacio euclidiano bidimensional, la suma de
vectores puede interpretarse como la **diagonal del paralelogramo** definido por ambos
vectores. La magnitud o longitud de un vector se mide mediante la **norma euclidiana** o
**norma $L_2$**, definida como:

$$
||x|| = \sqrt{\sum_i x_i^2}.
$$

Esta norma representa la distancia del vector al origen del sistema de coordenadas y
constituye una medida fundamental en la evaluación de magnitudes y distancias.

Otra operación esencial es el **producto escalar** (o **producto punto**), definido
como:

$$
x \cdot y = \sum_i x_i \cdot y_i,
$$

cuyo resultado es un escalar con una interpretación geométrica directa: permite
determinar el **ángulo entre dos vectores** y, en consecuencia, su **similitud
direccional**. Esta relación se expresa mediante la siguiente ecuación:

$$
\cos(\theta) = \frac{x \cdot y}{||x|| , ||y||}.
$$

De acuerdo con este principio:

- Si $\cos(\theta) = 1$, los vectores apuntan en la misma dirección.
- Si $\cos(\theta) = 0$, los vectores son ortogonales, es decir, forman un ángulo de 90
  grados entre sí.
- Si $\cos(\theta) = -1$, los vectores son opuestos, con un ángulo de 180 grados entre
  ambos.

Esta medida se conoce como **similitud del coseno** y desempeña un papel fundamental en
tareas de **agrupamiento (_clustering_)**, búsqueda semántica y **representaciones
latentes**.

Este principio tiene una aplicación directa en los **_word embeddings_**,
representaciones vectoriales del lenguaje en las que cada palabra se codifica como un
punto dentro de un espacio semántico de alta dimensionalidad. Modelos como GPT-2 y GPT-3
utilizan representaciones de entre 768 y más de 12000 dimensiones, lo que permite
capturar relaciones semánticas y sintácticas a través de simples operaciones
vectoriales.

A continuación, se presenta un ejemplo de implementación de la similitud del coseno
utilizando Python con la biblioteca **NumPy**:

```python
import numpy as np

def normalizar_matriz(matriz: np.ndarray) -> np.ndarray:
    return matriz / np.expand_dims(np.sqrt(np.sum(np.power(matriz, 2), axis=1)), axis=-1)

def cosine_similarity(matriz: np.ndarray) -> np.ndarray:
    return matriz @ matriz.T

X = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [1, 0, 0],
    [0, 1, 0]
], dtype=float)

X_normalized = normalizar_matriz(X)
similarity_matrix = cosine_similarity(X_normalized)
print(similarity_matrix)
```

El siguiente código muestra el mismo procedimiento utilizando **PyTorch**:

```python
import torch
import torch.nn.functional as F

X = torch.tensor([
    [1., 2., 3.],
    [4., 5., 6.],
    [1., 0., 0.],
    [0., 1., 0.]
])

# Normalización
X_norm = F.normalize(X, p=2, dim=1)

# Cálculo de la matriz de similitud
similarity = X_norm @ X_norm.T
print(similarity)
```

El resultado obtenido en ambos casos es una matriz de tamaño $N \times N$ que contiene
los valores de similitud entre cada par de vectores. En la diagonal principal aparecen
valores iguales a 1, ya que cada vector presenta similitud máxima consigo mismo.

### 2.3. Operaciones matriciales

Una matriz es un arreglo bidimensional que organiza los datos en filas y columnas, por
lo que puede entenderse como una **colección ordenada de vectores**. Matemáticamente,
una matriz $X \in \mathbb{R}^{A \times B}$ está compuesta por $A$ filas y $B$ columnas,
donde cada elemento $x_{ij}$ representa el valor ubicado en la fila $i$ y la columna
$j$.

Si se dispone de una matriz $X \in \mathbb{R}^{A \times B}$ y otra
$Y \in \mathbb{R}^{B \times C}$, su **producto matricial** se define como:

$$
Z = X Y,
$$

donde $Z \in \mathbb{R}^{A \times C}$. Esta operación es válida únicamente cuando el
número de columnas de $X$ coincide con el número de filas de $Y$. En términos
algebraicos, cada elemento de la matriz resultante se calcula como:

$$
Z_{ij} = \sum_{k=1}^{B} X_{ik} \cdot Y_{kj}.
$$

El producto matricial es una de las operaciones más utilizadas en el aprendizaje
profundo, ya que permite procesar simultáneamente grandes volúmenes de información. En
el contexto de una capa neuronal, los datos de entrada suelen representarse mediante una
matriz donde cada fila corresponde a una muestra y cada columna a una característica. Al
multiplicar esta matriz por otra que contiene los pesos del modelo, se obtiene una
transformación lineal de las entradas, a la cual se suma posteriormente un vector de
sesgo.

Además del producto matricial convencional, existen otras operaciones de gran relevancia
en el cálculo numérico y el aprendizaje profundo. Una de ellas es el **producto de
Hadamard**, también conocido como multiplicación elemento a elemento. A diferencia del
producto matricial, esta operación se realiza exclusivamente entre matrices del mismo
tamaño y se define como:

$$
Z_{ij} = X_{ij} \cdot Y_{ij}.
$$

El producto de Hadamard se emplea en múltiples contextos, entre los cuales destaca su
uso en **mecanismos de enmascaramiento** (_masking_) durante el entrenamiento de
modelos. Esta técnica permite ignorar valores específicos de un tensor para impedir que
influyan en el cálculo de los gradientes o en la propagación de errores. Dicha propiedad
es esencial en arquitecturas modernas como **_Transformers_**, donde se aplica para
restringir la atención a determinadas posiciones o para manejar secuencias de longitud
variable sin afectar el aprendizaje global del modelo.

### 2.4. Operaciones con tensores en PyTorch

La biblioteca PyTorch proporciona un conjunto amplio, eficiente y flexible de
herramientas para la creación, manipulación y transformación de tensores, que
constituyen la estructura de datos fundamental en el aprendizaje profundo. Los tensores
generalizan los conceptos de escalares, vectores y matrices hacia dimensiones
superiores, lo que permite representar datos complejos de manera multidimensional y
realizar operaciones matemáticas de forma vectorizada y optimizada.

A continuación, se presentan ejemplos prácticos y comentados que ilustran las
operaciones más comunes con tensores en PyTorch. Estas operaciones son esenciales para
comprender el funcionamiento interno del código empleado en la creación de modelos de
aprendizaje profundo. En la práctica, muchas arquitecturas modernas o modificaciones de
arquitecturas existentes surgen a partir de pequeñas variaciones en la manipulación de
tensores, ya sea mediante la selección de elementos específicos (_slicing_), la
optimización de cálculos o el uso de estrategias que reduzcan el coste computacional.

Para crear tensores, es posible hacerlo a partir de listas, mediante inicialización
aleatoria o con valores fijos, por ejemplo:

```python
import torch

# Tensores básicos
escalar = torch.tensor(7)
vector = torch.tensor([1, 2, 3])
matriz = torch.tensor([[1, 2, 3], [4, 5, 6]])

# Tensores aleatorios y de valores fijos
tensor_aleatorio = torch.rand((2, 3))
ceros = torch.zeros((2, 3))
unos = torch.ones((2, 3))
rango = torch.arange(0, 10, 2)

print("Escalar:", escalar)
print("Vector:", vector)
print("Matriz:\n", matriz)
print("Tensor aleatorio:\n", tensor_aleatorio)
print("Tensor de ceros:\n", ceros)
print("Tensor de unos:\n", unos)
print("Rango:", rango)
```

Cada tensor contiene información sobre su **tipo de dato**, sus **dimensiones** y el
**dispositivo de almacenamiento** (CPU o GPU). El tipo de dato (`dtype`) determina la
precisión numérica del tensor, a mayor precisión, mayor será el rango de valores
posibles, pero también el consumo de memoria. El dispositivo (`device`) es relevante
porque un tensor ubicado en la GPU no puede ser manipulado directamente desde la CPU,
por lo que es necesario transferirlo o copiarlo según sea necesario. Por ejemplo:

```python
tensor = torch.rand((2, 3, 4))
print("Tipo de dato:", tensor.dtype)
print("Forma:", tensor.shape)
print("Dispositivo:", tensor.device)
print("Número de elementos:", tensor.numel())
```

Las operaciones de agregación permiten resumir la información contenida en un tensor.
Algunas de las más comunes son la suma, la media o la obtención del valor máximo o
mínimo. El parámetro `dim` indica el eje sobre el cual se aplica la operación, donde
`dim=0` actúa sobre las filas (por columnas), mientras que `dim=1` actúa sobre las
columnas (por filas).

```python
tensor = torch.tensor([[1., 2., 3.], [4., 5., 6.]])

print("Suma total:", tensor.sum())
print("Promedio:", tensor.mean())
print("Máximo por columna:", tensor.max(dim=0))
print("Promedio por fila:", tensor.mean(dim=1))
```

Otras funciones, como `view()`, `reshape()`, `unsqueeze()` y `squeeze()`, permiten
modificar la forma del tensor sin alterar sus datos subyacentes. Estas operaciones son
fundamentales para adaptar las dimensiones de los tensores según las necesidades de las
redes neuronales.

```python
x = torch.arange(1, 7)
print("Tensor original:", x)

# Cambiar forma
x_reshaped = x.view(2, 3)
print("Tensor 2x3:\n", x_reshaped)

# Añadir dimensión
x_unsqueezed = x.unsqueeze(0)
print("Tensor con nueva dimensión:", x_unsqueezed.shape)

# Eliminar dimensión
x_squeezed = x_unsqueezed.squeeze()
print("Tensor tras eliminar dimensión:", x_squeezed.shape)
```

Las funciones `permute()` y `transpose()` permiten reordenar las dimensiones de un
tensor, lo cual es especialmente útil en el procesamiento de imágenes o secuencias, por
ejemplo, al desplazar canales de color o mapas de características.

```python
tensor = torch.rand((2, 3, 4))
print("Forma original:", tensor.shape)

# Transposición (intercambio de dos dimensiones)
tensor_T = tensor.transpose(1, 2)
print("Forma tras transpose:", tensor_T.shape)

# Permutación general de ejes
tensor_P = tensor.permute(2, 0, 1)
print("Forma tras permute:", tensor_P.shape)
```

También es posible combinar tensores mediante funciones como `torch.cat()` y
`torch.stack()`. La primera une tensores existentes a lo largo de un eje específico,
mientras que la segunda crea una nueva dimensión para apilarlos.

```python
a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([[5, 6], [7, 8]])

# Concatenación (mismo número de filas)
cat_0 = torch.cat((a, b), dim=0)
cat_1 = torch.cat((a, b), dim=1)

# Apilamiento (nueva dimensión)
stacked = torch.stack((a, b), dim=0)

print("Concatenación por filas:\n", cat_0)
print("Concatenación por columnas:\n", cat_1)
print("Apilamiento (nueva dimensión):\n", stacked)
```

PyTorch implementa una gran cantidad de **operaciones vectorizadas**, que permiten
realizar cálculos sin recurrir a bucles explícitos. Este enfoque no solo mejora la
legibilidad del código, sino que también aprovecha las optimizaciones internas del
framework y del hardware subyacente, como las implementaciones en CUDA para GPU.

```python
x = torch.tensor([1., 2., 3.])
y = torch.tensor([4., 5., 6.])

print("Suma:", x + y)
print("Producto elemento a elemento:", x * y)
print("Exponencial:", torch.exp(x))
print("Raíz cuadrada:", torch.sqrt(y))
print("Seno:", torch.sin(x))
```

Estas operaciones resultan especialmente útiles para inspeccionar distribuciones de
datos o normalizar tensores antes del entrenamiento, tareas que contribuyen a
estabilizar el aprendizaje de los modelos.

```python
tensor = torch.randn((3, 4))  # Distribución normal
print("Tensor aleatorio:\n", tensor)
print("Media:", tensor.mean())
print("Desviación estándar:", tensor.std())
print("Valor mínimo:", tensor.min())
print("Índice del máximo:", tensor.argmax())
```

Finalmente, PyTorch permite una **conversión directa entre tensores y arreglos de
NumPy**, lo que facilita su integración con bibliotecas de análisis y visualización.
Esta interoperabilidad permite combinar el poder de cálculo de PyTorch con la
versatilidad de ecosistemas como NumPy, Matplotlib o Pandas.

```python
import numpy as np

# Tensor a NumPy
tensor = torch.tensor([[1, 2], [3, 4]])
array = tensor.numpy()
print("Tensor a NumPy:\n", array)

# NumPy a Tensor
nuevo_tensor = torch.from_numpy(array)
print("NumPy a Tensor:\n", nuevo_tensor)
```

En conjunto, estas operaciones proporcionan una visión integral de las capacidades de
PyTorch en la manipulación de tensores, mostrando su versatilidad, eficiencia y
facilidad de integración con otros entornos de análisis. En capítulos posteriores, se
emplearán estos fundamentos para la construcción de modelos de aprendizaje profundo
basados en esta biblioteca.

## 3. Regresión lineal y logística

Los modelos de regresión lineal y logística constituyen la base conceptual del
aprendizaje profundo. También se conocen como modelos diferenciables, ya que su
estructura está compuesta por transformaciones lineales seguidas de funciones no
lineales que son derivables, lo que permite aplicar cálculo diferencial para optimizar
sus parámetros mediante métodos basados en gradientes. Este principio es el fundamento
de todas las arquitecturas de redes neuronales modernas.

El entrenamiento de una neurona, o de una red neuronal, se apoya en dos procesos
fundamentales: la **propagación hacia adelante (_forward propagation_)** y la
**propagación hacia atrás (_backpropagation_)**.

La propagación hacia adelante consiste en calcular la predicción del modelo a partir de
los datos de entrada. En este proceso, los datos ingresan por la capa de entrada y
atraviesan las distintas capas de la red, aplicando sucesivas combinaciones lineales y
no lineales hasta obtener una salida numérica. El resultado que produce el modelo antes
de aplicar una función de activación final se conoce como **_logit_**. Este valor
representa una proyección numérica de los datos de entrada en el espacio interno del
modelo, resultado de las transformaciones que la red realiza. Posteriormente, el modelo
compara esta salida con el valor real esperado y calcula una tasa de error o función de
pérdida, la cual mide qué tan precisa ha sido la representación aprendida.

Por otro lado, la propagación hacia atrás es el proceso mediante el cual el modelo
ajusta sus parámetros internos con el objetivo de minimizar el error obtenido en la
propagación hacia adelante. En este proceso, los gradientes (las derivadas parciales de
la función de pérdida respecto a cada parámetro) se propagan desde la salida hasta las
capas iniciales del modelo. Dichos gradientes indican cómo deben modificarse los pesos y
sesgos para reducir el error en las siguientes iteraciones, permitiendo así un
aprendizaje progresivo y dirigido por el descenso del gradiente.

Un modelo lineal puede expresarse matemáticamente como:

$$
\hat{y} = \mathbf{w}^\top \mathbf{x} + b,
$$

donde $\mathbf{x} \in \mathbb{R}^n$ es el vector de entrada,
$\mathbf{w} \in \mathbb{R}^n$ representa el vector de pesos del modelo,
$b \in \mathbb{R}$ es el sesgo o término independiente, y $\hat{y} \in \mathbb{R}$ es la
salida predicha por el modelo. Cuando la salida $\hat{y}$ no está restringida a un rango
específico, el modelo se utiliza en tareas de regresión, donde el objetivo es predecir
valores continuos. En este contexto, la salida puede tomar cualquier valor real,
positivo o negativo.

Sin embargo, cuando la salida está asociada a un conjunto discreto de clases
$
\mathcal{C} = {1, 2, \dots, M}$, el modelo aborda un problema de clasificación. En
estos casos, la representación numérica (_logits_) generada por el modelo se transforma
en probabilidades mediante una función no lineal, generalmente una función sigmoide para
clasificación binaria o una función _Softmax_ para clasificación multiclase. En la
clasificación binaria ($M = 2$), el modelo aprende a distinguir entre dos posibles
categorías (por ejemplo, “positivo” y “negativo”, o “clase 0” y “clase 1”). En cambio,
en los problemas multiclase, el modelo puede asignar cada entrada a una de varias
categorías posibles, como en la clasificación de imágenes por tipo de objeto o raza de
perro. Además, existen escenarios de clasificación multietiqueta, donde una misma
entrada puede pertenecer simultáneamente a varias clases. Un ejemplo típico se da en los
sistemas de visión artificial para conducción autónoma, en los cuales una sola imagen
puede contener múltiples elementos etiquetables, como peatones, vehículos y señales de
tráfico.

En los modelos diferenciables, la estructura general se puede describir como una
composición de funciones lineales y no lineales:

$$
f(\mathbf{x}) = f_{L} \circ f_{L-1} \circ \dots \circ f_1 (\mathbf{x}),
$$

donde cada capa aplica una transformación de la forma:

$$
f_{\ell}(\mathbf{x}) = \sigma_{\ell}(\mathbf{W}_{\ell}\mathbf{x} + \mathbf{b}_{\ell}).
$$

En esta formulación, $\mathbf{W}_\ell$ y $\mathbf{b}_\ell$ representan los pesos y
sesgos de la capa $\ell$, respectivamente, mientras que $\sigma_{\ell}(\cdot)$ es una
función de activación diferenciable. Esta función introduce no linealidad al modelo y
permite restringir el rango de valores de salida, lo que dota al modelo de la capacidad
de aproximar relaciones complejas y no lineales entre los datos de entrada y salida.

### 3.1. Clasificación mediante regresión logística

En lugar de desarrollar manualmente una aplicación con reglas explícitas para
identificar si una imagen contiene un gato u otro tipo de animal, se puede adoptar un
enfoque basado en aprendizaje profundo. En este contexto, se construye un conjunto de
datos compuesto por múltiples ejemplos de imágenes etiquetadas, algunas con gatos y
otras sin ellos. Este conjunto permite que el modelo aprenda automáticamente a
distinguir un gato de otros animales a partir de los patrones estadísticos presentes en
los datos, sin requerir instrucciones específicas para cada caso.

El objetivo principal de este proceso es modelar la distribución de los datos de manera
que el sistema sea capaz de identificar diferencias entre las distintas clases. En un
escenario de aprendizaje supervisado, cada ejemplo del conjunto de datos se asocia con
una etiqueta que indica si pertenece o no a la clase “gato”. Con ello, el modelo aprende
la relación entre las características de las imágenes y su respectiva clasificación.

Durante este proceso, las etiquetas se representan mediante valores numéricos. Cada
clase tiene asignado un identificador único. Este identificador puede gestionarse
mediante un diccionario, en el que la clave representa el identificador numérico y el
valor que corresponde al nombre de la clase. Una vez que el modelo produce sus
predicciones, se selecciona la clase con el valor más alto y se traduce nuevamente al
nombre de la clase utilizando dicho diccionario. Por ejemplo, si el modelo predice que
el índice más alto corresponde al identificador `1`, el sistema puede mapear este valor
a la clase `"gato"`.

Gracias a la disponibilidad de grandes volúmenes de datos etiquetados, los sistemas
supervisados se han convertido en los más empleados en la práctica. Cada muestra del
conjunto de datos se considera independiente e idénticamente distribuida (i.i.d.), lo
que significa que cada ejemplo es representativo y estadísticamente consistente con la
distribución global de los datos. Este supuesto garantiza que el modelo pueda aprender
patrones estables y generalizables, de modo que las representaciones internas que genera
(también conocidas como espacios embebidos o espacios de representación) resulten
estructuradas y separables, permitiendo agrupar ejemplos similares en regiones cercanas
del espacio de características que crea el modelo internamente.

Siguiendo con el ejemplo de la clasificación de gatos, cada imagen de entrada se
representa mediante un conjunto de píxeles con tres canales de color (rojo, verde y
azul). Si cada canal tiene una resolución de, por ejemplo, $64 \times 64$ píxeles, el
número total de valores por imagen es $64 \times 64 \times 3 = 12288$.

Para que esta información pueda ser procesada por un modelo de red neuronal, las tres
matrices de color se **aplanan (_flatten_)**, convirtiéndose en un único vector columna
de dimensión $12288 \times 1$. Este vector conserva la información de los píxeles, pero
la reorganiza en una estructura unidimensional apta para cálculos matriciales.

Si se dispone de $M$ ejemplos, la matriz de características $X$ tendrá dimensión
$(n, M)$, donde $n = 12288$, mientras que el vector de etiquetas $Y$ tendrá dimensión
$(1, M)$ y contendrá los valores binarios correspondientes a cada muestra.

Para resolver este problema, se emplea la regresión logística, un algoritmo de
aprendizaje supervisado diseñado específicamente para tareas de clasificación binaria.
Su funcionamiento es similar al de la regresión lineal, pero incorpora una **función de
activación sigmoide** que transforma la salida del modelo en un valor comprendido entre
0 y 1, interpretable como una probabilidad. La función sigmoide se define como:

$$
\sigma(z) = \frac{1}{1 + e^{-z}},
$$

donde:

$$
z = \mathbf{w}^\top \mathbf{x} + b.
$$

En esta formulación, $\mathbf{w}$ representa el vector de pesos, $b$ el término de
sesgo, y $\mathbf{x}$ el vector de características de la imagen. La predicción final del
modelo se expresa como:

$$
\hat{y} = \sigma(\mathbf{w}^\top \mathbf{x} + b),
$$

donde $\hat{y}$ indica la probabilidad de que la imagen pertenezca a la clase positiva
(es decir, que contenga un gato). Si el valor de $\hat{y}$ supera un determinado umbral
de decisión (por ejemplo, 0.5), la imagen se clasifica como perteneciente a la clase
“gato”, en caso contrario, se clasifica como “no gato”.

### 3.2. Función de pérdida y función de coste

Una vez obtenidos los datos, es necesario formalizar el proceso mediante el cual un
modelo ajusta sus predicciones a los resultados esperados. Este procedimiento se
fundamenta en la **función de pérdida**, una magnitud escalar y diferenciable que
cuantifica el error cometido por el modelo en una predicción individual. Su valor
refleja el grado de discrepancia entre la salida estimada y el valor real, constituyendo
así un indicador directo del rendimiento del modelo.

Durante el entrenamiento, el objetivo principal es **minimizar la función de pérdida**,
reduciendo la diferencia entre las predicciones generadas y los valores verdaderos. En
el caso del aprendizaje supervisado, esta minimización se realiza comparando las
etiquetas reales con las salidas del modelo. Por el contrario, en contextos no
supervisados, donde no existen etiquetas explícitas, se optimizan otras métricas, como
las distancias entre muestras o el error cuadrático medio entre reconstrucciones y los
datos originales, entre otras.

El proceso de optimización se ejecuta habitualmente mediante el **descenso del
gradiente**. Durante este proceso, los parámetros del modelo, los pesos ($w$) y el sesgo
($b$), se ajustan iterativamente con el fin de minimizar la discrepancia entre las
predicciones y las etiquetas reales.

Es importante distinguir entre función de pérdida y función de coste. La función de
pérdida mide el error correspondiente a un único ejemplo de entrenamiento, mientras que
la función de coste representa el promedio de dichas pérdidas a lo largo de todo el
conjunto de entrenamiento.

En el aprendizaje supervisado, el modelo genera una predicción $\hat{y}$ a partir de un
ejemplo de entrada $x$, que luego se compara con la etiqueta real $y$. Este proceso se
repite para todas las muestras, y el promedio de las pérdidas individuales define la
función de coste total:

$$
J(w, b) = \frac{1}{M} \sum_{i=1}^{M} \mathcal{L}(\hat{y}^{(i)}, y^{(i)}),
$$

donde $M$ representa el número total de ejemplos. El objetivo del entrenamiento es
encontrar los parámetros óptimos de $(w, b)$ que minimicen la función de pérdida
$\mathcal{L}$, la cual mide la discrepancia entre las predicciones $\hat{y}_i$ y los
valores verdaderos $y_i$.

En la regresión logística, la función de pérdida más empleada es la **_log-loss_** o
pérdida logarítmica, definida como:

$$
\mathcal{L}(\hat{y}, y) = - \big( y \cdot \log(\hat{y}) + (1 - y) \cdot \log(1 - \hat{y}) \big).
$$

A partir de esta definición, la función de coste correspondiente se expresa como el
promedio de todas las pérdidas individuales:

$$
J(w, b) = \frac{1}{M} \sum_{i=1}^{M} \mathcal{L}(\hat{y}^{(i)}, y^{(i)})
= -\frac{1}{M} \sum_{i=1}^{M} \Big[ y^{(i)} \log(\hat{y}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)}) \Big],
$$

donde $M$ es el número total de ejemplos, $\hat{y}^{(i)} = \sigma(w^T x^{(i)} + b)$ es
la probabilidad estimada por el modelo para el ejemplo $i$, $x^{(i)}$ representa el
vector de características del ejemplo, $y^{(i)}$ es la etiqueta real y $\sigma(z)$ es la
**función sigmoide**, que transforma valores reales en el intervalo $(0, 1)$. Esta
formulación penaliza de forma más efectiva los errores en problemas de clasificación
binaria que el error cuadrático medio (_Mean Square Error_, MSE), ya que la _log-loss_
proporciona gradientes más estables y evita ciertos problemas de convergencia asociados
a funciones no logarítmicas.

Sin embargo, el MSE sigue siendo ampliamente utilizado en tareas de regresión, donde se
define como:

$$
\text{MSE} = \frac{1}{M} \sum_{i=1}^{M} (\hat{y}^{(i)} - y^{(i)})^2.
$$

En problemas de regresión, la elección de la función de pérdida depende de la naturaleza
de los datos y de la sensibilidad deseada frente a valores atípicos. La MSE penaliza con
mayor intensidad los errores grandes, por lo que resulta sensible a la presencia de
valores extremos. En contraposición, la pérdida absoluta media (_Mean Average Error_,
MAE) ofrece una alternativa más robusta frente a valores atípicos, aunque su derivada no
está definida en los puntos donde $y_i = \hat{y}_i$:

$$
\mathcal{L}_{\text{MAE}} = \frac{1}{N} \sum_{i=1}^{N} |y_i - \hat{y}_i|.
$$

Para equilibrar las ventajas de ambas métricas, se utiliza con frecuencia la pérdida de
Huber, que introduce un parámetro de transición $\delta > 0$ y combina los
comportamientos del MSE y del MAE en una sola formulación:

$$
\mathcal{L}_{\text{Huber}} =
\begin{cases}
\frac{1}{2}(y_i - \hat{y}_i)^2, & \text{si } |y_i - \hat{y}_i| \leq \delta \\
\delta \cdot (|y_i - \hat{y}_i| - \frac{1}{2}\delta), & \text{en otro caso.}
\end{cases}
$$

La pérdida de Huber es diferenciable en casi todos los puntos, salvo en el límite
$|y_i - \hat{y}_i| = \delta$, aunque esta discontinuidad no genera inestabilidad
numérica debido a la precisión finita de los cálculos. Por este motivo, se aplica
habitualmente en contextos donde se busca un equilibrio entre la robustez frente a
valores atípicos y la estabilidad del proceso de optimización.

Finalmente, destacar que un modelo que obtiene un coste bajo en el conjunto de
entrenamiento no garantiza un buen rendimiento general. Este fenómeno, conocido como
sobreajuste (_overfitting_), se presenta cuando el modelo alcanza una elevada precisión
en los datos de entrenamiento, pero su desempeño se degrada significativamente al
evaluarse en datos nuevos. En tales casos, el modelo no aprende patrones generalizables,
sino que memoriza los ejemplos específicos del conjunto de entrenamiento.

El sobreajuste puede deberse a un número insuficiente de muestras, a arquitecturas
excesivamente complejas o a problemas en la representación de los datos, como etiquetado
incorrecto, desequilibrio de clases o sesgos en el conjunto de entrenamiento. Asimismo,
las diferencias entre las distribuciones de los datos de entrenamiento y los de
producción pueden comprometer la capacidad de generalización del modelo.

### 3.3. Descenso del gradiente

El descenso del gradiente constituye uno de los algoritmos fundamentales para el
entrenamiento de modelos en aprendizaje automático. Su propósito es encontrar los
valores de los parámetros que minimizan una determinada función de coste, garantizando
que las predicciones del modelo se ajusten lo mejor posible a los datos observados.

En el caso de la regresión logística, recordar que la función de coste $J(w, b)$ se
define a partir de la función de pérdida logarítmica:

$$
J(w, b) = \frac{1}{M} \sum_{i=1}^{M} \mathcal{L}(\hat{y}^{(i)}, y^{(i)})
= -\frac{1}{M} \sum_{i=1}^{M} \Big[ y^{(i)} \log(\hat{y}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)}) \Big].
$$

Para reducir el valor de $J(w, b)$, se calculan las derivadas parciales con respecto a
los parámetros del modelo. Estas derivadas determinan la dirección del gradiente, es
decir, el sentido en el que la función de coste crece más rápidamente. Dado que el
objetivo es minimizarla, el algoritmo ajusta los parámetros en la dirección opuesta al
gradiente:

$$
\frac{\partial J}{\partial w} = dw = \frac{1}{M} \sum_{i=1}^{M} (\hat{y}^{(i)} - y^{(i)}) x^{(i)},\\ \quad
\frac{\partial J}{\partial b} = db = \frac{1}{M} \sum_{i=1}^{M} (\hat{y}^{(i)} - y^{(i)}).
$$

Estos términos indican cómo deben modificarse $w$ y $b$ en cada iteración para disminuir
el error. El procedimiento completo del descenso del gradiente se desarrolla de forma
iterativa y puede resumirse en las siguientes fases:

1. **Inicialización de los parámetros**: Se asignan valores iniciales, generalmente
   pequeños, ya sean ceros o valores aleatorios.
2. **Propagación hacia adelante**: Se calculan las predicciones $\hat{y}$ a partir de
   los datos de entrada $X$ y se evalúa la función de pérdida $\mathcal{L}(\hat{y}, y)$
   y la función de coste $J(w, b)$.
3. **Propagación hacia atrás**: Se obtienen las derivadas parciales $dw$ y $db$, que
   indican la dirección del ajuste de los parámetros.
4. **Actualización de parámetros**: Se actualizan los valores de $w$ y $b$ según la
   regla:

$$
w := w - \alpha \cdot dw,\\ \quad b := b - \alpha \cdot db,
$$

donde $\alpha$ es la tasa de aprendizaje o ratio de aprendizaje, un hiperparámetro que
controla el tamaño del paso dado en cada iteración. Si $\alpha$ es demasiado grande, el
algoritmo puede divergir y si es demasiado pequeño, la convergencia será muy lenta. El
proceso se repite hasta alcanzar un mínimo adecuado de $J(w, b)$, lo que se traduce en
predicciones más precisas.

En la práctica, el descenso del gradiente se implementa de forma vectorizada,
aprovechando operaciones matriciales sobre todos los ejemplos del conjunto de
entrenamiento en paralelo. Esta formulación no solo simplifica la implementación, sino
que también permite aprovechar la capacidad de cómputo de las GPU.

Para ilustrar el funcionamiento del algoritmo, considérese la función bidimensional:

$$
f(x) = \sin(x_1)\cos(x_2) + \sin(0.5x_1)\cos(0.5x_2), \quad x \in [0, 10].
$$

El objetivo consiste en aplicar el descenso del gradiente sobre esta función, calculando
explícitamente las derivadas parciales respecto a $x_1$ y $x_2$, e implementando el
algoritmo en Python mediante la biblioteca NumPy:

```python
import numpy as np
import matplotlib.pyplot as plt

# Definición de la función
def function(input: np.ndarray) -> np.ndarray:
    assert input.shape[-1] == 2, "La entrada debe contener 2 elementos"
    return np.sin(input[:, 0]) * np.cos(input[:, 1]) + np.sin(0.5 * input[:, 0]) * np.cos(0.5 * input[:, 1])

# Cálculo del gradiente (derivadas parciales)
def gradiente(input: np.ndarray) -> np.ndarray:
    assert input.shape[-1] == 2, "La entrada debe contener 2 elementos"

    df_x1 = np.cos(input[:, 0]) * np.cos(input[:, 1]) + 0.5 * np.cos(0.5 * input[:, 0]) * np.cos(0.5 * input[:, 1])
    df_x2 = -np.sin(input[:, 0]) * np.sin(input[:, 1]) - 0.5 * np.sin(0.5 * input[:, 0]) * np.sin(0.5 * input[:, 1])

    return np.stack([df_x1, df_x2], axis=1)

# Algoritmo de descenso del gradiente
def descenso_gradiente(num_puntos: int = 10, num_iteraciones: int = 30, learning_rate: float = 1e-3):
    dim = 2

    # Inicialización en el dominio [0,10]
    X = np.random.rand(num_puntos, dim) * 10
    trayectorias = [X.copy()]

    for _ in range(num_iteraciones):
        X = X - learning_rate * gradiente(input=X)
        trayectorias.append(X.copy())

    return np.array(trayectorias)

# Ejecución del descenso del gradiente
trayectoria = descenso_gradiente(num_puntos=5, num_iteraciones=30)

# Visualización de las trayectorias
for i in range(trayectoria.shape[1]):
    plt.plot(trayectoria[:, i, 0], trayectoria[:, i, 1], marker="o")

plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Trayectorias del descenso del gradiente")
plt.show()
```

Al ejecutar este código, se observa que los puntos iniciales evolucionan siguiendo
trayectorias determinadas por el gradiente de la función. En cada iteración, las
posiciones se actualizan desplazándose en la dirección opuesta a la pendiente local, lo
que permite avanzar hacia valores más bajos de la función objetivo.

En el contexto de redes neuronales, el cálculo de derivadas necesarias para aplicar el
descenso del gradiente se realiza mediante sistemas de **diferenciación automática**,
que en el caso de PyTorch este proceso se realiza mediante el módulo **`autograd`**.
Este sistema permite calcular derivadas de manera automática sobre operaciones
tensoriales, lo que constituye la base del algoritmo de _backpropagation_ utilizado para
ajustar los parámetros de las redes neuronales profundas. Cada tensor en PyTorch puede
llevar asociada la propiedad `requires_grad=True`, que indica si debe participar en el
cálculo de gradientes. PyTorch construye un grafo computacional dinámico que registra
las operaciones realizadas sobre los tensores, y al invocar el método `backward()`,
aplica la regla de la cadena para calcular las derivadas necesarias. El algoritmo de
propagación hacia atrás también se conoce como el modo automático inverso, de
diferenciación en sistemas de la computación con ciencias de la computación.

Un ejemplo simple de su funcionamiento es el siguiente:

```python
import torch

x = torch.tensor([2.0, 3.0], requires_grad=True)
y = x**2 + 2*x + 1
z = y.sum()
z.backward()

# Derivadas parciales de z respecto a x
print(x.grad)
```

Además, PyTorch permite desactivar el cálculo de gradientes cuando no es necesario, como
durante la fase de inferencia, utilizando el contexto `with torch.no_grad():` o el modo
`with torch.inference_mode():`, que resulta aún más eficiente. Esto reduce
significativamente el consumo de memoria y mejora el rendimiento computacional.

Para mejorar la eficiencia, en lugar de calcular los gradientes utilizando el conjunto
completo de datos, se emplea el **descenso de gradiente estocástico (SGD)**, que utiliza
pequeños subconjuntos (_mini-batches_). Esta aproximación introduce aleatoriedad,
disminuye el coste computacional y ayuda a escapar de regiones problemáticas, como los
puntos de silla, donde los gradientes se anulan sin representar un mínimo real.

El descenso de gradiente básico puede resultar ineficiente en ciertos escenarios, por lo
que se han desarrollado variantes que mejoran su rendimiento. Una de ellas es el
**algoritmo Momentum**, que introduce un efecto de inercia acumulando información de
gradientes previos para suavizar las actualizaciones. Se define mediante las ecuaciones

$$
v_t = \beta v_{t-1} + (1-\beta) \, \nabla_\theta \mathcal{L}(\theta_t),
$$

$$
\theta_{t+1} = \theta_t - \eta \, v_t,
$$

donde $v_t$ representa la “velocidad” acumulada y $\beta \in [0,1)$ es el coeficiente de
decaimiento, generalmente fijado en 0.9. Este mecanismo reduce las oscilaciones en
direcciones de alta curvatura y acelera la convergencia en valles estrechos.

Otro método es **RMSprop**, que adapta la tasa de aprendizaje a cada parámetro mediante
el escalado de los gradientes por una media móvil de sus valores al cuadrado

$$
s_t = \rho s_{t-1} + (1-\rho) \left(\nabla_\theta \mathcal{L}(\theta_t)\right)^2,
$$

$$
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{s_t + \epsilon}} \, \nabla_\theta \mathcal{L}(\theta_t),
$$

donde $\rho \approx 0.9$ y $\epsilon \approx 10^{-8}$ para evitar divisiones por cero.
Este ajuste permite que los parámetros con gradientes grandes reciban pasos más
pequeños, mientras que aquellos con gradientes pequeños se actualizan más rápidamente,
mejorando la estabilidad del entrenamiento.

El optimizador **Adam** combina las ventajas de Momentum y RMSprop, acumulando tanto la
media de los gradientes como la media de sus cuadrados. Su formulación se realiza en
cuatro etapas.

1. **Media de gradientes (primer momento):**

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1) \, \nabla_\theta \mathcal{L}(\theta_t).
$$

2. **Media de cuadrados de gradientes (segundo momento):**

$$
v_t = \beta_2 v_{t-1} + (1-\beta_2) \, \left(\nabla_\theta \mathcal{L}(\theta_t)\right)^2.
$$

3. **Corrección del sesgo inicial:**

$$
\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t}.
$$

4. **Actualización final de los parámetros:**

$$
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \, \hat{m}_t.
$$

Los valores recomendados son $\beta_1 = 0.9$, $\beta_2 = 0.999$ y $\epsilon = 10^{-8}$.
Este optimizador se utiliza ampliamente debido a su rapidez, estabilidad y robustez
frente a configuraciones no óptimas de hiperparámetros.

A modo de ilustración, se presenta una implementación de los principales optimizadores
aplicada a la función de prueba $f(\theta) = \theta^2$, cuyo mínimo global se encuentra
en $\theta=0$. En este ejemplo, todos los optimizadores parten de un valor inicial
$\theta=5$ y buscan reducir la función de pérdida. Aunque cada algoritmo sigue
trayectorias distintas, todos tienden hacia el mínimo global en $\theta=0$.

```python
import numpy as np

# Función de pérdida y gradiente
loss = lambda theta: theta**2
grad = lambda theta: 2*theta

# Valor inicial
theta_init = 5.0

# Descenso de gradiente estocástico (SGD)
def sgd(theta, grad, eta=0.1, steps=20):
    for t in range(steps):
        theta -= eta * grad(theta)
    return theta

# Momentum
def momentum(theta, grad, eta=0.1, beta=0.9, steps=20):
    v = 0
    for t in range(steps):
        v = beta * v + (1 - beta) * grad(theta)
        theta -= eta * v
    return theta

# RMSprop
def rmsprop(theta, grad, eta=0.1, rho=0.9, eps=1e-8, steps=20):
    s = 0
    for t in range(steps):
        g = grad(theta)
        s = rho * s + (1 - rho) * g**2
        theta -= eta / (np.sqrt(s) + eps) * g
    return theta

# Adam
def adam(theta, grad, eta=0.1, beta1=0.9, beta2=0.999, eps=1e-8, steps=20):
    m, v = 0, 0
    for t in range(1, steps+1):
        g = grad(theta)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g**2
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        theta -= eta / (np.sqrt(v_hat) + eps) * m_hat
    return theta

print("SGD:", sgd(theta_init, grad))
print("Momentum:", momentum(theta_init, grad))
print("RMSprop:", rmsprop(theta_init, grad))
print("Adam:", adam(theta_init, grad))
```

### 3.4. Métodos de regularización

En el contexto de la regresión lineal, es posible obtener una solución analítica para
los pesos del modelo mediante la **pseudoinversa de Moore–Penrose**, que proporciona una
estimación cerrada de los parámetros cuando la matriz de diseño no es cuadrada o no
tiene inversa directa. Esta solución se expresa como:

$$
\mathbf{w} = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{y},
$$

donde $\mathbf{X} \in \mathbb{R}^{N \times n}$ representa la matriz de datos de entrada
y $\mathbf{y} \in \mathbb{R}^N$ los valores objetivo. Sin embargo, este enfoque puede
resultar numéricamente inestable cuando la matriz $(\mathbf{X}^\top \mathbf{X})$ es casi
singular, es decir, cuando algunos de sus valores propios son muy pequeños o cercanos a
cero. En tales casos, pequeñas variaciones en los datos pueden producir grandes cambios
en los parámetros estimados, lo que conduce a un modelo **sobreajustado** y con escasa
capacidad de generalización.

Para mitigar este problema y mejorar la estabilidad del modelo, se introduce un
**término de regularización** en la función de coste. La regularización actúa como un
mecanismo de control que penaliza los pesos excesivamente grandes, favoreciendo
soluciones más estables y reduciendo la varianza del modelo. De este modo, se logra un
equilibrio entre el ajuste a los datos de entrenamiento y la capacidad de generalización
ante nuevos ejemplos. Los métodos más comunes son la **regularización L2 (_Ridge
Regression_)** y la **regularización L1 (_Lasso Regression_)**.

La regularización L2 agrega al término de error un componente proporcional al cuadrado
de la magnitud de los pesos. Este término penaliza los parámetros de gran magnitud,
promoviendo valores pequeños y distribuidos de manera más uniforme. Su función de
pérdida se define como:

$$
\mathcal{L}*{\text{Ridge}} = \frac{1}{N} \sum*{i=1}^{N} (y_i - \hat{y}_i)^2 + \lambda |\mathbf{w}|_2^2,
$$

donde $\lambda$ es un hiperparámetro que controla la intensidad de la penalización.
Cuanto mayor sea su valor, más fuerte será la restricción sobre los pesos. La
regularización L2 produce modelos más suaves y estables, ya que evita oscilaciones
excesivas en los parámetros y contribuye a que el proceso de entrenamiento sea más
controlado.

Por otro lado, la regularización L1 incorpora un término basado en la suma de los
valores absolutos de los pesos:

$$
\mathcal{L}*{\text{Lasso}} = \frac{1}{N} \sum*{i=1}^{N} (y_i - \hat{y}_i)^2 + \lambda |\mathbf{w}|_1.
$$

A diferencia de la regularización L2, el término L1 tiende a forzar algunos coeficientes
a ser exactamente cero, lo que induce esparsidad en el modelo. En la práctica, esto
significa que ciertos parámetros se eliminan completamente, dando lugar a modelos más
simples y con menos variables efectivamente activas. Este comportamiento convierte a la
regularización L1 en una herramienta útil para selección de características, ya que
identifica de manera implícita las variables más relevantes para la predicción.

La regularización y la normalización son técnicas fundamentales para mejorar la
capacidad de generalización de los modelos de aprendizaje profundo y reducir el riesgo
de sobreajuste. Ambas estrategias buscan limitar la dependencia excesiva del modelo
respecto a los datos de entrenamiento, promoviendo representaciones más robustas y
estables que permitan un rendimiento consistente en datos no vistos.

Entre las técnicas de regularización más utilizadas destacan:

- **Regularización L2 (ridge)**: Añade una penalización proporcional al cuadrado de los
  pesos del modelo. Esta restricción evita que los parámetros adquieran valores
  demasiado grandes, estabiliza la optimización y favorece soluciones más suaves y
  generalizables.

- **Regularización L1 (lasso)**: Penaliza la magnitud absoluta de los pesos, induciendo
  que muchos de ellos se reduzcan a cero. Esto simplifica el modelo al conservar
  únicamente las variables más relevantes, actuando además como un método de selección
  de características.

- **Dropout**: Desactiva aleatoriamente un subconjunto de neuronas durante el
  entrenamiento, impidiendo que las unidades desarrollen dependencias excesivas entre
  sí. Esto obliga a la red a generar representaciones redundantes y más robustas.
  Durante la inferencia, todas las neuronas se utilizan, con los pesos ajustados de
  forma apropiada.

- **Aumentación de datos (data augmentation)**: Crea ejemplos adicionales a partir de
  transformaciones aplicadas a los datos originales, como rotaciones, traslaciones,
  cambios de escala o variaciones de iluminación. Esta técnica incrementa la diversidad
  del conjunto de entrenamiento y hace que el modelo sea menos sensible a variaciones
  irrelevantes.

- **Detención temprana (early stopping)**: Supervisa el rendimiento del modelo sobre el
  conjunto de validación y detiene el entrenamiento cuando el error deja de mejorar,
  evitando que la red se ajuste demasiado a las particularidades del conjunto de
  entrenamiento.

- **Normalización de entradas**: Escala y centra las características de los datos para
  garantizar magnitudes comparables, acelerando la convergencia, mejorando la
  estabilidad numérica y evitando que ciertos parámetros dominen el aprendizaje.

En complemento a la regularización, las técnicas de **normalización de activaciones**
resultan esenciales para estabilizar el entrenamiento y acelerar la convergencia.
Durante la optimización, las activaciones pueden variar significativamente entre capas,
lo que genera inestabilidad y dificulta el ajuste de los parámetros. La normalización
busca mantener distribuciones equilibradas de las activaciones a lo largo de la red.

- **_Batch Normalization_**: Normaliza las activaciones de cada capa utilizando la media
  y la varianza calculadas sobre los ejemplos de un mini-lote. Esto reduce el problema
  del _internal covariate shift_, acelera el aprendizaje, permite tasas de aprendizaje
  más altas y simplifica el ajuste de hiperparámetros. Sin embargo, su efectividad
  depende del tamaño y la composición de los lotes, siendo menos adecuada en lotes
  pequeños o en datos con distribuciones muy variables.

- **_Layer Normalization_**: Normaliza las activaciones a nivel de capa, calculando
  estadísticas por muestra en lugar de por mini-lote. Es especialmente útil en
  arquitecturas secuenciales, como los transformadores, y en escenarios de entrenamiento
  distribuido, ya que no requiere compartir estadísticas entre lotes, facilitando la
  paralelización y la escalabilidad. A diferencia de _Batch Normalization_, proporciona
  mayor estabilidad en situaciones donde la composición de los lotes puede variar
  significativamente.

### 3.5. Sistemas de clasificación multiclase y la función Softmax

En los sistemas de clasificación multiclase, el objetivo del modelo es asignar una
probabilidad a cada una de las posibles clases, de modo que la suma de todas ellas sea
igual a uno. Para lograrlo, la capa de salida del modelo suele aplicar la **función
_Softmax_** sobre los valores de activación o _logits_ generados por la red neuronal.
Estos logits se definen como:

$$
z_i = \mathbf{w}_i^\top \mathbf{x} + b_i,
$$

donde $\mathbf{x}$ representa el vector de entrada, $\mathbf{w}_i$ los pesos asociados a
la clase $i$, y $b_i$ el sesgo correspondiente. A partir de estos valores, la función
_Softmax_ transforma los logits en una distribución de probabilidad normalizada:

$$
p_i = \text{Softmax}(z_i) = \frac{e^{z_i / T}}{\sum_{j=1}^{M} e^{z_j / T}},
$$

donde $T > 0$ es el **parámetro de temperatura** que controla la “nitidez” de la
distribución.

Cuando el valor de $T$ es grande, las diferencias entre los exponentes se atenúan, y la
distribución resultante $p_i$ se aproxima a una distribución uniforme, lo que refleja
una mayor incertidumbre del modelo. Por el contrario, cuando $T$ tiende a cero, la
probabilidad se concentra en la clase más probable, haciendo que las predicciones sean
más deterministas y el modelo más “seguro” en sus decisiones.

El proceso de predicción final se realiza seleccionando la clase con la probabilidad más
alta en el vector resultante. Matemáticamente, la clase predicha se obtiene mediante:

$$
\hat{y} = \arg\max_i , p_i.
$$

En la práctica, este índice suele corresponder a la posición del valor máximo en el
tensor de salida, que actúa como identificador de la clase predicha.

Con el fin de evitar que el modelo sea excesivamente confiado en sus predicciones, se
utiliza una técnica denominada **suavizado de etiquetas (_label smoothing_)**. Este
procedimiento ajusta las etiquetas verdaderas, reduciendo ligeramente la probabilidad
asignada a la clase correcta y redistribuyendo parte de ella entre las demás clases.
Así, se disminuye la rigidez del modelo y se mejora su capacidad de generalización,
especialmente en entornos con ruido o ambigüedad en los datos. El suavizado se aplica
comúnmente sobre etiquetas codificadas en formato **_one-hot_**, según la expresión:

$$
y_i' = (1 - \varepsilon) y_i + \frac{\varepsilon}{M},
$$

donde $\varepsilon \in [0,1]$ determina el grado de suavizado. Un valor de $\varepsilon$
pequeño mantiene las etiquetas casi idénticas a las originales, mientras que valores
mayores hacen que el modelo asigne probabilidades más equilibradas entre las clases.

Para el entrenamiento de modelos de clasificación, la **función de pérdida por entropía
cruzada** es una de las más empleadas. Esta mide la discrepancia entre la distribución
verdadera $\mathbf{y}$ y la distribución predicha $\mathbf{p}$:

$$
\mathcal{L}*{\text{CE}} = - \sum*{i=1}^{M} y_i \log(p_i).
$$

El objetivo del aprendizaje consiste en minimizar esta pérdida, lo cual equivale a
**maximizar la probabilidad asignada a la clase correcta**. Desde un punto de vista
teórico, la entropía cruzada puede descomponerse como:

$$
\mathcal{L}*{\text{CE}} = H(\mathbf{y}, \mathbf{p}) = H(\mathbf{y}) + D*{KL}(\mathbf{y} ,||, \mathbf{p}),
$$

donde:

- $H(\mathbf{y})$ representa la entropía de las etiquetas verdaderas, y
- $D_{KL}(\mathbf{y} ,||, \mathbf{p})$ es la **divergencia de Kullback–Leibler**,
  definida como:

$$
D_{KL}(\mathbf{y} ,||, \mathbf{p}) = \sum_{i=1}^{M} y_i \log \frac{y_i}{p_i}.
$$

Minimizar la entropía cruzada implica reducir la divergencia entre la distribución
predicha y la distribución verdadera, lo que conlleva que las probabilidades generadas
por el modelo se aproximen progresivamente a las etiquetas esperadas.

### 3.6. Incertidumbre, calibración y funciones de pérdida focal

Aunque la función _Softmax_ transforma los _logits_ en probabilidades dentro del
intervalo $[0, 1]$ que suman 1, estas **no reflejan necesariamente la verdadera
incertidumbre del modelo**. Un valor de probabilidad elevado no garantiza que la
predicción sea fiable. Este fenómeno se debe a que muchos modelos modernos,
especialmente las redes neuronales profundas, tienden a ser excesivamente confiados en
sus predicciones, incluso cuando son erróneas.

La **calibración del modelo** surge precisamente para corregir este comportamiento. Su
objetivo es alinear las probabilidades predichas con las frecuencias empíricas
observadas, de modo que la confianza expresada por el modelo corresponda con la
realidad. En otras palabras, un modelo calibrado asigna una probabilidad del 70% a una
clase si, efectivamente, aproximadamente el 70% de las muestras con esa predicción son
correctas.

Un modelo se considera perfectamente calibrado cuando, para cualquier clase $k$ y
probabilidad $p$, se cumple la relación:

$$
P(Y = k \mid \hat{P}(Y = k) = p) = p.
$$

Esto implica que, si el modelo predice una probabilidad $p$ para la clase $k$, la
frecuencia real de acierto entre todas las predicciones con esa confianza debe coincidir
con $p$. Así, las probabilidades producidas por el modelo pueden interpretarse como
estimaciones realistas de la incertidumbre.

El flujo general del proceso de calibración se desarrolla en las siguientes etapas:

1. **Entrenamiento del modelo** sobre el conjunto de entrenamiento, optimizando una
   función de pérdida estándar (por ejemplo, entropía cruzada).
2. **Obtención de los _logits_ o probabilidades** sobre un conjunto de validación
   independiente, sin utilizar los datos de entrenamiento.
3. **Aplicación de un método de calibración**, como el escalado de temperatura, el
   _Platt scaling_ o la regresión isotónica, según la naturaleza del problema y la
   complejidad del modelo.
4. **Optimización de los parámetros del calibrador**, minimizando una medida de
   discrepancia entre las probabilidades predichas y las etiquetas verdaderas, como la
   entropía cruzada o el error cuadrático medio.
5. **Evaluación del grado de calibración**, empleando métricas especializadas como el
   _Expected Calibration Error (ECE)_ o el _Maximum Calibration Error (MCE)_, que
   cuantifican la diferencia entre la confianza predicha y la precisión observada.
6. **Implementación del calibrador final** para ajustar las probabilidades generadas por
   el modelo en el conjunto de prueba o durante la fase de inferencia en producción.

Este proceso garantiza que las probabilidades emitidas por el modelo no solo sirvan para
clasificar correctamente, sino que también expresen con precisión el nivel de confianza
asociado a cada predicción. Esta propiedad resulta especialmente relevante en
aplicaciones críticas, como el diagnóstico médico asistido por inteligencia artificial,
la conducción autónoma o los sistemas de apoyo a la toma de decisiones, donde la
fiabilidad de la confianza predicha puede ser tan importante como la propia precisión
del modelo.

#### 3.6.1. Calibración mediante escalado de temperatura

Uno de los métodos más simples y eficaces para calibrar redes neuronales es el
**escalado de temperatura** (_temperature scaling_). Este procedimiento consiste en
ajustar un único parámetro $T > 0$ que reescala los _logits_ antes de aplicar la función
_Softmax_:

$$
p_i = \text{Softmax}\left(\frac{z_i}{T}\right).
$$

El valor de $T$ modifica la dispersión de la distribución de probabilidades:

- Si $T = 1$, no se aplica ningún ajuste.
- Si $T > 1$, la distribución se suaviza, produciendo predicciones menos confiadas (más
  uniformes).
- Si $T < 1$, la distribución se concentra, incrementando la confianza del modelo.

El parámetro $T$ se optimiza sobre un conjunto de validación minimizando la entropía
cruzada entre las probabilidades ajustadas y las etiquetas verdaderas:

$$
T^* = \arg \min_{T > 0} \frac{1}{N_{\text{val}}} \sum_{i=1}^{N_{\text{val}}} -y_i \log \text{Softmax}\left(\frac{z_i}{T}\right).
$$

Cabe destacar que este ajuste no altera la clase predicha (el valor de $\arg\max$
permanece igual), sino que modifica la confianza asociada a cada predicción, mejorando
así la calibración del modelo.

#### 3.6.2. Métodos alternativos de calibración

Para problemas de clasificación binaria, el **_Platt Scaling_** ofrece una alternativa
paramétrica al escalado de temperatura. En este caso, los _logits_ se ajustan mediante
una función sigmoide:

$$
\hat{p}_i = \frac{1}{1 + \exp(A z_i + B)},
$$

donde $A$ y $B$ son parámetros optimizados sobre un conjunto de validación con el fin de
minimizar la entropía cruzada. Este método generaliza el escalado de temperatura,
permitiendo además un desplazamiento lineal mediante el parámetro $B$.

Otro enfoque ampliamente utilizado es la regresión isotónica, un método no paramétrico
que ajusta las probabilidades predichas $\hat{p}_i$ mediante una función monótonamente
creciente $f$:

$$
p_i^{\text{cal}} = f(\hat{p}_i).
$$

El objetivo consiste en minimizar la desviación cuadrática entre las probabilidades
ajustadas y las observaciones reales:

$$
f^* = \arg \min_f \sum_{i=1}^{N_{\text{val}}} (y_i - f(\hat{p}_i))^2.
$$

A diferencia del escalado de temperatura, este método es más flexible, aunque requiere
un mayor número de muestras de validación para evitar el sobreajuste.

#### 3.6.3. Métricas de evaluación de la calibración

Para cuantificar el grado de calibración de un modelo, se emplean métricas específicas,
entre las cuales destacan:

- **Expected Calibration Error (ECE):**

$$
\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} \Big| \text{acc}(B_m) - \text{conf}(B_m) \Big|,
$$

donde $B_m$ representa el conjunto de predicciones cuyo nivel de confianza se encuentra
dentro del intervalo del bin $m$, $|B_m|$ es el número de ejemplos en ese bin,
$\text{acc}(B_m)$ la precisión observada, y $\text{conf}(B_m)$ la confianza promedio.

- **Maximum Calibration Error (MCE):**

$$
\text{MCE} = \max_m \Big| \text{acc}(B_m) - \text{conf}(B_m) \Big|.
$$

Ambas métricas miden la discrepancia entre la confianza del modelo y la frecuencia real
de aciertos, valores bajos indican una mejor calibración.

#### 3.6.4. Pérdida focal

En escenarios con desbalance de clases o gran cantidad de ejemplos “fáciles”, la función
de pérdida tradicional por entropía cruzada puede resultar insuficiente, ya que el
modelo tiende a sobreajustar los ejemplos dominantes. Para mitigar este efecto, Lin et
al. (2017) propusieron la **pérdida focal (_Focal Loss_)**, definida como:

$$
\mathcal{L}_{\text{Focal}} = - (1 - p_t)^\gamma \log(p_t),
$$

donde:

- $p_t$ es la probabilidad predicha para la clase verdadera, y
- $\gamma \ge 0$ es un parámetro de enfoque que aumenta el peso relativo de los ejemplos
  difíciles (aquellos con $p_t$ pequeño) y reduce la influencia de los ejemplos fáciles.

Cuando $\gamma = 0$, la pérdida focal se reduce a la entropía cruzada estándar. Valores
mayores de $\gamma$ incrementan la penalización sobre los ejemplos en los que el modelo
muestra menor confianza, favoreciendo un aprendizaje más equilibrado.

### 3.7. Implementación de la regresión logística

Para ilustrar de forma práctica los conceptos presentados anteriormente, a continuación
se muestra una implementación básica de la regresión logística utilizando Python y la
librería NumPy. Este ejemplo incluye todas las etapas fundamentales del modelo: la
inicialización de parámetros, _forward propagation_, _backward propagation_, la
actualización de los parámetros mediante descenso del gradiente y, finalmente, la
generación de predicciones y la evaluación del modelo.

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
    """Función sigmoide."""
    return 1 / (1 + np.exp(-z))

def initialize_params(n):
    """Inicializa los parámetros del modelo."""
    w = np.zeros((n, 1))
    b = 0
    return w, b

def forward_propagation(w, b, X, Y):
    """Calcula la activación y el coste."""
    m = X.shape[1]
    Z = np.dot(w.T, X) + b
    A = sigmoid(Z)
    cost = -(1/m) * np.sum(Y*np.log(A) + (1-Y)*np.log(1-A))
    return A, cost

def backward_propagation(A, X, Y):
    """Calcula los gradientes del coste respecto a los parámetros."""
    m = X.shape[1]
    dw = (1/m) * np.dot(X, (A - Y).T)
    db = (1/m) * np.sum(A - Y)
    return dw, db

def update_params(w, b, dw, db, learning_rate):
    """Actualiza los parámetros usando descenso del gradiente."""
    w -= learning_rate * dw
    b -= learning_rate * db
    return w, b

# 3. Entrenamiento del modelo
def logistic_regression(X, Y, num_iterations=1000, learning_rate=0.1, print_cost=False):
    """Entrena el modelo de regresión logística."""
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
    """Genera predicciones binarias a partir de los parámetros entrenados."""
    A = sigmoid(np.dot(w.T, X) + b)
    return (A > 0.5).astype(int)

# 5. Entrenar y evaluar el modelo
w, b, costs = logistic_regression(X, Y, num_iterations=1000, learning_rate=0.1, print_cost=True)
Y_pred = predict(w, b, X)
accuracy = 100 - np.mean(np.abs(Y_pred - Y)) * 100
print(f"\nExactitud del modelo: {accuracy:.2f}%")

# 6. Visualización de la evolución del coste
plt.plot(costs)
plt.xlabel("Iteraciones (x100)")
plt.ylabel("Coste")
plt.title("Reducción del coste durante el entrenamiento")
plt.show()
```

## 4. Neurona artificil y redes neuronales

Para ilustrar el funcionamiento básico de este tipo de modelos, puede considerarse el
problema de estimar el precio de una vivienda. Si se representa gráficamente el tamaño
de la casa frente a su precio, se observa una tendencia creciente positiva: a mayor
tamaño de la vivienda, mayor precio. Una forma de capturar esta relación es mediante la
**regresión lineal**, que consiste en ajustar una línea recta que describe la relación
entre ambas variables. Esta línea se caracteriza por dos parámetros fundamentales: su
posición vertical, determinada por el término independiente, y su pendiente, que define
la tasa de cambio del precio respecto al tamaño. Sin embargo, este enfoque presenta
limitaciones importantes. Por ejemplo, al extrapolar la línea recta hacia valores muy
pequeños de tamaño, el modelo podría asignar precios negativos a viviendas
extremadamente reducidas, lo cual carece de sentido práctico. Para resolver este
problema, se incorporan funciones que restringen los resultados a intervalos válidos de
salidas, garantizando que las predicciones mantengan coherencia con la realidad física
del problema.

Este procedimiento puede comprenderse mejor mediante la analogía de una neurona
artificial, también conocida como perceptrón. La neurona recibe el tamaño de la vivienda
como entrada y aplica un cálculo lineal parametrizado, cuyos parámetros se han obtenido
a partir de ejemplos de entrenamiento. Posteriormente, utiliza una función de activación
que filtra valores inválidos, produciendo una estimación coherente del precio dentro de
un rango válido. Por ejemplo, la función puede garantizar que la salida sea siempre
positiva, evitando precios negativos. De este modo, la neurona artificial transforma la
entrada mediante una combinación de operaciones lineales y no lineales, ajustándose
progresivamente a los patrones presentes en los datos.

No obstante, el valor de una vivienda depende de múltiples factores adicionales, como el
número de dormitorios, el número de baños, la ubicación geográfica, la proximidad a
servicios públicos, la calidad del vecindario o el estado de conservación de la
propiedad. La incorporación de estas características incrementa la **dimensionalidad**
de los datos, es decir, el número de variables o dimensiones necesarias para describir
cada ejemplo. En este escenario, la simple regresión lineal se vuelve insuficiente,
puesto que una única línea recta solo es capaz de relacionar linealmente dos variables.
Para abordar problemas de mayor complejidad, resulta necesario combinar múltiples
perceptrones organizados en **capas**, lo que da lugar a arquitecturas que permiten
modelar no solo relaciones lineales individuales entre pares de variables, sino también
combinaciones complejas de múltiples parámetros de entrada. Además, estas arquitecturas
posibilitan que las neuronas de capas sucesivas procesen y combinen las representaciones
generadas por capas anteriores, construyendo progresivamente abstracciones de mayor
nivel que capturan patrones sofisticados en los datos.

En las arquitecturas de **_Deep Learning_** se distinguen tres tipos de capas
fundamentales. La **capa de entrada** recibe las características iniciales del problema,
es decir, los datos de entrada tras aplicar las transformaciones oportunas para obtener
valores numéricos que el modelo pueda procesar. Las **capas ocultas** (_hidden layers_)
se sitúan entre la entrada y la salida, y su función consiste en procesar y transformar
progresivamente dichas características, extrayendo representaciones intermedias cada vez
más abstractas y relevantes para la tarea en cuestión. Finalmente, la **capa de salida**
genera la predicción final del modelo, que en el ejemplo de las viviendas correspondería
al precio estimado. La profundidad de la red, determinada por el número de capas
ocultas, influye directamente en su capacidad para aprender relaciones complejas y no
lineales entre las variables.

Cada neurona artificial asigna un **peso** a cada característica de entrada, indicando
la importancia relativa de esa variable en el resultado final. Estos pesos determinan
cuánto contribuye cada entrada a la activación de la neurona. Además, cada neurona
incluye un **sesgo** (_bias_), un valor adicional que permite ajustar la función de
salida y otorga mayor flexibilidad al modelo. El sesgo modula la activación de la
neurona en función de los datos de entrada, desplazando efectivamente el umbral a partir
del cual la neurona se activa. Tanto los pesos como el sesgo se inicializan de manera
aleatoria al comienzo del entrenamiento y se ajustan progresivamente mediante algoritmos
de optimización, como el descenso del gradiente, optimizando así el rendimiento del
sistema. Estos constituyen los parámetros aprendibles de los modelos de inteligencia
artificial, cuya configuración final determina el comportamiento y las capacidades del
modelo entrenado.

El resultado de la combinación lineal de las entradas ponderadas por los pesos, sumado
al sesgo, pasa posteriormente por una función de activación no lineal. Este componente
es esencial, ya que otorga a la red la capacidad de capturar relaciones complejas y no
lineales entre variables, superando las limitaciones de los modelos puramente lineales.
Sin funciones de activación no lineales, una red neuronal multicapa se comportaría
simplemente como un modelo lineal, independientemente de su profundidad. Además, la
función de activación permite definir intervalos coherentes para las salidas.

Las redes neuronales profundas constituyen una extensión de las redes neuronales
artificiales tradicionales y parten de los mismos fundamentos ya estudiados. Su
principal diferencia radica en la presencia de múltiples capas ocultas, dispuestas de
manera secuencial como una pila, lo que permite construir representaciones jerárquicas
de la información. Dichas representaciones reflejan un nivel progresivo de conocimiento
que se amplía tanto en profundidad como en amplitud. La amplitud depende del número de
parámetros aprendibles de cada capa, como ocurre con la cantidad de neuronas en una capa
densa o con el número de canales y filtros en las capas convolucionales, que se
analizarán en capítulos posteriores.

En este contexto, las primeras capas de la red, situadas cerca de la entrada, poseen una
capacidad limitada para reconocer patrones complejos y suelen detectar únicamente
características elementales. Por ejemplo, en arquitecturas diseñadas para procesar
imágenes, las capas iniciales tienden a identificar líneas horizontales, verticales o
diagonales. Conforme se avanza hacia capas más profundas, las representaciones se
vuelven progresivamente más sofisticadas, ya que se construyen combinando las
características detectadas en etapas anteriores. De este modo, en niveles intermedios es
posible identificar formas más estructuradas, mientras que en las capas finales se
logran representaciones de alto nivel que corresponden a objetos completos o conceptos
abstractos.

Este proceso resulta coherente porque la red aplica de manera sucesiva operaciones no
lineales que integran y transforman la información proveniente de distintas
características. Cada capa contribuye con una representación parcial que, al combinarse
con las anteriores, incrementa la complejidad y la capacidad de entendimiento del
modelo. En consecuencia, las redes neuronales profundas destacan por su habilidad para
aprender patrones complejos y abstractos a partir de grandes volúmenes de datos
diversos, lo que las convierte en una herramienta esencial en el campo del aprendizaje
automático y la inteligencia artificial.

La principal motivación para utilizar redes profundas radica precisamente en esta
capacidad de aprender representaciones jerárquicas. Las primeras capas tienden a
detectar características elementales, como bordes en imágenes o frecuencias simples en
señales de audio. Las capas intermedias combinan dichas características para identificar
estructuras más complejas, como formas, texturas o partes específicas de un objeto.
Finalmente, las capas más profundas integran la información previa y logran representar
entidades completas, como rostros, palabras o categorías semánticas. Cuanto mayor es la
profundidad de la red, mayor es la capacidad para modelar patrones de alta complejidad.

La principal motivación, utilizar funciones no lineales es que al final la regresión
lineal está limitada a poder combinar características de forma lineal, ya que no puede
crear relaciones entre características que no sean lineales. Un ejemplo muy común es el
de la puerta Xor donde a pesar de tener solo cuatro posibles combinaciones entre parejas
de dos valores, valorar el valor B, no se puede realizar mediante separaciones lineales
se necesitan realizar combinaciones no lineales al final una X sobres si es 00 es 0601
es uno si es 10 es uno y si es 11 0

Existe un teorema universal vale que es la aproximación universal de las redes
neuronales, y es que dice que múltiples capas sucesivas conectadas entre sí con una
determinada profundidad o anchura donde la profundidad sería el número de capas que
tiene la red y la anchura es el número de neuronas que tienes por capa con una cierta
función no lineal se puede llegar a aproximar cualquier función sin embargo esto es +1
teorema no practico.

### 4.1. De neuronas a redes neuronales

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
  Añadir capas ocultas hace que el problema de optimización de las redes neuronales al
  final se han no convexas que sea no conversa significa que puede tener múltiples
  mínimos. Hay que optimizar porque al final lo que se pretende en el proceso de
  entrenamiento u optimización de los modelos es alcanzar el mínimo error que maximice
  una cierta recompensa, entonces dependiendo de el número de capas que se añaden y la
  cantidad de parámetros que tenga la red, es decir, a mayor número de grados de
  libertad de la red, más no convexa es la función que tiene que optimizar, por lo que
  pequeños cambios en la inicialización de los parámetros, puede alterar el resultado de
  la red.

El cálculo de la salida de una red neuronal consiste en aplicar repetidamente la
operación de combinación lineal seguida de activación. La complejidad del aprendizaje
profundo aumenta con el número de capas y conexiones, incrementando la capacidad de
representación del modelo, pero también dificultando su interpretación.

Un concepto muy importante es la de composición que al final consiste en coger partes
que son mucho más complejas y que sean partes recursivas, es decir, qué pasos de temas
uno dependen de pasos anteriores, donde cada parte puede ser expresada de manera simple
con operaciones mucho más manejables este es el principal funcionamiento de la
composición de capas en redes neuronales, donde al final el funcionamiento de una
neurona o una red neuronal puede ser descompuestas como múltiples operaciones para
metalizadas una detrás de otra, esto se representa de la siguiente manera: f (x) = ( f2
◦ f1)(x) where f2 ◦ f1 is the composition of the two functions: ( f2 ◦ f1)(x) = f2(
f1(x)), al final podemos combinar tantas operaciones, siempre que esa función no cambien
el tipo de dato de los pasos anteriores, es decir que se conserve el tipo de datos. En
la principal cambio que se produce son los cambios de parámetros o ajuste de parámetros
en cada una de esas funciones al final cada una de esas funciones es como una capa del
modelo entonces al final están parametrizado de manera diferente a que tienen una
relación entre las diferentes funciones y se irán ajustando, pues dependiendo una de las
otras. Suponiendo que por ejemplo tenemos dos funciones lineales, dando una función y
depende de la función lineal anterior H. Al final al ese parámetro de H de la función
anterior es el parámetro dependiente que viene multiplicado por el bar por la matriz de
pesos de la siguiente función lineal a la que se le añade el sexo entonces al realizar
estas operaciones. Al final tienden a colapsar en una única función lineal. Por ello hay
que utilizar funciones no lineales para romper con este colapso.

### 4.2. Parámetros e hiperparámetros

En el entrenamiento de redes neuronales profundas resulta esencial distinguir entre
parámetros e hiperparámetros. Los **parámetros** incluyen los pesos y sesgos de la red,
los cuales se aprenden automáticamente mediante algoritmos de optimización. Los
**hiperparámetros**, en cambio, se definen antes del entrenamiento y controlan aspectos
estructurales y dinámicos del modelo. Entre ellos destacan la **tasa de aprendizaje**,
el número de iteraciones o épocas, la cantidad de capas ocultas, el número de neuronas
por capa y la elección de funciones de activación. La búsqueda de hiperparámetros
constituye un proceso iterativo en el que se combinan prueba y error con estrategias más
sistemáticas, con el fin de encontrar la configuración que produzca el mejor desempeño.

### 4.3. Funciones de activación

Las **funciones de activación** introducen no linealidad en la red neuronal, permitiendo
que el modelo aprenda relaciones complejas entre los datos. Sin funciones de activación,
una red neuronal se reduce a una combinación lineal de las entradas, comportándose de
manera similar a métodos clásicos no basados en redes neuronales. La elección de la
función de activación es fundamental y depende del tipo de capa y del problema a
resolver. La salidas de la función parametrizar de la neurona o red neuronal, pues se
conocerán como los Logits donde realizaremos una pequeña distinción entre la salidas pre
activa pre activación o Post activación, que son la salidas de la neurona renombró anal,
antes de aplicar, la no linealidad o la linealidad de la función de activación.

En las **capas ocultas**, se emplean funciones de activación como:

- **ReLU (Rectified Linear Unit)**: Es ampliamente utilizada en redes profundas, ya que
  acelera el entrenamiento y evita problemas de gradientes muy pequeños. No obstante,
  puede provocar **neuronas muertas**, que siempre devuelven cero. Para mitigar este
  efecto se utilizan variantes como _Leaky ReLU_, que mantiene un pequeño gradiente para
  valores negativos. Se representa :

  $$
  f(x) = \max(0, x).
  $$

- **Sigmoide**: Transforma los valores en el rango $[0,1]$. Se utiliza en redes
  recurrentes, aunque presenta el problema de **gradientes que desaparecen** en los
  extremos. Se representa como:

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

### 4.4. Implementación de una red neuronal

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

### 4.5. División del conjuntos de datos

En el entrenamiento de modelos de aprendizaje automático y, en particular, de redes
neuronales profundas, la gestión adecuada de los datos constituye un paso fundamental
para garantizar un proceso de optimización eficiente y una evaluación rigurosa del
rendimiento.

Como se mencionó anteriormente, uno de los enfoques empleados para agilizar el descenso
del gradiente es el descenso de gradiente estocástico. Esta técnica permite aplicar el
algoritmo de optimización no sobre la totalidad del conjunto de entrenamiento, sino
sobre subconjuntos de datos. Al evaluar el gradiente en un lote reducido, se obtiene
información temprana sobre el progreso de la optimización sin necesidad de procesar
todas las muestras, lo que facilita un aprendizaje más rápido y actualizaciones de los
parámetros del modelo con mayor frecuencia.

El uso de lotes resulta especialmente ventajoso en entornos con GPU, ya que estas
permiten almacenar los datos en memoria gráfica y ejecutar cálculos de manera altamente
paralelizada. El tamaño de los lotes depende principalmente de la capacidad de memoria
disponible, siendo comunes valores como 32, 64, 128 o superiores. Aunque frecuentemente
se prefieren potencias de dos, esta elección no conlleva necesariamente una mejora en la
eficiencia, por lo que pueden emplearse otros valores sin detrimento del rendimiento. En
general, se tiende a utilizar lotes tan grandes como lo permita la memoria, aunque el
tamaño seleccionado puede afectar las métricas de evaluación del modelo. Por ejemplo, en
arquitecturas basadas en _autoencoders_, se observa un mejor desempeño con lotes
pequeños, ya que esto limita la tendencia de la red a memorizar patrones específicos. En
contraste, en tareas supervisadas de clasificación de imágenes o en metodologías
contrastivas, los lotes más grandes suelen ser beneficiosos, ya que permiten calcular un
mayor número de métricas de distancia entre pares de muestras y construir matrices de
similitud más robustas, mejorando así la calidad de las representaciones aprendidas.

En contextos de aprendizaje auto-supervisado, el modo en que se agrupan las muestras en
lotes afecta directamente tanto a las funciones de coste como al proceso de
optimización. Esto se debe a que muchas de estas funciones se basan en medidas de
distancia entre elementos de un mismo lote. La inclusión de un mayor número de muestras
similares o variadas dentro de un lote puede modificar sustancialmente las estadísticas
empleadas en el aprendizaje, especialmente cuando se aplican técnicas de normalización
por lotes o por capas.

Incluso en modelos de lenguaje de gran escala se ha observado que la forma de dividir
los datos en lotes repercute en la salida final del modelo. A pesar de que las capas de
estos modelos son deterministas y que parámetros como la temperatura de muestreo podrían
sugerir resultados estables, en la práctica se generan salidas distintas para una misma
entrada. Parte de esta variabilidad se explica no solo por errores numéricos o de
redondeo, sino también por la manera en que se conforman los mini-lotes y las
distribuciones de las muestras que los componen.

De manera clásica, durante el desarrollo de modelos de aprendizaje automático los
conjuntos de datos se dividen en tres subconjuntos principales:

1. **Conjunto de entrenamiento**: Se emplea para ajustar los parámetros internos del
   modelo mediante el proceso de optimización. Puede contener o no etiquetas.
2. **Conjunto de validación**: Formado por ejemplos no utilizados en el entrenamiento
   directo. Su función es evaluar la capacidad de generalización del modelo y guiar la
   selección de hiperparámetros, reduciendo el riesgo de sobreajuste.
3. **Conjunto de prueba**: Reservado para la evaluación final y objetiva del modelo una
   vez completado el entrenamiento y optimizados los hiperparámetros.

La proporción destinada a cada subconjunto depende de la cantidad de datos disponibles.
Con bases de datos pequeñas, se suele aplicar una partición del 70 % para entrenamiento
y 30 % para prueba. En bases de datos más extensas, resulta común asignar un 60 % al
entrenamiento, 20 % a la validación y 20 % a la prueba. Es esencial que los subconjuntos
de validación y prueba sigan la misma distribución que los datos de entrenamiento, ya
que una discrepancia significativa puede generar degradaciones en las métricas de
evaluación.

En entornos de producción este problema es frecuente, dado que las distribuciones de los
datos suelen variar con el tiempo. Por ejemplo, en sistemas de telecomunicaciones, los
patrones de uso de los usuarios pueden cambiar drásticamente entre diferentes épocas del
año, lo que ocasiona desviaciones entre los datos de entrenamiento y los datos reales en
producción. Para detectar y cuantificar estas desviaciones se utilizan métricas como la
divergencia de Kullback-Leibler, la divergencia de Jensen-Shannon u otras medidas de
distancia entre distribuciones. Asimismo, técnicas como el análisis de entropía, los
_autoencoders_ o el Análisis de Componentes Principales (PCA) permiten medir errores de
reconstrucción y establecer umbrales (por ejemplo, basados en percentiles) para
identificar muestras fuera de distribución.

La detección de datos fuera de distribución constituye una línea de investigación activa
dentro del aprendizaje profundo, con aplicaciones en el aprendizaje activo, el _meta
learning_ y la mitigación del olvido catastrófico. Estas metodologías buscan adaptar los
modelos a nuevos datos de producción no observados previamente, ampliando su
conocimiento sin necesidad de reentrenarlos por completo. En ciertos casos, se investiga
la posibilidad de incorporar nuevos datos sin requerir acceso a los datos originales, lo
que resulta especialmente relevante en contextos donde estos ya no están disponibles. Un
ejemplo claro se encuentra en las telecomunicaciones, donde la transición de tecnologías
como 4G a 5G exige reutilizar modelos entrenados con datos de la generación anterior,
pese a que estos no puedan recuperarse, lo que demanda estrategias capaces de transferir
conocimiento sin perder la información previa.

Para agilizar el proceso de aprendizaje, se utilizan los mini lotes en el descenso del
gradiente estocástico, ya que es lo que te permite es tener una primera idea de la
función de pérdida o de costa en una parte reducida del conjunto de datos, sin tener que
esperar a procesar todos los datos en si esto es muy beneficioso, porque podemos hacer
Optimizaciones en pasos más pequeños que van a hacer una contribución global una vez
visualizado todos los datos sin embargo, utilizar tamaños de lotes o utilizar lotes
pequeños se tienen que asumir que los elementos son completamente diferenciados y únicos
unos de otros, lo que se conoce como i.i.d que si mano recuerdo, significa en
distribuciones independientes e identificables, .al final realizar el proceso de
aprendizaje por mini lotes, es como realizar una aproximación por Montecarlo, de la
función de pérdida o de coste, teniendo en cuenta todos los datos al completo del
conjunto de datos y esto es lo mismo para los gradientes, es decir, que calcular los
gradientes o la función de pérdida en la parte reducida del conjunto de datos, puede ser
lo suficientemente significativo para actualizar los parámetros de todos los datos de la
red, porque tengo una contribución global sin embargo, decir qué tamaños de lotes es
decir que la cantidad de elementos que hay en un lote que es pequeñito, eso supone que
existe una mayor varianza entre los diferentes datos o elementos de diferentes lotes, y
cuanto mayor es el número de elementos en un lote, pues menor será esa varianza, al
final eso se traduce en cómo de suave se optimiza la función de pérdida dependiendo
también mucho del tipo de arquitectura, modelo tipo de aprendizaje, tipo de datos, etc.
pero utilizar una cantidad de elementos en un lote. Al final actúa como un hiper
parámetros por ejemplo en autoencoders suelen ser bastante sensibles a la cantidad de
elementos por lote y se suelen utilizar menor cantidad de elementos porque tienden a
memorizar más. Luego las funciones de costes suelen ser más sensibles a la cantidad de
elementos. Suelen quedarse estancado es conforme, se aumenta el número de elementos,
etc. aunque también es una de las medidas que se suelen utilizar para elegir la cantidad
de elementos de un lote. Es lo máximo que ocupe en la memoria de la GPu. Lo que se
tiende a hacer es coger el set de entrenamiento. Al final se aplica un Shuffle aleatorio
de estos datos y se empiezan a hacer indexación de los datos por ejemplo si el vamos a
coger lotes de 32 ejemplos pues coges y las primeras 32 muestras para que formen parte
de un lote, luego los siguientes 32 ejemplos a los 32 anteriores, pues van para otro
lote y así sucesivamente esto es entrena con el descenso de gradiente estocástico y una
vez visualizado todos los lotes lo que se vuelve a hacer es coger ese conjunto de datos
original y se puede se vuelve a hacer un shuffle y se vuelve a obtener los lotes
aleatorios. Esta división de los lotes se pueden repartir entre múltiples nodos o GP V,
y es lo que se conoce como la paralización de datos. Normalmente lo que se tiene es el
conjunto de datos en memoria y se hace un Split, una división de cada uno de los lotes a
las diferentes GPS, entonces cada GPT procesa de manera independiente cada uno de estos
lotes, calculo los gradientes para esos lotes y posteriormente se esperan a que todos
los nodos calculen esos lotes, es decir, se hace una paralización de datos de manera
síncrona. Una vez calculados los gradientes se recopilan. Todos se agregan, se hace
algún tipo de operación de Broadcast para ajustarlos el formato y se actualizan los
parámetros globales de la red.

A modo de ejemplo aqui tenemos una implementacion utilizando pytorch:

```python
from torch.utils.data import DataLoader
dataloader = DataLoader(dataset, shuffle=True, batch_size=32)
for xb, yb in dataloader:
```

### 4.6. Sesgo y varianza

El análisis de sesgo y varianza constituye una herramienta fundamental para comprender
las fuentes de error en los modelos de aprendizaje automático. Ambos conceptos permiten
diagnosticar si un modelo está fallando por su incapacidad para representar
correctamente el problema subyacente o por su exceso de adaptación a los datos
disponibles.

El **sesgo** se define como la diferencia sistemática entre las predicciones del modelo
y los valores reales. Un sesgo alto indica la presencia de subajuste, lo que significa
que el modelo no logra capturar de manera adecuada la complejidad de la relación
existente en los datos. En este caso, las predicciones tienden a ser demasiado simples o
alejadas de la realidad, aun cuando se disponga de una gran cantidad de datos de
entrenamiento.

Por otro lado, la **varianza** mide la sensibilidad del modelo frente a pequeñas
variaciones en los datos de entrenamiento. Una varianza alta refleja la existencia de
sobreajuste, fenómeno en el que el modelo se ajusta excesivamente a los ejemplos
utilizados en el entrenamiento, incorporando incluso el ruido presente en ellos, lo que
impide una correcta generalización hacia datos nuevos.

La reducción del sesgo suele requerir un aumento en la capacidad de representación del
modelo. Esto puede lograrse mediante arquitecturas más profundas o complejas, el
incremento del número de parámetros, un mayor tiempo de entrenamiento o la adopción de
algoritmos alternativos que permitan capturar de manera más fiel las dependencias de los
datos. En contraste, para disminuir la varianza se recurre a estrategias orientadas a
mejorar la capacidad de generalización, tales como el incremento de la cantidad y
diversidad de datos de entrenamiento, la aplicación de técnicas de regularización (como
dropout, penalizaciones de norma $L_1$ o $L_2$), así como ajustes en la arquitectura y
en los hiperparámetros del modelo.

En la práctica, el análisis de sesgo y varianza se complementa con la noción de **techo
de referencia humano**, empleado para evaluar modelos cuyo desempeño se compara con el
nivel de expertos humanos en una tarea específica. En este marco, se introduce el
concepto de **sesgo evitable**, entendido como la diferencia entre el error mínimo
alcanzable por un ser humano y el error observado en el modelo. A su vez, la
**varianza** se cuantifica como la diferencia entre el error en el conjunto de
entrenamiento y el error en el conjunto de validación.

### 4.7. Desvanecimiento y explosión de gradientes

Uno de los principales desafíos en el entrenamiento de redes neuronales profundas es el
fenómeno conocido como desvanecimiento o explosión de gradientes. Ambos problemas se
presentan durante el proceso de _backpropagation_, cuando los gradientes de los
parámetros (es decir, las derivadas parciales de la función de pérdida respecto a los
pesos) tienden a disminuir hasta valores cercanos a cero o, por el contrario, a crecer
de manera exponencial. Esta inestabilidad dificulta o incluso imposibilita el
aprendizaje, ya que los parámetros no se actualizan de manera adecuada. En la práctica,
este comportamiento puede provocar que la función de pérdida devenga en valores **NaN**
(_Not a Number_), interrumpiendo el proceso de optimización.

El desvanecimiento de gradientes ocurre cuando los valores derivados se reducen
progresivamente en cada capa durante el proceso de _backpropagation_. Esto provoca que
las capas más cercanas a la entrada reciban señales de error muy débiles, limitando la
capacidad del modelo para aprender representaciones jerárquicas profundas. Por otro
lado, la explosión de gradientes se manifiesta cuando los valores derivados aumentan
exponencialmente a medida que se propagan hacia atrás, generando actualizaciones de
parámetros excesivamente grandes y conduciendo a un entrenamiento inestable o
divergente.

Diversos factores contribuyen a la aparición de estos problemas, destacando
especialmente el tipo de **inicialización de pesos**, la elección de las **funciones de
activación** y la **profundidad de la red**. En arquitecturas recurrentes tradicionales,
como las redes neuronales recurrentes simples (RNN) o incluso las LSTM, el cálculo
iterativo de múltiples derivadas sobre secuencias largas amplifica la tendencia a sufrir
desvanecimiento o explosión. Esto se debe, en gran medida, al uso de funciones de
activación como la tangente hiperbólica o la sigmoide, cuyos rangos limitados y simetría
alrededor de valores fijos provocan que los gradientes se saturen en los extremos,
reduciendo la señal útil para el aprendizaje.

Para mitigar estos fenómenos se emplean diversas estrategias:

- **Inicialización adecuada de los pesos**: Métodos como Xavier o He ajustan la escala
  inicial de los parámetros según la cantidad de neuronas por capa, evitando que los
  gradientes crezcan o decrezcan de manera descontrolada desde el inicio del
  entrenamiento.

- **Normalización de los datos de entrada**: Escalar las características de entrada para
  que tengan media cero y varianza unitaria contribuye a estabilizar el flujo de
  gradientes durante la retropropagación.

- **Funciones de activación más estables**: El uso de activaciones como ReLU y sus
  variantes (Leaky ReLU, ELU, entre otras) reduce la saturación observada en funciones
  como la sigmoide o la tangente hiperbólica, favoreciendo gradientes más consistentes.

- **Clipado de gradientes**: Consiste en limitar el rango de valores que pueden alcanzar
  los gradientes durante la retropropagación. Cuando los gradientes exceden un umbral
  predefinido, se ajustan a dicho límite, evitando actualizaciones excesivas. Es común
  emplear intervalos como $[-1, 1]$, aunque también existen variantes dinámicas que
  adaptan este rango según el estado del entrenamiento.

- **Diseño arquitectónico específico**: La introducción de mecanismos de memoria y
  compuertas en redes como LSTM o GRU permite manejar dependencias de largo plazo,
  reduciendo los problemas de desvanecimiento de gradientes. Más recientemente, los
  Transformers han reemplazado en gran medida a las RNN en el procesamiento de
  secuencias reduciendo estas limitaciones.

### 4.8. Estrategia en el proceso de optimización

El diseño de una estrategia adecuada en el desarrollo de modelos de aprendizaje
automático resulta crucial para alcanzar un rendimiento óptimo y garantizar la utilidad
práctica de los sistemas. No todas las mejoras introducidas durante el proceso de
construcción del modelo tienen el mismo impacto en su desempeño. En muchos casos,
incrementar la cantidad y diversidad de datos disponibles o modificar de manera
sustancial la arquitectura de la red genera beneficios mucho mayores que ajustes menores
sobre los hiperparámetros. Por este motivo, se requiere establecer métricas claras y
bien definidas que orienten las decisiones y permitan evaluar de manera objetiva cada
iteración del proceso.

Las métricas de evaluación dependen directamente del tipo de aprendizaje empleado
(supervisado, no supervisado o por refuerzo), aunque comparten el objetivo común de
cuantificar la calidad de las predicciones. En aprendizaje supervisado de clasificación,
destacan medidas como la **precisión**, que expresa la proporción de verdaderos
positivos entre todas las predicciones positivas, el **recall** o sensibilidad, que
calcula la proporción de verdaderos positivos sobre el total de casos positivos reales,
y la **puntuación F1**, definida como la media armónica entre la precisión y el recall,
que equilibra ambas perspectivas.

Más allá de las métricas de exactitud, es indispensable considerar indicadores de
**eficiencia computacional**, tales como el tiempo de entrenamiento, la latencia en la
inferencia, el consumo de memoria y la escalabilidad del modelo. Estas métricas permiten
valorar no solo el grado de acierto, sino también la viabilidad práctica de la solución
en contextos de producción. Asimismo, en entornos empresariales resulta común integrar
indicadores de **impacto económico y de experiencia de usuario**, como el retorno de
inversión, la satisfacción percibida, la calidad de las respuestas o los tiempos de
espera, de manera que el rendimiento técnico se vincule con objetivos estratégicos.

Para implementar una estrategia de aprendizaje coherente y sostenible, se recomienda
emplear plataformas especializadas en la gestión de experimentos. Estas herramientas
permiten registrar y organizar de forma sistemática todos los artefactos generados
durante el desarrollo del modelo, incluidos los parámetros de configuración, el número
de capas, las funciones de activación utilizadas, la arquitectura adoptada y las
métricas obtenidas en cada ejecución. De este modo, se garantiza la **replicabilidad de
los experimentos**, se facilita la **comparación justa entre diferentes
configuraciones** y se optimiza la toma de decisiones. Entre las plataformas más
utilizadas se encuentran **MLflow**, **Weights & Biases (wandb)** y soluciones
similares, que proporcionan entornos integrados para el seguimiento, análisis y
visualización de experimentos de aprendizaje automático.

### 4.9. Aprendizaje por transferencia

Además de las arquitecturas tradicionales, en el campo del aprendizaje profundo se han
desarrollado enfoques que no constituyen arquitecturas en sí mismas, sino **paradigmas
de aprendizaje** que buscan aprovechar de manera más eficiente los recursos
computacionales y los datos disponibles. Entre los más relevantes se encuentran el
**aprendizaje por transferencia** y el **aprendizaje multitarea**, ambos orientados a
mejorar la capacidad de generalización de los modelos y a reducir el coste de
entrenamiento.

El **aprendizaje por transferencia** consiste en reutilizar el conocimiento adquirido
por un modelo previamente entrenado en una tarea determinada para aplicarlo en otra
tarea relacionada. Por ejemplo, un modelo entrenado para clasificar imágenes en un
dominio amplio (como el conjunto de datos ImageNet) puede transferirse a un modelo más
específico, encargado de identificar defectos en piezas industriales o clasificar tipos
de cultivos a partir de imágenes satelitales. En este contexto, la similitud entre las
tareas es un requisito fundamental. No resulta viable transferir directamente el
conocimiento de un modelo entrenado en visión por computadora a uno diseñado para
procesar texto, ya que las representaciones internas aprendidas difieren por completo.

El grado de reutilización depende en gran medida de la disponibilidad de datos en la
nueva tarea. Cuando los datos son escasos, suele reajustarse únicamente la parte final
de la red, por ejemplo, las últimas capas densas de un clasificador. Mientras que el
resto de la arquitectura se congela, preservando así las representaciones generales
previamente aprendidas. En cambio, cuando se dispone de una cantidad suficiente de
datos, es posible aplicar un ajuste fino o **_fine-tuning_**, que consiste en reentrenar
toda la red para adaptar gradualmente los parámetros a las particularidades del nuevo
dominio.

## 5. Diferenciacion automatica

# Diferenciación Numérica, Simbólica y Automática: Una Explicación Integrada

La diferenciación numérica, simbólica y automática constituye un conjunto de enfoques
complementarios para obtener derivadas de funciones. Cada método se fundamenta en
principios distintos y presenta características particulares que determinan su
precisión, su coste computacional y su aplicabilidad. Comprender sus diferencias
requiere analizar cómo opera cada técnica, qué tipo de información utiliza y qué
compromisos establece entre exactitud y eficiencia.

## 1. Diferenciación Numérica

La diferenciación numérica aproxima la derivada a partir de valores concretos de la
función, sin manipular expresiones algebraicas ni reglas simbólicas. Se basa
directamente en la definición de derivada y sustituye el límite por un incremento finito
$h$ suficientemente pequeño. Esta aproximación adopta diversas formulaciones, entre
las cuales la más simple es la diferencia hacia adelante:

$$f'(x) \approx \frac{f(x+h) - f(x)}{h}.$$

Una versión más precisa es la diferencia centrada, que utiliza dos evaluaciones de la
función y presenta un error de orden $O(h^2)$:

$$f'(x) \approx \frac{f(x+h) - f(x-h)}{2h}.$$

El método opera exclusivamente con números y produce resultados aproximados cuya calidad
depende de la elección de $h$. Si $h$ es demasiado grande, la aproximación se
degrada; si es demasiado pequeño, emergen errores de redondeo asociados a la aritmética
de coma flotante. Además, cada derivada requiere varias evaluaciones de la función, lo
que vuelve esta técnica poco viable para problemas con grandes cantidades de variables,
como los modelos de aprendizaje profundo. Por ello se emplea sobre todo con fines de
validación o en contextos de baja dimensionalidad.

Para ilustrarlo, considérese la derivada de $\sin(x)$ en $x=1$. Con $h =
10^{-5}$, la diferencia hacia adelante produce un valor aproximado cercano a 0.5,
notablemente alejado de $\cos(1) \approx 0.5403$. En cambio, la diferencia centrada
ofrece un resultado mucho más cercano al valor exacto, del orden de 0.5400. Este
contraste refleja la sensibilidad del método a su formulación y a la elección de $h
\).

## 2. Diferenciación Simbólica

La diferenciación simbólica opera directamente sobre la expresión matemática de la
función y utiliza reglas formales de derivación para obtener una fórmula exacta. Dado un
ejemplo como

$$f(x) = a \sin(x) + bx\sin(x),$$

un sistema simbólico desarrolla la derivada aplicando la regla del producto y la
derivada del seno, lo que conduce a

$$f'(x) = a\cos(x) + b\sin(x) + bx\cos(x).$$

Este enfoque trabaja con símbolos en lugar de valores numéricos y permite obtener
derivadas sin aproximaciones. Sin embargo, al manipular expresiones complejas puede
generar fórmulas extremadamente grandes, fenómeno conocido como _expression swell_. Esta
explosión combinatoria limita su aplicación en programas extensos o en funciones
definidas de forma procedimental.

## 3. Diferenciación Automática

La diferenciación automática (AD) se sitúa conceptualmente entre los dos métodos
anteriores. No se basa en aproximaciones numéricas ni en transformaciones simbólicas
exhaustivas, sino en la evaluación sistemática de la estructura computacional de la
función. Aplica las reglas del cálculo diferencial durante la ejecución del programa y
propaga derivadas elementales a través de las operaciones que lo componen.

El resultado es exacto hasta los límites de la precisión de máquina, sin incurrir en
errores de aproximación como en la diferenciación numérica ni en crecimiento explosivo
de expresiones como en la simbólica. Además, su coste computacional es muy reducido. En
modo directo, el coste es proporcional al número de variables; en modo inverso,
utilizado en aprendizaje automático para implementar _backpropagation_, el coste es
comparable al de evaluar la propia función. Esta eficiencia explica su papel central en
la optimización de modelos contemporáneos.

## 4. Comparación de los Métodos

Aunque los tres enfoques comparten el objetivo de obtener derivadas, sus fundamentos y
resultados difieren. La diferenciación numérica utiliza exclusivamente números y
proporciona aproximaciones; la diferenciación simbólica manipula expresiones y produce
resultados exactos; la diferenciación automática combina precisión numérica y eficiencia
computacional sin generar fórmulas complejas. De este modo, la diferenciación automática
recoge ventajas de ambos extremos, pero no puede considerarse idéntica a ninguno.

## 5. Redes neuronales convolucionales

### 5.1. Procesamiento visual humano y su analogía con las redes neuronales convolucinales

El procesamiento visual humano es un proceso jerárquico que transforma la información
lumínica captada por los ojos en representaciones visuales complejas y significativas.
Este proceso involucra múltiples etapas funcionales que van desde la captación inicial
de la luz hasta el análisis de formas y objetos en áreas corticales especializadas.

La luz ingresa al ojo a través de la córnea y atraviesa el cristalino, el cual actúa
como una lente convexa que invierte la imagen proyectándola sobre la retina, ubicada en
la parte posterior del globo ocular. En la retina, los fotorreceptores (conos y
bastones) convierten la energía lumínica en señales eléctricas, iniciando así la
codificación neuronal de la información visual.

Estas señales se transmiten por el nervio óptico de cada ojo hasta el quiasma óptico,
donde ocurre un cruce parcial de la información visual, los campos visuales izquierdos
de ambos ojos se dirigen al hemisferio derecho, mientras que los campos visuales
derechos se proyectan al hemisferio izquierdo. Esta organización permite la percepción
binocular y contribuye a la percepción de profundidad.

Posteriormente, las señales continúan a través del tracto óptico hasta el núcleo
geniculado lateral (LGN) del tálamo, que funciona como estación de relevo y organiza la
información entrante. El LGN está compuesto por capas diferenciadas que procesan señales
provenientes de distintos tipos de células ganglionares, permitiendo un preprocesamiento
especializado que facilita el análisis visual posterior.

Desde el LGN, las señales visuales se transmiten mediante las radiaciones ópticas hacia
la corteza visual primaria (V1), localizada en el lóbulo occipital. La V1 se organiza de
manera retinotópica, de modo que cada región del campo visual se proyecta a un área
cortical específica. En esta región, las neuronas responden a patrones visuales básicos
mediante campos receptivos, que representan pequeñas regiones del espacio visual a las
que responden selectivamente.

En la corteza visual primaria se distinguen tres tipos principales de células. Células
simples, que responden a bordes con orientación específica, células complejas, que
detectan bordes o movimientos en rangos más amplios, y células hipercomplejas, que
reaccionan ante combinaciones más sofisticadas, como esquinas o terminaciones de líneas.
El procesamiento continúa en áreas corticales posteriores (V2, V4, IT), donde se
analizan características más complejas, incluyendo texturas, formas tridimensionales,
rostros y objetos completos.

Las redes neuronales convolucionales, también conocidas como **_Convolutional Neural
Networks_ (CNNs)**, son modelos computacionales diseñados para procesar datos visuales
de manera eficiente, inspirados directamente en la arquitectura del sistema visual
humano, especialmente en la corteza visual primaria. En este contexto, se establecen
correspondencias claras entre las estructuras biológicas y los componentes de una red
convolucional:

| Sistema Visual Humano                      | Redes Convolucionales (CNNs)                                     |
| ------------------------------------------ | ---------------------------------------------------------------- |
| Retina                                     | Imagen de entrada                                                |
| Nervio óptico / Quiasma óptico             | Preprocesamiento y alineación de la información visual           |
| LGN (núcleo geniculado lateral)            | División en canales o filtros por tipo de característica         |
| Corteza visual (V1, V2, V4, IT)            | Capas convolucionales jerárquicas                                |
| Células simples, complejas, hipercomplejas | Filtros convolucionales de bajo, medio y alto nivel              |
| Campos receptivos                          | Regiones locales (_receptive fields_) de los filtros (_kernels_) |
| Percepción jerárquica                      | Aprendizaje progresivo de características visuales               |

En las CNNs, cada unidad procesa únicamente una región limitada de la imagen, análoga a
los campos receptivos de las neuronas en V1. Los filtros convolucionales permiten
detectar bordes, texturas, formas y patrones complejos, emulando las funciones de las
células visuales especializadas. La estructura jerárquica de las CNNs permite que las
primeras capas capturen patrones simples, las intermedias estructuras más complejas y
las últimas integren estos elementos para identificar objetos completos. Adicionalmente,
técnicas como el _max pooling_ reducen la dimensionalidad de manera selectiva,
conservando información relevante, similar al resumen jerárquico que realiza el cerebro
al procesar escenas visuales complejas.

### 5.2. Campo receptivo y jerarquía de procesamiento visual

El campo receptivo se define como la región del campo visual que influye directamente en
la actividad de una neurona específica. En las etapas iniciales del procesamiento
visual, como en V1, los campos receptivos son pequeños y especializados en detectar
patrones simples, como líneas u orientaciones. A medida que se avanza jerárquicamente en
la corteza visual, los campos receptivos se expanden y se vuelven más complejos,
integrando información de múltiples regiones para formar representaciones más abstractas
y globales. Por ejemplo, una neurona en una capa inicial de un sistema visual podría
tener un campo receptivo de 3×3 píxeles, mientras que en capas sucesivas la combinación
de varios campos receptivos previos permite formar unidades con campos más amplios, como
5×5 o 7×7 píxeles. Esta organización favorece la abstracción progresiva y la
especialización en el análisis visual.

Podremos simular el campo receptivo en la redes, convolucionales definiendo ceros en la
matrices de pesos de aquella región de píxeles que están fuera de la región de interés
es decir, si tú tienes una imagen y tiene sus matrices tienes o tienes 1 px tienes una
cierta relación con tus píxeles vecinos entonces para eliminar la dependencia de ciertos
píxeles vecinos con respecto al píxel que estás evaluando ahora pues lo que haces es
colocar un cero en la matriz de pesos porque estás haciendo una suma ponderada o una
composición ponderada de la importancia de todos los píxeles que hay con respecto al
píxel que estás evaluando ahora entonces si colocas un cero pues eliminas esa
importancia al final esto se puede ver como un grafo conectado donde cada píxel es un es
un nudo y cada nodo de o sea cada píxel está relacionado con el resto de píxeles, que es
el resto de nuevos del brazo, cuyos enlaces o cuyas conexiones se realizan dependiendo
de la importancia que tengan unos píxeles con respecto al otro al final es un grafo que
está completamente conectado, es decir cada píxel tiene cierta relación con el resto de
píxeles, pero que el enlace a la importancia pues dependerá un poco de la distancia que
tenga.

Hence, the receptive field increases linearly in the number of convolutional layers.
This motivates our notion of locality: even if a single layer is limited in its
receptive field by the kernel size, a sufficiently large stack of them results in a
global receptive field.

### 5.3. Conceptos fundamentales de la convolución

La visión computacional constituye uno de los campos más dinámicos y transformadores de
la inteligencia artificial. A través de ella se han desarrollado aplicaciones que
abarcan desde la conducción autónoma hasta el reconocimiento facial, la clasificación
automática de imágenes y la segmentación de objetos en entornos complejos. La relevancia
de este ámbito es tal que sus fundamentos han trascendido el análisis de imágenes,
inspirando avances en disciplinas distintas, como el procesamiento del lenguaje natural
o el reconocimiento de voz. Pueden trabajar con datos secuenciales de cualquier tipo,
utilizando dos características que son fundamentales, la capacidad de localización y el
compartimiento de parámetros, esperemos que esto es lo que permite a las como los genes
a la operación, como tal tener cierta en varianza a la traslación de la ventana, que es
el propio kernel o filtro o matriz de pesos que aprende la capa de como lo son, aunque
en realidad, esta propiedad se rompe cuando se combinan con otras operaciones, que
veremos más adelante como mecanismos de pooling, que rompen con Line varianza a los
desplazamientos.

El principio subyacente que permite esta transferencia de conocimiento es la existencia
de una estructura espacio-temporal en los datos. En el caso de las imágenes, esta
estructura se refleja en la disposición relativa de los píxeles. Si se logra transformar
otros tipos de datos en representaciones visuales que conserven dicha organización, es
posible aplicar arquitecturas convolucionales de manera eficaz. Un ejemplo de este
enfoque se observa en la conversión de series temporales en imágenes mediante técnicas
como los _Gramian Angular Fields_. Este método transforma la serie temporal en
coordenadas polares y genera una matriz de ángulos que produce una imagen con
información espacio-temporal equivalente a la contenida en la secuencia original. De
modo similar, las señales de audio pueden convertirse en espectrogramas de tipo _Mel_,
lo que permite aprovechar las propiedades de las redes convolucionales para tareas de
clasificación, identificación o análisis acústico.

El principal desafío al trabajar con imágenes radica en la elevada cantidad de
información que contienen. Considérese, por ejemplo, una imagen en color de 64 × 64
píxeles. Dado que cada píxel posee tres componentes de color (rojo, verde y azul), la
representación requiere tres matrices de 64 × 64 valores, lo que equivale a **12288
entradas** para la red neuronal. Introducir directamente esta cantidad de datos en una
arquitectura tradicional obligaría a disponer de capas iniciales con decenas de miles de
neuronas, lo que genera un coste computacional muy alto y un elevado riesgo de
sobreajuste a medida que la resolución de las imágenes aumenta.

La solución a este problema se encuentra en la operación de **convolución**. Este
procedimiento aplica filtros, también denominados _kernels_, que recorren la imagen en
busca de patrones característicos como bordes, esquinas o texturas. El resultado de cada
aplicación es un **mapa de características**, que cuantifica la presencia del patrón
detectado en diferentes regiones de la imagen. Una propiedad fundamental de este
mecanismo es la **invarianza al desplazamiento**, que permite reconocer un mismo patrón
independientemente de su ubicación. Es importante señalar que esta propiedad se
manifiesta de manera estricta únicamente cuando la convolución se realiza con un tamaño
de paso igual a uno.

Los filtros que aprende una red convolucional pueden compararse con operadores clásicos
de detección de bordes, como Sobel o Scharr, capaces de identificar direcciones
verticales u horizontales. Sin embargo, la principal ventaja de las redes
convolucionales es que los filtros no se definen manualmente, sino que sus valores se
aprenden automáticamente mediante el algoritmo de _backpropagation_. Gracias a ello, el
modelo adquiere la capacidad de descubrir patrones mucho más complejos y adaptados a la
tarea específica.

A medida que la información avanza a través de las capas convolucionales, el tamaño
espacial de las representaciones disminuye mientras que el número de canales se
incrementa. Este proceso permite capturar progresivamente patrones de mayor nivel de
abstracción, que van desde contornos simples hasta estructuras semánticamente
significativas. Finalmente, las capas totalmente conectadas integran la información
extraída para producir la predicción final, que puede consistir en clasificar una
imagen, reconocer un objeto o identificar un rostro.

En sistemas de procesamiento de imágenes al final también aparecen nuevos conceptos como
la influencia que tiene 1 px en relación con el resto de píxeles de una imagen donde la
influencia de 1 px suele con bueno con respecto a sus vecinos o con respecto al resto de
píxeles de la imagen suele reducirse con el incremento de la distancia, es decir, que la
mayor distancia entre píxeles menor correlación habrá entre ellos y por tanto menos
importancia puede existir. Esto es importante en mecanismos actuales donde se utilizan
por ejemplo mecanismos de atención para ver la importancia que tiene una serie de
píxeles vecinos en un determinado píxel y con ello poder tener un mejor entendimiento de
la semántica de la imagen. Por tanto, podemos decir que las imágenes tienen cierta
localidad.

From the point of view of signal processing, equation (E.7.7) corresponds to a filtering
operation on the input signal X through a set of finite impulse response (FIR) filters
[Unc15], implemented via a discrete convolution (apart from a sign change). Each filter
here corresponds to a slice W:,:,:,i of the weight matrix. In standard signal
processing, these filters can be manually designed to perform specific operations on the
image. In convolutional layers, instead, these filters are initialized randomly and
trained via gradient descent. We consider the design of convolutional models built on
convolutional layers in the next section. An interesting aspect of convolutional layers
is that the output maintains a kind of “spatial consistency” and it can be plotted: we
call a slice H:,:,i of the output an activation map of the layer, representing how much
the specific filter was “activated” on each input region. We will consider in more
detail the exploration of these maps in the next volume.

Las capas como los finales en realidad si pueden procesar datos de entrada de cualquier
dimensión porque no dependen del tamaño de la entrada. Sin embargo, la práctica suele
ser bastante complicado procesar esto porque al final no podemos pasar de una manera
actualizada tensores que tengan diferentes tamaños y luego también aparecen problemas
como el olvido catastrófico porque al final estás entrenando con distribuciones de datos
que son completamente diferentes porque varían la el tamaño. Al final si utilizas
imágenes y tienen diferencias de tamaño pues intervienen diferentes tipos de texturas
patrones también puede haber diferentes cambios en la relaciones de los píxeles vecinos.
Pueden verse alterado las componentes de alta frecuencia que pueden alterar el
comportamiento de la arquitectura.

### 5.4. Componentes de una capa convolucional

El uso de convoluciones en redes neuronales introduce una serie de elementos esenciales
que determinan el comportamiento y la eficacia del modelo. Uno de ellos es el **relleno
(_padding_)**, que consiste en añadir bordes artificiales alrededor de la imagen para
evitar la pérdida de información en los márgenes y mantener las dimensiones originales
de la entrada. Este procedimiento resulta necesario porque, a medida que se aplican
operaciones de convolución, las dimensiones de las representaciones intermedias tienden
a reducirse respecto a la imagen inicial. El relleno garantiza que el tamaño de salida
coincida con el de entrada, lo que permite preservar información espacial en las etapas
iniciales del procesamiento.

Otro concepto clave es el **desplazamiento (_stride_)**, que define el número de píxeles
que el filtro avanza en cada paso al recorrer la imagen. Un valor de _stride_ mayor
reduce las dimensiones de la salida y, en consecuencia, disminuye el número de cálculos
necesarios. Cuando el tamaño del _stride_ coincide con el del filtro, el proceso es
equivalente a dividir la imagen en fragmentos independientes (_patches_), lo que aísla
secciones completas y permite analizarlas de manera separada. The definition is only
valid for pixels which are at least k steps away from the borders of the image: we will
ignore this point for now and return to it later. Each patch is of shape (s, s, c),
where s = 2k + 1, since we consider k pixels in each direction along with the central
pixel. For reasons that will be clarified later on, we call s the filter size or kernel
size. Consider a generic layer H = f (X ) taking as input a tensor of shape (h, w, c)
and returning a tensor of shape (h, w, c′). If the output for a given pixel only depends
on a patch of predetermined size, we say that the layer is local. Este concepto de los
fragmentos independientes o los patches es lo que ha dado arquitecturas basa más
avanzadas basadas en Transformers, donde lo que se hace es una descomposición en sus
matrices de una imagen donde cada su matriz es una representación numérica de un tensor
lo que pierde información de la semántica de la imagen, convirtiéndose en un toquen y
eso es lo que se basa en arquitecturas como VIT. Podemos realizar este tipo de
arquitecturas también utilizando combo lución es con tamaño de paso igual al tamaño del
kernel y con ello podemos realizar subdirecciones de la imagen en sus matrices de manera
eficiente.

En el caso de imágenes en color, los filtros no se limitan a ser matrices
bidimensionales, sino que se extienden a tres dimensiones para recorrer simultáneamente
los canales rojo, verde y azul. Este aspecto resulta especialmente relevante porque el
número de parámetros de una capa convolucional depende únicamente del tamaño y la
cantidad de filtros, y no de las dimensiones de la imagen de entrada. De este modo, se
logra una gran eficiencia en comparación con las redes totalmente conectadas. Por
ejemplo, una capa con 10 filtros de 3 × 3 × 3 requiere solo 280 parámetros, cifra muy
reducida frente a los millones de conexiones que implicaría una arquitectura densa de
tamaño equivalente.

No obstante, conviene señalar que, al introducir variaciones como _stride_, _padding_ o
capas densas posteriores, se puede perder parcialmente la invarianza al desplazamiento
que caracteriza a la convolución estándar con _stride_ igual a uno. En consecuencia, una
misma imagen trasladada circularmente hacia la izquierda o la derecha no siempre genera
representaciones idénticas a las de la imagen original.

Las convoluciones resultan efectivas por dos motivos principales. En primer lugar,
permiten una **reducción drástica del número de parámetros**, lo que simplifica el
entrenamiento y reduce el riesgo de sobreajuste. En segundo lugar, implementan la
**compartición de parámetros**, ya que un patrón aprendido en una región de la imagen
puede aplicarse en cualquier otra, lo que favorece la capacidad de generalización del
modelo.

Tras la convolución, suele aplicarse una etapa de **agrupamiento (_pooling_)**,
destinada a reducir las dimensiones intermedias y aportar robustez frente a pequeñas
variaciones espaciales. Esta operación contribuye, en muchos casos, a recuperar
parcialmente la invarianza al desplazamiento. La técnica más extendida es el **_max
pooling_**, que selecciona el valor máximo dentro de cada región, priorizando la
detección de la presencia de una característica por encima de su ubicación exacta. Otra
variante frecuente es el **_average pooling_**, que sustituye cada región por el valor
promedio de sus elementos, ofreciendo una representación más suavizada de la
información. Los mecanismos de Poulin como mecanismos globales al final de eliminan o
destruyen la información espacial. Para ello se implementaron técnicas parciales como el
Max Pulín, que permite reducir la resolución espacial a la mitad manteniendo el número
de canales intacto al final lo que se hace es que por cada uno de los canales o mapas de
características que sean o que se tienen durante el proceso de entrenamiento para cada
capa, pues se coge y se elige los valores máximos, dependiendo del tamaño del filtro de
este mecanismo de Pooling por ejemplo, si tenemos una imagen de 4x4 y tenemos un
mecanismo de Max bullying de 2 × 2 pues al final cogeremos sus matrices de esa imagen de
4x4 de 2 × 2, o sea que cogeremos como mucho cuatro elementos de esos cuatro elementos
que se cogen de la imagen, pues elige el valor máximo y ese es el valor resultante, y
así se hace con todos los píxeles de una imagen de 4x4, dependiendo también del
desplazamiento que se tenga del Stride dónde podemos utilizar pasos de uno o pasos de
dos dependiendo también si queremos tener en cuenta o mantener cierta localidad o
relación entre los píxeles.

Podríamos utilizar los datos de entrada de manera aplanada, es decir, de una manera
vector izada, en vez de utilizar como lociones para poder utilizar redes neuronales
completamente conectabas y sin embargo, esto es una manera bastante mala, porque se
pierden propiedades importantes de las capas anteriores que es lo que se conoce como la
Composability, por eso, en la gran mayoría de arquitecturas modernas que se basan en la
combo lución, ya no usan las capas finales para las predicciones o es bastante raro
utilizar mecanismos de aplanamiento de los tensores para pasar de tensores en cuatro
dimensiones, teniendo en cuenta el lote y se utilizan mecanismos como global Average
pooling, porque son invariantes a los desplazamientos en comparación a utilizar redes
neuronales, completamente conectados permite tener una mejor información a nivel local y
global de cada uno de los mapas de características aprendidos de la red convolucional es
decir, que tiene en cuenta información espacial, no la rompe. Además utilizar además
permite reducir la cantidad de parámetros que tiene la red porque al final cuando tienes
una imagen la plana si utilizas múltiples redes neuronales pues tienes que tener vas a
tener una conexión de cada uno de los píxeles con la siguiente neurona entonces al final
si tú tienes una imagen de 24 × 24 px vas a tener esa cantidad total de píxeles que
están conectadas a una única neurona donde cada conexión es una matriz de pesos y sesgos
si a eso se le suma que todos esos píxeles tienen que conectarse con todas las neuronas,
una misma capa el número de parámetros incrementa y se hace ineficiente. Al final para
cada píxel y cada canal de la salida sería una combinación ponderada de todos los
canales y todos los píxeles en la entrada de la imagen. Esa X varianza en la traslación,
pues se origina gracias al compartimiento de pesos, entre los diferentes filtros que
aprende la comvolucion, al final lo que aprendes en la parte de una imagen, se traslada
a otra parte de la imagen, hay como una transferencia del conocimiento.

Este mecanismo es muy eficiente, porque al al final el proceso de las capas como lución
Álex no dependen del tamaño de la imagen de entrada, sino que depende de el tamaño del
filtro, que se va a utilizar para el aprendizaje y del número de canales que se van a
utilizar, es decir, el número de filtros que quieres aprender para cada uno de los
filtros utilizados.

También las capas combo lución Álex no suelen utilizarse durante toda la arquitectura o
creación de los modelos. Normalmente se utilizan dos partes una parte que se conoce como
el Backbone que sería como el esqueleto del modelo y luego tenemos la parte utilizada
para clasificación regresión o para la tarea que subyace no hay problema entonces
utilizamos estas técnicas de combo lución principalmente para extraer características de
los datos de entrada y luego utilizamos las capas finales para realizar la tarea en
cuestión. Esta capa final utilizada para la tarea se conoce como la cabeza de la
arquitectura del modelo y consiste básicamente en utilizar redes neuronales
completamente conectadas. Lo que se hace es pasar de mapas de características obtenidos
de la convolucion donde personalmente suelo utilizar global a Break bullying que
consiste en realizar el promedio global de los valores de cada mapa de características
para tener un único valor por cada uno de las capas que se obtienen al final lo que
significa que si tenemos por ejemplo al final de la arquitectura, mapas de
características de 3 × 3 y 64 mapas, pues tendremos un tensor un único tensor que
dependerá del tamaño del lote y de eso 64 mapas de características, un valor por cada
mapa y eso se puede conectar a una red neuronal y se reduce la cantidad de conexiones o
número de parámetros, si utilizásemos técnicas de aplanamiento, además de no perder
información espacial.

### 5.5. Operaciones convolucion

First, consider a convolutional layer with k = 0, i.e., a so-called 1 × 1 convolution.
This corresponds to updating each pixel’s embedding by a weighted sum of its channels,
disregarding all other pixels: Hi jz = cX t=1 Wz t Xi j t It is a useful operation for,
e.g., modifying the channel dimension esto se conoce como point-wise convolution.

Second, consider an “orthogonal” variant to 1 × 1 convolutions, in which we combine
pixels in a small neighborhood, but disregarding all channels except one: Hi jc = 2k+1X
i′=1 2k+1X j′=1 Wi′, j′,c Xi′+t(i), j′+t( j),c where t(•) is the offset defined in
(E.7.6). In this case we have a rank-3 weight matrix W of shape (s, s, c), and each
output channel H:,:,c is updated by considering only the corresponding input channel
X:,:,c . This is called a depthwise convolution, and it can be generalized by
considering groups of channels, in which case it is called a groupwise convolution (with
the depthwise convolution being the extreme case of a group size equal to 1).

We can also combine the two ideas and have a convolution block made of alternating 1 × 1
convolutions (to mix the channels) and depthwise convolutions (to mix the pixels). This
is called a depthwise separable convolution and it is common in CNNs targeted for
low-power devices como mobilenet.

### 5.5. Evolución de las arquitecturas

El incremento en la profundidad de las redes neuronales ha permitido avances
significativos en la visión computacional. Sin embargo, este aumento también genera un
problema conocido. A partir de cierto punto, el rendimiento no mejora, sino que se
degrada. La causa principal radica en fenómenos como la desaparición o explosión de
gradientes, que dificultan el ajuste de los parámetros durante el entrenamiento y
limitan la capacidad de la red para aprender de manera efectiva.

La solución a este desafío surgió con la introducción de las **redes residuales
(ResNet)**. Estas arquitecturas incorporan **conexiones de atajo (_skip connections_)**,
que transmiten directamente las activaciones de una capa hacia otra más profunda, como
si se establecieran puentes dentro de la red. En consecuencia, cada bloque residual no
aprende únicamente una transformación completa, sino la diferencia (_residuo_) entre la
entrada y la salida esperada. Este diseño facilita el flujo de gradientes, permite
entrenar redes mucho más profundas y marcó un hito en el desarrollo de la visión
computacional, siendo la base de numerosos modelos posteriores.

Otra innovación relevante fue la **arquitectura Inception**, implementada en modelos
como **GoogLeNet**. Su principio fundamental consiste en aplicar en paralelo filtros de
distintos tamaños (1×1, 3×3 y 5×5), junto con una operación de _pooling_, y concatenar
los resultados obtenidos. Esta estrategia permite capturar información a diferentes
escalas espaciales, favoreciendo la detección de patrones tanto locales como globales.
Para controlar el coste computacional asociado a este diseño, se introdujeron
convoluciones de 1×1 que actúan como cuellos de botella, reduciendo la dimensionalidad
de los datos antes de aplicar filtros más grandes. De este modo, la arquitectura logra
un equilibrio entre expresividad y eficiencia.

Con la expansión de los dispositivos móviles y las aplicaciones en tiempo real, surgió
la necesidad de arquitecturas más livianas y rápidas. En este contexto aparecieron las
**MobileNet**, basadas en el concepto de **convoluciones separables en profundidad**.
Este enfoque divide el proceso en dos etapas:

1. **Convolución en profundidad (_depthwise convolution_):** Cada filtro se aplica de
   manera independiente a un canal de la entrada.
2. **Convolución puntual (1×1, _pointwise convolution_):** Combina los resultados de
   todos los canales para generar una representación integrada.

Gracias a esta separación, se logra una reducción drástica en el número de operaciones y
parámetros, lo que convierte a MobileNet en una arquitectura altamente eficiente.

La segunda generación, **MobileNetV2**, incorporó **conexiones residuales** junto con
**capas de expansión** mediante filtros 1×1, que permiten aumentar la capacidad de
representación sin comprometer la eficiencia.

### 5.6. Sistemas de detección de objetos

En muchas aplicaciones de la visión computacional, como la conducción autónoma o la
vigilancia inteligente, no basta con clasificar una imagen en su conjunto. Es
imprescindible identificar **qué objetos aparecen en la escena y en qué lugar se
encuentran**. Este desafío se aborda mediante la **detección de objetos**, un problema
que combina simultáneamente la **clasificación** y la **localización** de los elementos
presentes a través de recuadros delimitadores (_bounding boxes_).

En el caso más simple, se entrena un modelo para detectar un único objeto en cada
imagen. Este modelo debe predecir tres aspectos fundamentales:

1. La probabilidad de presencia del objeto.
2. Las coordenadas del recuadro delimitador.
3. La clase a la que pertenece el objeto detectado.

Sin embargo, la mayoría de escenarios reales presentan múltiples objetos de diferentes
clases y tamaños. Para abordar esta complejidad, una estrategia habitual consiste en
dividir la imagen en una **malla de celdas**, donde cada celda se encarga de predecir la
presencia de objetos cuyo centro se encuentra en su interior, junto con las coordenadas
y la categoría correspondiente.

Uno de los algoritmos más influyentes en este ámbito es **YOLO (_You Only Look Once_)**,
que aplica la red convolucional a toda la imagen de manera simultánea. Gracias a este
diseño, el modelo puede realizar detecciones en tiempo real, lo que lo convierte en una
solución idónea para aplicaciones prácticas donde la velocidad es un requisito crítico.

El desempeño de los modelos de detección se evalúa y optimiza mediante diferentes
métricas y técnicas:

- **Intersección sobre Unión (IoU):** Mide la calidad de la predicción comparando el
  grado de solapamiento entre la caja predicha y la caja real. Un valor más alto de IoU
  indica una localización más precisa.
- **Supresión de No Máximos (NMS):** Elimina predicciones redundantes en torno a un
  mismo objeto, conservando únicamente la más confiable.
- **Cajas de Anclaje (_anchor boxes_):** Permiten que cada celda prediga múltiples
  objetos con distintas proporciones y escalas. Estas cajas se definen habitualmente a
  partir de algoritmos de agrupamiento, como _k-means_, que identifican formas y tamaños
  más representativos en los datos de entrenamiento.

Además de la detección convencional basada en recuadros, existen variantes del problema
orientadas a tareas más específicas. Entre ellas destacan:

- **Detección de puntos de referencia:** En lugar de delimitar áreas rectangulares, se
  predicen coordenadas concretas, como las posiciones de rasgos faciales o
  articulaciones en el cuerpo humano.
- **Métodos basados en regiones:** Generan propuestas de posibles áreas de interés en la
  imagen, que posteriormente se clasifican. Este enfoque suele lograr mayor precisión,
  aunque con un coste computacional más elevado.

### 5.7. Segmentación semántica y convoluciones transpuestas

La **segmentación semántica** constituye una de las tareas más avanzadas de la visión
computacional, ya que no se limita a identificar qué objetos aparecen en una imagen,
sino que asigna una **clase específica a cada píxel**. El resultado es un mapa detallado
que representa con precisión la forma y extensión de cada objeto presente en la escena.
Esta técnica resulta fundamental en campos como la medicina, donde se emplea para
delimitar órganos o tumores en imágenes médicas, en la agricultura, para el análisis de
cultivos, y en la robótica, para la percepción precisa del entorno.

Para alcanzar este nivel de detalle, es necesario reconstruir la resolución espacial
original de la imagen a partir de representaciones comprimidas obtenidas en las etapas
iniciales de la red. Esta tarea se logra mediante la **convolución transpuesta**,
también conocida como _deconvolución_. A diferencia de la convolución tradicional, que
reduce progresivamente las dimensiones espaciales, la convolución transpuesta expande
dichas dimensiones, incrementando la resolución hasta aproximarse a la escala original
de la entrada.

Un hito en este ámbito lo constituye la **arquitectura U-Net**, diseñada inicialmente
para aplicaciones médicas, pero posteriormente adoptada en numerosos contextos. Esta
arquitectura se estructura en dos fases principales:

- **Etapa de compresión (_encoder_):** Reduce la resolución de la imagen y aumenta el
  número de canales, con el objetivo de extraer representaciones cada vez más abstractas
  y ricas en información semántica.
- **Etapa de expansión (_decoder_):** Recupera la resolución original mediante
  convoluciones transpuestas, reconstruyendo la información espacial a partir de las
  características extraídas.

Para mitigar la pérdida de detalles espaciales ocasionada por la compresión, U-Net
incorpora **conexiones de omisión (_skip connections_)**, que transfieren información
directamente desde las capas de compresión a las capas de expansión correspondientes.
Gracias a estas conexiones, la red conserva detalles finos de bordes y contornos,
alcanzando segmentaciones de alta precisión.

La salida de una arquitectura de segmentación puede variar según el problema abordado.
En casos simples, puede consistir en un único recuadro, en otros, en un conjunto de
coordenadas (por ejemplo, 2·n para la localización de _n_ puntos de referencia), y en
aplicaciones más complejas, en una máscara completa de segmentación que clasifica cada
píxel de manera independiente.

### 5.8. _One-Shot Learning_

El cerebro humano posee una notable capacidad para reconocer objetos a partir de muy
pocos ejemplos, fenómeno conocido como _few-shot_ o _one-shot learning_. Esta habilidad
se sustenta en dos mecanismos principales. Primero, la generación de representaciones
jerárquicas generalizables, que no dependen de la memorización exacta de píxeles, sino
de la extracción de características abstractas como formas, colores, movimientos y
estructuras espaciales. Estas representaciones se construyen progresivamente a lo largo
de la jerarquía cortical, de V1 a IT, donde las regiones superiores codifican objetos
completos y conceptos visuales. Segundo, la asociación de estas representaciones con
memoria y significado, mediante la interacción con áreas como la corteza prefrontal, el
hipocampo y otras estructuras del sistema límbico, permite reconocer un objeto
previamente visto incluso cuando se presenta en condiciones distintas de orientación,
tamaño o estilo visual.

Los modelos de visión por computador suelen requerir grandes volúmenes de datos para
alcanzar un entrenamiento eficaz. Sin embargo, en numerosos escenarios prácticos solo se
dispone de un número muy reducido de ejemplos por clase. Este desafío se aborda mediante
técnicas como el **_One-Shot Learning_** (aprendizaje con una sola muestra) o el
**_Few-Shot Learning_** (aprendizaje con pocas muestras), que buscan dotar a los
sistemas de la capacidad de generalizar a partir de datos escasos.

El principio fundamental de estos enfoques consiste en aprender un **espacio de
representación** en el que las imágenes similares se ubiquen próximas entre sí, mientras
que las correspondientes a clases distintas aparezcan más alejadas. La proximidad o
distancia entre representaciones puede evaluarse con métricas como la **distancia
euclidiana**, aunque también se emplean otras funciones de similitud más sofisticadas
según el problema.

Una de las arquitecturas más representativas en este ámbito son las **redes siamesas**,
que procesan en paralelo dos imágenes mediante una misma red convolucional que comparte
parámetros. El resultado son vectores de características que pueden compararse
directamente para decidir si ambas imágenes pertenecen o no a la misma clase. Este
diseño permite que la red aprenda relaciones de similitud sin necesidad de entrenar un
clasificador rígido para cada categoría.

Otra estrategia ampliamente utilizada en este contexto es la basada en la **pérdida
triple (_triplet loss_)**, que organiza el entrenamiento a partir de tríos de imágenes:

- **Anchor:** La imagen de referencia.
- **Positiva:** Una imagen de la misma clase que el _anchor_.
- **Negativa:** Una imagen de una clase distinta.

El objetivo de esta función de pérdida es reducir la distancia entre el _anchor_ y la
muestra positiva, al tiempo que incrementa la distancia entre el _anchor_ y la negativa.
Este mecanismo genera representaciones más robustas y discriminativas, capaces de
capturar similitudes profundas entre las imágenes.

Las aplicaciones del _One-Shot Learning_ son amplias y de gran relevancia. En el
reconocimiento facial, por ejemplo, permite identificar personas a partir de una sola
fotografía de referencia. En el ámbito médico se emplea para la **clasificación de
enfermedades raras**, donde es difícil disponer de grandes bases de datos. Asimismo,
resulta útil en la identificación de especies poco comunes en biología o en tareas de
clasificación en las que los datos disponibles son limitados.

### 5.9. Aprendizaje contrastivo y autosupervisado

El **aprendizaje autosupervisado** ocupa un lugar central en el ámbito de la visión por
computador, ya que permite entrenar modelos robustos sin la necesidad de disponer de
grandes volúmenes de datos etiquetados. Este enfoque se fundamenta en la generación
automática de señales de supervisión a partir de los propios datos, lo que reduce de
manera significativa la dependencia de procesos de anotación manual.

Dentro de este paradigma, una de las estrategias más influyentes es la que emplea la
**pérdida contrastiva**. Su propósito es aprender un espacio de representación en el que
las imágenes similares se ubiquen próximas entre sí, mientras que aquellas que difieren
se sitúen a mayor distancia. Para lograrlo, se construyen pares de datos que se utilizan
durante el proceso de optimización.

En escenarios sin etiquetas, los **pares positivos** se generan mediante
transformaciones aplicadas a una misma imagen, tales como rotaciones, cambios de escala,
recortes aleatorios o modificaciones en el color. De este modo, el modelo aprende que
diferentes vistas de un mismo contenido deben compartir una representación cercana. Por
el contrario, las **imágenes distintas** se consideran pares negativos, lo que obliga al
modelo a situar sus representaciones en regiones alejadas del espacio embebido.

<p align="center">
  <img src={require("../../../static/img/blogs/meta-learning/contrastive-learning-example.png").default}/>
  <br />
  <em>Ejemplo de aprendizaje contrastivo</em>
</p>

El **aprendizaje contrastivo** no se corresponde de manera directa con el
_meta-learning_, aunque contribuye a mejorar la calidad de las representaciones
aprendidas por los modelos de aprendizaje profundo. Su objetivo principal consiste en
agrupar en el espacio embebido las representaciones de datos similares y separar de
manera efectiva aquellas que pertenecen a clases distintas. En aplicaciones de visión
computacional, este principio se materializa mediante transformaciones de los datos que
permiten al modelo reconocer que ciertas variaciones no alteran la esencia de la
información original. Esto incrementa la comprensión semántica y refuerza la capacidad
del modelo para generalizar ante datos fuera de distribución.

El proceso de entrenamiento contrastivo constituye una metodología clave para obtener
representaciones efectivas y discriminativas a partir de datos no etiquetados. Dicho
proceso incluye varias etapas esenciales que, combinadas con técnicas de _fine-tuning_ y
_transfer learning_, optimizan el rendimiento de los modelos:

1. **Obtención de un conjunto de datos no etiquetados**: Se establece la base sobre la
   cual el modelo aprende de manera automática las características más relevantes.
2. **Generación de _embeddings_**: Se emplea un modelo preentrenado, como una red
   neuronal profunda ajustada previamente con grandes volúmenes de datos, por ejemplo
   ResNet en visión computacional. Este modelo transforma las entradas en
   representaciones de menor dimensionalidad que preservan la información esencial, como
   bordes, texturas o patrones semánticos.
3. **Optimización mediante _fine-tuning_**: Se ajustan los parámetros del modelo
   utilizando medidas de distancia entre _embeddings_, como la **distancia euclidiana**
   o la **similitud de coseno**. El objetivo es minimizar la **pérdida contrastiva**,
   que busca reducir la distancia entre representaciones de datos similares y aumentar
   la separación respecto de las representaciones de datos distintos. Entre las
   funciones de pérdida más empleadas destacan la **Triplet Loss**, que considera un
   ancla, un par positivo y un par negativo, y la **InfoNCE Loss**, ampliamente
   utilizada en entornos no supervisados. Un aspecto crítico consiste en evitar el
   colapso de las representaciones, situación en la que todos los _embeddings_ se
   vuelven indistinguibles. Para prevenirlo, se introducen márgenes o restricciones que
   aseguran diferencias suficientes entre clases.
4. **Iteración y ajuste manual**: Una vez entrenado el modelo, se revisan las muestras
   que generan mayores pérdidas. En casos necesarios, estas muestras se etiquetan
   manualmente, lo que refina el desempeño del modelo. Este ciclo iterativo favorece una
   mejora progresiva de las predicciones y reduce la necesidad de intervención humana
   con el tiempo.

Este proceso se apoya en los principios del **_transfer learning_**, que permiten
reutilizar el conocimiento aprendido en una tarea fuente $T_a$ para mejorar el desempeño
en una tarea objetivo $T_b$. En lugar de comenzar el entrenamiento desde cero, se
aprovechan las características generales aprendidas en la tarea inicial, lo que reduce
la carga computacional y acelera el entrenamiento. Incluso cuando las distribuciones de
datos en $T_a$ y $T_b$ difieren, dichas características resultan útiles al encapsular
información fundamental como bordes, texturas o relaciones semánticas. En determinados
casos, el ajuste fino no se limita a la capa final del modelo. Estudios como el
**Surgical Fine-Tuning** demuestran que refinar de manera selectiva las capas
intermedias puede incrementar significativamente la precisión.

El aprendizaje contrastivo se ha convertido en un componente esencial para mejorar la
calidad de las representaciones en múltiples dominios, especialmente en visión
computacional y procesamiento del lenguaje natural. A continuación, se describen algunas
de las funciones de pérdida más utilizadas.

<p align="center">
  <img src={require("../../../static/img/blogs/meta-learning/triplet-loss-example.png").default}/>
  <br />
  <em>Ejemplo de Triplet Loss</em>
</p>

La **Triplet Loss** se fundamenta en tres elementos principales:

- **Ancla ($X$)**: Una muestra de datos que sirve como referencia.
- **Par positivo ($X^+$)**: Una muestra similar al ancla, ya sea porque pertenece a la
  misma clase o porque corresponde a una transformación de esta.
- **Par negativo ($X^-$)**: Una muestra distinta al ancla, normalmente perteneciente a
  otra clase.

El objetivo de la Triplet Loss es minimizar la distancia entre el ancla y el par
positivo, a la vez que se maximiza la distancia entre el ancla y el par negativo.
Matemáticamente, se expresa como:

$$
L = \min_{\theta}\left(\max\left(0, \text{dist}(X, X^+) - \text{dist}(X, X^-) + \text{margen}\right)\right),
$$

donde $\text{dist}(\cdot, \cdot)$ corresponde a una función de distancia, como la
euclidiana o la similitud coseno, y el **margen** es un valor positivo que determina
cuánto mayor debe ser la separación entre el ancla y el par negativo respecto de la
distancia entre el ancla y el par positivo. Esta condición evita el colapso de las
representaciones, garantizando que los pares negativos permanezcan adecuadamente
diferenciados de los positivos.

La **Contrastive Loss** constituye otra función central en este campo, aunque se aplica
a pares de datos en lugar de tríos. Su formulación es:

$$
L = (1 - y) \frac{1}{2} \left( \text{dist}(X_1, X_2) \right)^2 + y \frac{1}{2} \left( \max(0, m - \text{dist}(X_1, X_2)) \right)^2,
$$

donde $y$ indica si $X_1$ y $X_2$ son similares ($y = 0$) o distintos ($y =
1$), $m$
representa el margen mínimo deseado entre ejemplos diferentes, y $\text{dist}(X_1, X_2)$
suele calcularse con la distancia euclidiana. En este caso, los ejemplos similares se
obligan a aproximarse en el espacio embebido, mientras que los diferentes deben
separarse al menos una distancia equivalente al margen $m$.

Finalmente, la **InfoNCE Loss** (_Noise-Contrastive Estimation_) se utiliza de manera
destacada en arquitecturas como **SimCLR**. Su objetivo es maximizar la similitud entre
un dato ancla y su par positivo, mientras se minimiza la similitud con los pares
negativos presentes en el mismo lote de datos. La fórmula se define como:

$$
L = -\frac{1}{N} \sum_{i=1}^N \log \frac{\exp(\text{sim}(z_i, z_i^+)/\tau)}{\sum_{j=1}^K \exp(\text{sim}(z_i, z_j^-)/\tau)},
$$

donde $z_i$ corresponde al _embedding_ del ancla, $z_i^+$ al _embedding_ del par
positivo, $z_j^-$ a los _embeddings_ de los pares negativos, $\text{sim}(\cdot,
\cdot)$ a
una medida de similitud como el producto escalar o la similitud coseno, y $\tau$ a una
constante de temperatura que ajusta la distribución de probabilidades. Este mecanismo
fomenta la generación de representaciones ricas y diferenciadas, adecuadas para tareas
de clasificación o transferencia.

Limitaciones del aprendizaje contrastivo

- **Dependencia de transformaciones adecuadas**: El rendimiento requiere el diseño
  cuidadoso de técnicas de aumento de datos, como _resizes_, _crops_, modificaciones de
  color, _CutMix_ o _MixUp_, que mejoren la robustez del modelo.
- **Necesidad de un gran número de épocas**: El desempeño depende tanto del tamaño de
  los lotes como del número de iteraciones necesarias para obtener suficientes pares
  negativos efectivos.

## 5.x. Convolucion en otro tipo de datos

Las convulsiones también pueden ser aplicadas en otros tipos de datos no solo en
imágenes llegue al final son muy buenas sobre todo relacionando información espacio
temporal ejemplo cambios que hace que existen en la cierta dimensión. .Por ejemplo,
podemos utilizar combo lución es en series temporales utilizando con volúmenes en una
única dimensión. Luego también podemos utilizar combo lución es en sonido porque al
final el sonido puede ser interpretado en componentes de frecuencia por ejemplo
utilizando especto gramas Mel utilizando la transformada discreta no la transformada de
furia entonces al final está representando el sonido como una imagen que evoluciona con
la frecuencia y el tiempo. Luego también podemos representar el vídeo con combo lución
es porque al final un vídeo es una representación sucesiva de imágenes durante el tiempo
entonces al final tendremos cuatro dimensiones que es el alto el ancho y el profundo de
la imagen y te lo el tiempo que es la duración del vídeo que es la cantidad de imágenes
por unidad de tiempo que se van a transmitir.

We also note that the dimensions in these examples can be roughly categorized as either
“spatial dimensions” (e.g., images) or “temporal dimensions” (e.g., audio resolution).
While images can be considered symmetric along their spatial axes (in many cases, an
image flipped along the width is another valid image), time is asymmetric: an audio
sample inverted on its temporal axis is in general invalid, and an inverted time series
represents a series evolving from the future towards its past.

También podemos interpretar frases o palabras para dividirlas en parte su secuencia de
unidades más pequeñas lo que se conoce hoy en día como toquen que esto al final es muy
conocido en los modelos de lenguaje como por ejemplo cha GPT, donde al final se coge una
palabra se divide por ejemplo tiene una oración frase texto lo que sea y se divide en
palabras por ejemplo o se puede también dividir en una secuencia de letras o en
múltiples palabras lo que sea esta división al final depende mucho del tipo de
arquitectura y de la decisión de las personas durante la creación de este tipo de
modelos, pero cada una de esas subdivisiones del texto se denomina toquen, que es como
la unidad mínima de representación de una palabra que luego se representa mediante un
sensor conocido como representación en bebida del modelo, entonces al final lo que
tienes es una representación de una palabra, que es un toquen o una representación que
conoce el modelo este es el funcionamiento básico de los modelos de lenguaje, lo que se
conoce como el procesamiento natural del lenguaje.

## 6. Modelos secuenciales

Muchos problemas en inteligencia artificial se caracterizan por involucrar datos
**secuenciales**, es decir, información organizada en un orden temporal o lógico.
Ejemplos destacados incluyen el reconocimiento de voz, la generación de música, el
análisis de sentimientos en texto, la interpretación de secuencias de ADN o la
traducción automática de idiomas. A diferencia de las imágenes, donde la información
espacial es clave, en las secuencias la dependencia entre elementos previos y
posteriores resulta esencial. Para abordar este tipo de problemas se emplean los modelos
secuenciales, cuya función es procesar los datos respetando su orden y capturando las
dependencias que existen a lo largo de la secuencia.

### 6.1. Representación de secuencias

En el procesamiento del lenguaje natural, las palabras deben transformarse en
representaciones que puedan ser interpretadas por un modelo. Este procedimiento se
denomina **tokenización**. Consiste en asignar a cada palabra un índice único dentro de
un diccionario y, posteriormente, transformarla en un vector que codifica su
información.

El proceso de tokenización contempla también el uso de **tokens especiales**. Entre
ellos se incluyen:

- Un token reservado para representar palabras desconocidas o no registradas en el
  vocabulario.
- Un token de **fin de secuencia**, empleado en tareas de generación de texto para
  señalar el cierre de la frase.

Las secuencias de entrada y salida no siempre poseen la misma longitud, lo que obliga a
gestionar variables específicas que reflejen estos tamaños y permitan al modelo
adaptarse a diferentes contextos. Este tratamiento cuidadoso de la representación
resulta esencial para que los modelos secuenciales puedan captar correctamente la
estructura y el significado del lenguaje, así como de cualquier otro tipo de datos
organizados en secuencia.

### 6.2. Redes neuronales recurrentes

Las **redes neuronales recurrentes (_Recurrent Neural Networks_, RNN)** constituyen una
extensión de las redes tradicionales diseñada para procesar datos secuenciales. Su
principal característica es la capacidad de **recordar información previa**, ya que
reutilizan la salida de un paso anterior como parte de la entrada en el siguiente. En
cada instante temporal, la red combina la entrada actual con el estado oculto anterior
para generar un nuevo estado oculto y producir una salida.

Este mecanismo permite que los parámetros se compartan a lo largo de la secuencia, lo
que no solo reduce el número de variables que deben aprenderse, sino que también
posibilita que la predicción en un momento dado tenga en cuenta el contexto acumulado de
los pasos previos. Gracias a esta propiedad, las RNN resultan adecuadas para tareas
donde el orden de los elementos y las dependencias temporales son fundamentales, como en
la traducción automática, el modelado del lenguaje o el análisis de series temporales.

No obstante, en la práctica las RNN se enfrentan a dos problemas significativos:

1. **Desvanecimiento de gradientes:** Durante el entrenamiento, los gradientes asociados
   a pasos muy lejanos en la secuencia tienden a volverse extremadamente pequeños, lo
   que dificulta que el modelo aprenda dependencias a largo plazo.
2. **Explosión de gradientes:** En algunos casos, los gradientes crecen de manera
   descontrolada, afectando la estabilidad del entrenamiento e impidiendo la
   convergencia del modelo.

Para superar estas limitaciones, se desarrollaron variantes más sofisticadas y robustas:

- **RNN bidireccionales:** Procesan la secuencia tanto en la dirección hacia adelante
  como hacia atrás, integrando simultáneamente información del pasado y del futuro, lo
  que resulta especialmente útil en tareas donde el contexto completo de la secuencia es
  relevante.
- **LSTM (_Long Short-Term Memory_):** Introducen una estructura de **celdas de
  memoria** acompañada de **puertas de control** que regulan qué información se
  conserva, cuál se descarta y cuál se utiliza en cada paso. Esta arquitectura permite
  capturar dependencias a largo plazo de manera eficaz, mitigando el problema del
  desvanecimiento de gradientes.
- **GRU (_Gated Recurrent Unit_):** Constituyen una variante simplificada de las LSTM.
  Mantienen el uso de puertas para controlar el flujo de información, pero con una
  estructura más ligera y eficiente en términos computacionales, alcanzando un
  rendimiento comparable en muchos contextos sin requerir tanta capacidad de cómputo.

### 6.3. Modelos de lenguaje y predicción de secuencias

Los **modelos de lenguaje** son sistemas diseñados para asignar probabilidades a
secuencias de palabras, permitiendo predecir la siguiente palabra en un texto dado el
contexto previo. Su entrenamiento se realiza a partir de grandes corpus, con el objetivo
de que el modelo aprenda no solo asociaciones entre palabras, sino también patrones
gramaticales, sintácticos y contextuales. Este aprendizaje habilita una amplia variedad
de aplicaciones, tales como asistentes virtuales, análisis de emociones en texto,
generación automática de contenido y descifrado o interpretación de secuencias
genéticas, donde la dependencia entre elementos consecutivos es crucial.

En el **procesamiento del lenguaje natural (_Natural Language Processing_, NLP)**, un
concepto central es el de los **_word embeddings_**, que son vectores densos que
representan palabras en un espacio continuo. En este espacio, las relaciones semánticas
entre palabras se reflejan en la geometría, donde palabras con significados similares se
ubican cerca, y las analogías pueden representarse mediante operaciones vectoriales.
Este enfoque supera al **_one-hot encoding_**, que asigna a cada palabra un vector
binario sin reflejar relaciones semánticas ni similitudes contextuales.

El aprendizaje de embeddings puede realizarse mediante diferentes técnicas:

- **Word2Vec:** Entrena modelos para predecir palabras a partir de su contexto en
  ventanas de texto, utilizando estrategias como _negative sampling_ para reforzar la
  relevancia de las relaciones semánticas aprendidas.
- **GloVe:** Combina información de coocurrencia global de palabras con factorización de
  matrices, integrando tanto la información local de contexto como la estadística global
  del corpus, lo que permite generar representaciones más consistentes y ricas
  semánticamente.

Estas representaciones pueden preentrenarse en grandes corpus y transferirse a tareas
específicas, lo que facilita la generalización y reduce la necesidad de datos
etiquetados en contextos concretos. Sin embargo, es importante señalar que los
embeddings también reflejan **sesgos presentes en los datos de entrenamiento**, los
cuales pueden afectar el comportamiento de los modelos. Estos sesgos pueden
identificarse y mitigarse mediante técnicas de neutralización o ajuste
post-entrenamiento, garantizando que las aplicaciones resultantes sean más justas y
confiables.

El desarrollo de embeddings sentó las bases para la **revolución de los Transformers**,
que introducen mecanismos de atención capaces de modelar dependencias a largo plazo de
manera eficiente, desplazando gradualmente a las arquitecturas recurrentes tradicionales
en muchas tareas de NLP y secuencias en general.

### 6.4. Mecanismo de atención

El mecanismo de atención constituye un componente fundamental en las arquitecturas
modernas de procesamiento de secuencias, ya que permite al modelo centrarse en las
partes más relevantes de la entrada según la tarea a realizar. Este mecanismo se
implementa a través de tres vectores:

- **Query (Q):** Representa lo que se está buscando o la información a destacar en un
  momento determinado.
- **Key (K):** Codifica la información disponible que puede ser relevante para la
  consulta.
- **Value (V):** Contiene el contenido asociado que se utilizará para construir la
  representación final.

El funcionamiento consiste en comparar la _Query_ con cada _Key_ para calcular un
conjunto de pesos que reflejan la relevancia relativa de cada elemento. A continuación,
estos pesos se aplican a los _Values_ correspondientes, generando representaciones
contextuales que integran la información más significativa para la tarea específica.
Este enfoque permite al modelo enfocarse dinámicamente en diferentes partes de la
secuencia, mejorando la capacidad de capturar dependencias a largo plazo y relaciones
complejas.

### 6.5. Transformers

Los Transformers, introducidos en el artículo _Attention is All You Need_,
revolucionaron el procesamiento de secuencias al eliminar la necesidad de recurrir a
RNN, permitiendo un procesamiento paralelo de los datos. La arquitectura se organiza en
dos componentes principales:

- **Encoder:** Procesa la secuencia de entrada y genera representaciones internas
  enriquecidas.
- **Decoder:** Utiliza estas representaciones para generar la secuencia de salida de
  manera autoregresiva o condicional según la tarea.

Cada bloque del Transformer combina mecanismos de **autoatención (_self attention_)** y
redes totalmente conectadas, lo que permite capturar relaciones complejas dentro de la
secuencia. Dado que los Transformers no procesan los elementos de manera secuencial, se
incorporan **_positional encodings_** para preservar información sobre el orden de los
elementos, garantizando que la red pueda distinguir entre distintas posiciones en la
secuencia.

El **m*ulti-head attention*** constituye una extensión clave del mecanismo de atención,
ya que permite al modelo observar relaciones desde múltiples perspectivas
simultáneamente. Esto enriquece la representación al capturar distintos tipos de
dependencias y patrones contextuales en paralelo.

Gracias a estas innovaciones, los Transformers escalan de manera eficiente a secuencias
largas, capturan dependencias complejas y se han extendido más allá del procesamiento de
lenguaje natural hacia dominios como la visión computacional, la bioinformática y el
aprendizaje por refuerzo, consolidándose como una de las arquitecturas más versátiles y
poderosas en el aprendizaje profundo actual.

La arquitectura Transformer, desarrollada por Google, constituye la base de los actuales
modelos de lenguaje de gran escala. Esta arquitectura se compone de dos bloques
principales: **encoder** y **decoder**.

- **Encoder**: Se encarga de transformar la entrada (por ejemplo, un texto) en una
  representación vectorial interna, comprimiendo o expandiendo su dimensionalidad para
  que el modelo pueda procesarla adecuadamente. Este bloque aplica mecanismos de
  atención como la _self-attention_ y la _cross-attention_ para identificar las partes
  más relevantes del texto.

- **Decoder**: Utiliza la representación generada por el encoder, junto con información
  adicional (por ejemplo, parte de una traducción), para generar un nuevo contenido.
  Esta estructura fue inicialmente concebida para tareas de traducción automática, donde
  el encoder recibe un texto en un idioma y el decoder produce la traducción en otro
  idioma.

Un componente clave del Transformer es el mecanismo de **autoatención
(self-attention)**, que permite evaluar la relevancia relativa de cada token dentro de
la secuencia, facilitando la comprensión contextual y permitiendo al modelo concentrarse
en los elementos más significativos.

Existen múltiples variantes de la arquitectura Transformer, entre las cuales destacan:

- **BERT (Bidirectional Encoder Representations from Transformers)**: Utiliza únicamente
  la parte del encoder. Su entrenamiento se basa en la enmascaración de palabras en un
  texto y la predicción de las mismas. Es especialmente eficaz en tareas de
  clasificación y análisis de sentimientos.

- **GPT (Generative Pre-trained Transformer)**: Utiliza solamente el decoder y está
  orientado a la generación de texto. Se entrena proporcionando secuencias incompletas,
  que el modelo debe completar. Este enfoque es útil en tareas como la traducción, el
  resumen de textos, la generación de código, y es eficaz en contextos de _few-shot_ y
  _zero-shot learning_.

- **Autoencoders enmascarados**: Se aplican principalmente en modelos visuales. Dividen
  una imagen en múltiples parches, ocultan algunos de ellos, y el objetivo es
  reconstruir los parches faltantes minimizando el error entre la imagen reconstruida y
  la original. Este mecanismo es similar al entrenamiento de BERT, pero aplicado al
  dominio visual.

Los modelos basados únicamente en la arquitectura del decoder, como GPT, son
considerados **modelos autoregresivos**. Esto significa que la salida generada en el
tiempo $$t$$ se utiliza como entrada en el tiempo $$t+1$$. Esta característica permite
mantener una mayor coherencia en la generación de texto, ya que cada paso de predicción
toma en cuenta lo generado previamente.

Al final, este tipo de arquitecturas basadas en Transformers, han destacado, sobre todo
en el mundo de la inteligencia artificial generativa, donde que sea generativa,
significa que es capaz de generar nuevos datos a partir de una distribución de
probabilidades que se aprende para la generación de nuevos datos, los motivos por los
que se pueden aparecer alucinaciones, pues es que no hay datos suficiente o son ruidosos
o están sucios que no hay contexto o no tienen restricciones o pautas.

## 7. Redes neuronales de grafos

Los grafos constituyen una estructura flexible y poderosa para representar información
compleja. Están formados por **nodos** (o vértices) y **aristas** (o conexiones) que
describen las relaciones existentes entre los elementos. Esta formalización permite
modelar fenómenos muy diversos, desde redes sociales y moléculas hasta sistemas de
telecomunicaciones, imágenes y texto, ofreciendo un marco unificado para el análisis de
datos estructurados y relacionales.

Las **Redes Neuronales de Grafos (_Graph Neural Networks_, GNN)** están diseñadas para
procesar directamente estas estructuras, extrayendo representaciones cada vez más ricas
de los nodos y del grafo en su conjunto, de manera análoga a cómo las redes
convolucionales procesan imágenes.

### 7.1. Representación de nodos y flujo de información

Cada nodo de un grafo se representa mediante un vector de características que codifica
su información individual. Durante sucesivas iteraciones, este vector se actualiza
combinando la información propia del nodo con la de sus vecinos, enriqueciendo así su
representación con el contexto del grafo.

Dado que los grafos no poseen un orden natural de nodos o conexiones, las operaciones de
agregación, como suma, promedio o máximo, deben ser conmutativas, garantizando que el
resultado no dependa del orden en que se procesen los vecinos. Con cada iteración, los
nodos adquieren representaciones que integran tanto sus propiedades individuales como
las de su entorno inmediato, permitiendo al modelo capturar dependencias locales y
globales de manera eficiente.

### 7.2. Representación de la estructura del grafo

La topología de un grafo puede representarse mediante distintas estructuras de datos:

- **Matriz de adyacencia:** Indica la presencia o ausencia de aristas entre nodos. Su
  implementación es sencilla, pero su eficiencia depende del orden de los nodos y puede
  resultar costosa en grafos de gran tamaño.
- **Listas de adyacencia:** Enumeran explícitamente las conexiones de cada nodo,
  ofreciendo mayor flexibilidad y eficiencia en el manejo de grafos dispersos.

En la práctica, estas representaciones se traducen en tensores que almacenan tanto las
características de los nodos como las relaciones que los unen, constituyendo la base
para las operaciones de propagación y actualización de las GNN.

### 7.3. Tareas sobre grafos

Las GNN permiten abordar problemas a diferentes niveles de granularidad:

- **Nivel de grafo:** Predicción de propiedades globales, como la clasificación de
  moléculas, la estimación de propiedades químicas o el análisis de sentimiento de un
  texto completo.
- **Nivel de nodo:** Identificación de roles o categorías de nodos, útil en segmentación
  de imágenes, detección de usuarios influyentes en redes sociales o categorización de
  palabras en grafos semánticos.
- **Nivel de arista:** Predicción de la existencia o el valor de conexiones, aplicable
  en sistemas de recomendación, detección de enlaces en grafos de conocimiento o
  relaciones biológicas entre moléculas.

### 7.4. Arquitecturas y variantes

Entre las arquitecturas más destacadas se encuentran:

- **Graph Convolutional Networks (GCN):** Cada nodo se actualiza a partir de la
  información agregada de sus vecinos, de manera análoga a las convoluciones en
  imágenes, permitiendo capturar patrones locales y globales en el grafo.
- **Graph Attention Networks (GAT):** Incorporan un mecanismo de atención que pondera la
  relevancia relativa de cada vecino, mejorando la capacidad de la red para diferenciar
  relaciones críticas de otras menos importantes.

El flujo de información en una GNN se basa en el **intercambio de mensajes** entre
nodos. En grafos muy grandes, este proceso puede resultar costoso, por lo que se
introducen mecanismos como el **nodo maestro (_masternode_)**, que centraliza la
propagación de información global sin necesidad de mantener conexiones exhaustivas entre
todos los nodos.

### 7.5. Aplicaciones

Los grafos ofrecen un marco unificado para abordar problemas en múltiples dominios:

- **Visión por computadora:** Una imagen puede representarse como un grafo de nodos,
  donde cada nodo corresponde a un píxel o superpíxel, conectados según proximidad o
  similitud.
- **Lenguaje natural:** Palabras de una oración o documento pueden organizarse como
  nodos en grafos secuenciales o semánticos, capturando relaciones contextuales y
  sintácticas.
- **Biología y química:** Moléculas, proteínas o redes metabólicas se describen
  naturalmente como grafos de átomos y enlaces, permitiendo predecir propiedades
  químicas o interacciones biológicas.

## 8. Otros paradigmas de aprendizaje, _Multi-task learning_ y _meta learning_

<p align="center">
  <img src={require("../../../static/img/blogs/meta-learning/multi-task.png").default} height="350"/>
  <br />
  <em>Diagrama de una arquitectura Multi-Task</em>
</p>

El **aprendizaje multitarea**, por su parte, persigue que un mismo modelo sea capaz de
resolver de manera simultánea múltiples problemas relacionados. La idea central es que
al compartir representaciones internas entre diferentes tareas, la red aprende
descriptores más generales y robustos que benefician a todas ellas. Un ejemplo
paradigmático se encuentra en la conducción autónoma, un único modelo puede segmentar
imágenes para identificar peatones, carreteras y vehículos, clasificar señales de
tráfico, y, al mismo tiempo, predecir trayectorias de otros automóviles o del propio
vehículo. En este escenario, el entrenamiento implica optimizar distintas funciones de
coste (una para la segmentación, otra para la clasificación y otra para la predicción de
movimientos) que se combinan en un proceso conjunto de descenso del gradiente.

El **_Multi-Task Learning_ (MTL)** se refiere a la capacidad de un modelo para realizar
múltiples tareas relacionadas de forma simultánea, utilizando una estructura compartida
que permite adaptar parámetros y salidas según el entorno. Este enfoque busca optimizar
recursos y mejorar la capacidad de generalización del modelo en escenarios dinámicos,
transfiriendo conocimiento entre tareas y minimizando la necesidad de ajustes
específicos.

<p align="center">
  <img src={require("../../../static/img/blogs/meta-learning/meta-learning.png").default} height="350"/>
  <br />
  <em>Diagrama sobre el uso de *Meta-Learning*</em>
</p>

El _Meta-Learning_ se enfoca en dotar a los modelos de la habilidad de identificar y
aprovechar patrones subyacentes en los datos, lo que les permite adaptarse rápidamente a
nuevos problemas o entornos con un mínimo de información.

Este enfoque es particularmente útil en escenarios con datos limitados o costosos de
obtener, como aquellos que involucran problemas de privacidad. Al mejorar la capacidad
de generalización, los modelos son más robustos y eficientes, optimizando recursos y
ofreciendo mejores resultados en tareas variadas.

Por tanto, resulta ideal para conjuntos de datos donde la proporción de datos
etiquetados es significativamente menor que la de datos no etiquetados. El uso del
paradigma de _Meta-Learning_ permite extraer patrones de datos etiquetados y aplicarlos
a datos no etiquetados, detectando variaciones y cambios en las distribuciones.

### 8.1. Parámetros en _Multi-Task Learning_

Al desarrollar modelos para _multi-task learning_, es crucial definir ciertos
parámetros:

- **Parámetros aprendibles, $\theta$**: Representa todos los parámetros que el modelo
  puede aprender.
- **Función, $f_\theta(y|x)$**: Describe el modelo parametrizado $\theta$, generando una
  distribución de probabilidad para $y$ dado $x$.
- **Tarea, $T_i$**: Se define como $T_i = \{p_i(x), p_i(y|x), L_i\}$, donde:
  - $p_i(x)$: Distribución de entrada $x$ específica de la tarea $T_i$.
  - $p_i(y|x)$: Distribución de probabilidad de la salida $y$ dado $x$ para la tarea
    $T_i$.
  - $L_i$: Función de pérdida asociada con la tarea $T_i$.

El objetivo general es minimizar la pérdida total del modelo a lo largo de todas las
tareas. Esto se puede formular como:

$$
\min_{\theta} \sum_{i=1}^{T} L_i(\theta, D_i),
$$

donde $D_i$ es el conjunto de datos de la tarea $i$. Además, para ajustar la relevancia
de cada tarea, se puede incluir un peso $w_i$

$$
\min_{\theta} \sum_{i=1}^{T} w_i \cdot L_i(\theta, D_i).
$$

### 8.2. Estrategias para _Multi-Tasking_

Las principales estrategias para abordar múltiples tareas incluyen:

1. **Modelos específicos para cada tarea**: Este enfoque no es escalable debido al alto
   costo computacional.
2. **Uso de _embeddings_ condicionales**: Técnicas que combinan información mediante:
   - **Concatenación o suma de _embeddings_**: Métodos equivalentes que combinan
     características.
   - **Sistemas _Multi-head_**: Un modelo único con múltiples salidas, eficiente para
     tareas variadas. Un ejemplo avanzado es el **_Multi-Gate Mixture of Experts_**.
   - **Condicionales multiplicativos**: Ajustan los _embeddings_ mediante factores
     multiplicativos según la tarea.

### 8.3. _Few-Shot Learning_

<p align="center">
  <img src={require("../../../static/img/blogs/meta-learning/few-shot-learning.png").default}/>
  <br />
  <em>Ejemplo de uso de *Few-Shot Learning* de *3-ways* y *3-shots*</em>
</p>

El **_Few-Shot Learning_ (FSL)** se centra en entrenar modelos que logren un alto
rendimiento con un número muy limitado de ejemplos etiquetados por clase. Este enfoque
es esencial en situaciones donde la recopilación de datos es complicada o costosa, como
en aplicaciones médicas o donde se requiere la privacidad de los datos.

El FSL se organiza en torno a dos conjuntos principales:

- **_Support Set_**: Es el conjunto de datos de entrenamiento específico para una tarea,
  compuesto por unas pocas muestras etiquetadas que el modelo utiliza para aprender a
  clasificar.
- **_Query Set_**: Es el conjunto de datos de prueba utilizado para evaluar el
  rendimiento del modelo en la misma tarea.

El aprendizaje few-shot se describe según dos parámetros importantes:

- **_K-shot Learning_**: Se refiere al número $K$ de ejemplos disponibles por clase en
  el _support set_. Por ejemplo, en un escenario de 1-shot learning, hay solo un ejemplo
  por clase, mientras que en 5-shot learning hay cinco ejemplos por clase.
- **_N-way Classification_**: Indica el número $N$ de clases diferentes en la tarea. Por
  ejemplo, un problema de 5-way classification implica clasificar entre cinco posibles
  categorías.

Existen dos tipos de modelos en este regimen:

- **Modelos no parametrizados**: Métodos como _k-Nearest Neighbors (k-NN)_ son simples y
  eficaces cuando se dispone de pocos datos. Sin embargo, su eficacia depende de tener
  _embeddings_ de alta calidad que representen bien las relaciones entre los datos.
- **Modelos parametrizados**: Redes neuronales profundas o métodos similares se utilizan
  para generar _embeddings_ que capturan las características relevantes de los datos en
  un espacio de menor dimensionalidad, reduciendo problemas como la maldición de la
  dimensionalidad. Estos modelos se entrenan para producir representaciones invariantes
  a transformaciones y adecuadas para métodos como _k-NN_.
