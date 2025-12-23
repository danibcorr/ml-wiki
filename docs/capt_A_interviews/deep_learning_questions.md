---
sidebar_position: 3
authors:
  - name: Daniel Bazo Correa
description: Preguntas relacionadas con Deep Learning
title: Preguntas Deep Learning
toc_max_heading_level: 3
---

## Bibliografía

- [Deep Learning Interview Prep Course](https://youtu.be/BAregq0sdyY?si=xsq-901fJqlug4WY)

## Conceptos Básicos

<details>
<summary>
¿Qué es el aprendizaje profundo (deep learning)?
</summary>
 El aprendizaje profundo es una rama de la inteligencia artificial que se basa, sobre todo en utilizar algoritmos basados en redes neuronales. Y se caracteriza sobretodo por utilizar mucho de este tipo de arquitecturas, es decir, en crear un Stack una agrupación de múltiples neuronas con la idea de poder identificar patrones que son mucho más complejos sobre todo en datos que no presentaba una relación simple. Como ocurre con datos tabuladores esto están sobre todo muy pensado para datos, no estructurados, compensa, imágenes, sonido, etc.
</details>

<details>
<summary>
¿En qué se diferencia el aprendizaje profundo de los modelos de aprendizaje automático (machine learning) tradicionales?
</summary>
Se diferencian sobre todo en el tipo de algoritmos que utilizan al final hemos estado son parte del mismo grupo es el su conjunto al final son sus conjuntos en cada serie en la aprendizaje automático utiliza algoritmos más clásicos como por ejemplo tenemos sistema de regresión lineales sistemas regresiones, logísticos y no utilizan el algoritmo principal de la aprendizaje profundo que sales a las redes neuronales.
</details>

<details>
<summary>
¿Qué es una red neuronal?
</summary>
Bueno primero habría que definir lo que es una neurona y al final una neurona o percepción es lo que tiene o tiene el funcionamiento de una regresión lineal con la función de activación que permite dar una no linealidad de la salida al final lo que hacen es relacionar los datos de entrada al concierto beso y un sesgo para ver cómo se relaciona todos los datos de entrada y luego pasé por una función de activación que es la no linealidad entonces cuando combinamos múltiples neuronas es decir una neurona empieza transmiten información a la hora las posteriores existe una conexión directa entre monte personas entre mis manos en las diferentes capas por ejemplo pues esto conforma una red normal que es una agrupación o como lo merecen de neuronas.
</details>

<details>
<summary>
¿Qué es una función de activación en las redes neuronales?
</summary>
 Al final una función de activación, lo que te permites dar una no linealidad de la salida de la red de Ronaldo, permite hacerte la salida está en un rango comprendido de valores. Esto dependerá también mucho del tipo de capa la que nos encontremos no es lo mismo las capas intermedias que se encuentra entre la carga dentro de la capa de salida y las funciones de activación que se requieren para las capas de salida que sobre todo están relacionadas con el tipo de problema que estamos afrontando por ejemplo un sistema de un problema de clasificación tendrá una función de activación diferente a un sistema o problema de regresión, porque los resultados estarán comprendidos en rango de valores diferentes.
</details>

<details>
<summary>
Menciona algunas funciones de activación populares y descríbelas.
</summary>
algunas funciones de activación, pues son Relu cuyos valores permanecen a cero cuando los valores están por debajo de cero y es lineal con valores por encima de cero y luego tenemos diferentes variaciones a relu,. Donde añaden como un ruido océano, al principio, o añaden una pendiente con una cierta parámetro alfa, y esto se debe sobre todo a que este tipo de activaciones eso la generar una gran espasticidad, que consiste en tener activaciones prácticamente a cero lo que puede causar neuronas muertas dentro de una red neuronal sin embargo, esta sido una de las funciones de activación más estilizado y que permitía el aprendizaje en grandes modelos. No existen otras funciones como la tangente hiperbólica que pone el rango de valores entre -1 y uno la función sigmoidea entre rango de valores comprendidos entre cero y uno que por ejemplo se utiliza en sistemas de clasificación binaria, por ejemplo. Tenemos softmax queda a la salida un reparto de distribuciones representadas como probabilidades.
</details>

<details>
<summary>
¿Qué sucede si no utilizas ninguna función de activación en una red neuronal?
</summary>
Pues al final lo que aquí estás limitando la capacidad de poder relacionar patrones entre los datos de entrada y tienden a parecerse más a algoritmos clásicos. Al final si se lo puedes tener si no tienes funciones de activación es la neurona tiene un comportamiento lineal entonces solo es capaz de relacionar características de forma lineal, ya se ha demostrado algunas personas que utilizar redes normales que tienen que no tienen funciones de activación por ejemplo los auto codificadores también conocidas como tan covers pues tienen un comportamiento similar a por ejemplo utilizar PCA algo similar
</details>

<details>
<summary>
Describe cómo funciona el entrenamiento de las redes neuronales básicas.
</summary>
Pues se basa principalmente en el  descenso del gradiente, donde lo que se hace es calcular la derivada de una función con respecto a los pesos y el sexo que son los parámetros aprendibles de la neurona.
</details>

<details>
<summary>
¿Qué es el descenso de gradiente (gradient descent)?
</summary>
Es el algoritmo que se utiliza en el proceso de entrenamiento de los modelos, y que consiste en calcular la derivada parciales de los parámetros aprendido a la neurona y actualizarlos con el fin de alcanzar un mínimo global o local, que permita llegar a un punto de optimización en el que poder conseguir o lograr un objetivo es objetivo se suele llamar función de costes de pérdida y lo que se pretende es minimizar el error.
</details>

## Optimización y Problemas de Gradiente

<details>
<summary>
¿Cuál es el papel del optimizador en el aprendizaje profundo?
</summary>
El Optimizador se encarga de agilizar el proceso del descenso del gradiente, con la idea de llegar antes a conseguir esos mínimos local o global.
</details>

<details>
<summary>
¿Qué es el back propagation y por qué es importante en el aprendizaje profundo?
</summary>
La propagación hace tras consiste en una vez que se han calculado las múltiples derivadas parciales a lo largo de la red, pues consiste en actualizar esos parámetros desde el final de la del resultado de la red hasta el principio esto es porque es a la salida cuando se comprueba, se mide el error que procesos y parámetros esos parámetros que diga de la red que sea no tenido mediante el el ascensor, gradiente, entonces hacer preparaciones detrás para poder actualizar todos esos parámetros e ir mejorando el error de salida.
</details>

<details>
<summary>
¿En qué se diferencia la retropropagación del descenso de gradiente?
</summary>
Al final la propagación de atrás, lo único que hace es propagar la actualización de los parámetros, dado el error de salida por la métrica que se quiera utilizar y el descenso lo agradecen si consiste en calcular esos parámetros.
</details>

<details>
<summary>
Describe qué es el problema del gradiente que se desvanece (vanishing gradient) y su impacto en las redes neuronales.
</summary>

</details>

<details>
<summary>
¿Cuál es la conexión entre varias funciones de activación y el problema del gradiente que se desvanece?
</summary>
Respuesta.
</details>

<details>
<summary>
Hay una neurona en la capa oculta que siempre resulta en un gran error en la retropropagación, ¿cuál podría ser la razón?
</summary>
Respuesta.
</details>

<details>
<summary>
¿Qué entiendes por un gráfico computacional?
</summary>
Respuesta.
</details>

<details>
<summary>
¿Qué es la entropía cruzada (cross entropy) y por qué se prefiere como función de costo para problemas de clasificación multiclase?
</summary>
Respuesta.
</details>

<details>
<summary>
¿Qué tipo de función de pérdida podemos aplicar cuando se trata de clasificación multiclase?
</summary>
Respuesta.
</details>

<details>
<summary>
¿Qué es SGD (descenso de gradiente estocástico) y por qué se utiliza en el entrenamiento de redes neuronales?
</summary>
Respuesta.
</details>

## Variantes de Optimización y Tamaño de Lote

<details>
<summary>
¿Por qué el descenso de gradiente estocástico (SGD) oscila hacia el mínimo local?
</summary>
Respuesta.
</details>

<details>
<summary>
¿En qué se diferencia el GD del SGD?
</summary>
Respuesta.
</details>

<details>
<summary>
¿Cómo podemos usar métodos de optimización como GD de una manera más mejorada? ¿Cuál es el papel del término de momento (momentum)?
</summary>
Respuesta.
</details>

<details>
<summary>
Compara el descenso de gradiente por lotes (batch gradient descent) con el descenso de gradiente por mini-lotes (mini batch gradient descent) y con el descenso de gradiente estocástico (stochastic gradient descent).
</summary>
Respuesta.
</details>

<details>
<summary>
¿Cómo decidir el tamaño del lote (batch size) en el aprendizaje profundo, considerando tamaños demasiado pequeños y demasiado grandes?
</summary>
Respuesta.
</details>

<details>
<summary>
¿Cómo afecta el tamaño del lote al rendimiento de un modelo de aprendizaje profundo?
</summary>
Respuesta.
</details>

<details>
<summary>
¿Qué es la matriz Hessiana (Haitian) y cómo se puede usar para un entrenamiento más rápido? ¿Cuáles son sus desventajas?
</summary>
Respuesta.
</details>

<details>
<summary>
Discute el concepto de tasa de aprendizaje adaptativa (adaptive learning rate). Describe los métodos de aprendizaje adaptativo.
</summary>
Respuesta.
</details>

<details>
<summary>
¿Qué es RMS Prop y cómo funciona?
</summary>
Respuesta.
</details>

<details>
<summary>
¿Qué es Adam y por qué se usa la mayoría de las veces en redes neuronales?
</summary>
Respuesta.
</details>

## Normalización, Conexiones y Problemas de Gradiente

<details>
<summary>
¿Qué es Adam W y por qué se prefiere sobre Adam?
</summary>
Respuesta.
</details>

<details>
<summary>
¿Qué es la normalización por lotes (batch normalization) y por qué se utiliza en redes neuronales?
</summary>
Respuesta.
</details>

<details>
<summary>
¿Qué es la normalización por capa (layer normalization) y por qué se utiliza en redes neuronales?
</summary>
Respuesta.
</details>

<details>
<summary>
¿Qué son las conexiones residuales (residual connections) y cuál es su función en las redes neuronales?
</summary>
Respuesta.
</details>

<details>
<summary>
¿Qué es el recorte de gradiente (gradient clipping) y su impacto en la red neuronal?
</summary>
Respuesta.
</details>

<details>
<summary>
¿Cuáles son las diferentes formas de resolver el problema del gradiente que se desvanece?
</summary>
Respuesta.
</details>

<details>
<summary>
¿Cuáles son las formas de resolver los gradientes explosivos (exploding gradients)?
</summary>
Respuesta.
</details>

<details>
<summary>
¿Qué sucede si la red neuronal sufre de sobreajuste (overfitting) y cómo se relaciona con los pesos en la red neuronal?
</summary>
Respuesta.
</details>

<details>
<summary>
¿Qué es el abandono (dropout) y cómo funciona?
</summary>
Respuesta.
</details>

<details>
<summary>
¿Cómo previene el abandono el sobreajuste en las redes neuronales?
</summary>
Respuesta.
</details>

## Regularización y Modelos Generativos

<details>
<summary>
¿Es el abandono como Random Forest?
</summary>
Respuesta.
</details>

<details>
<summary>
¿Cuál es el impacto del abandono en el entrenamiento frente a las pruebas (testing)?
</summary>
Respuesta.
</details>

<details>
<summary>
¿Qué son las regularizaciones L2 o L1 y cómo previenen el sobreajuste en las redes neuronales?
</summary>
Respuesta.
</details>

<details>
<summary>
¿Cuál es la diferencia entre los enfoques de regularización L2 y L1?
</summary>
Respuesta.
</details>

<details>
<summary>
¿Cómo impactan las regularizaciones L1 y L2 a los pesos en una red neuronal en lo que respecta a la comparación del impacto en penalizaciones de pesos grandes versus pequeños?
</summary>
Respuesta.
</details>

<details>
<summary>
¿Qué es la maldición de la dimensionalidad (curse of dimensionality) en el aprendizaje automático o en la IA?
</summary>
Respuesta.
</details>

<details>
<summary>
¿Cómo abordan los modelos de aprendizaje profundo la maldición de la dimensionalidad?
</summary>
Respuesta.
</details>

<details>
<summary>
¿Qué son los modelos generativos? Da ejemplos.
</summary>
Respuesta.
</details>
