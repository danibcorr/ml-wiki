# Repaso de Aprendizaje Automático: Validación y Evaluación de Modelos

## 1. Validación cruzada (Cross Validation)

La **validación cruzada** es una técnica fundamental en aprendizaje automático para evaluar y comparar diferentes modelos. Su objetivo es estimar el rendimiento de un modelo en datos no vistos y seleccionar el algoritmo más adecuado. El proceso consiste en dividir el conjunto de datos en varios subconjuntos llamados **folds**, generalmente de tamaño similar, y entrenar el modelo de manera iterativa:

1. Se separa el conjunto de datos en (k) folds (por ejemplo, cinco o diez).
2. En cada iteración, se utiliza un fold como conjunto de prueba y los restantes como conjunto de entrenamiento.
3. El modelo se entrena con los folds de entrenamiento y se evalúa con el fold de prueba.
4. Este procedimiento se repite hasta que todos los folds hayan sido utilizados como conjunto de prueba una vez.
5. Finalmente, se promedian las métricas de evaluación obtenidas en cada iteración, como precisión, error, sensibilidad, etc.

El caso más común es el **k-Fold Cross Validation**, donde (k) suele ser cinco o diez, dependiendo del tamaño del conjunto de datos y de la complejidad del modelo.

## 2. Matriz de confusión

La **matriz de confusión** es una herramienta clave para evaluar la capacidad de un modelo de clasificación. Relaciona los valores predichos por el modelo con los valores reales, organizándolos en cuatro categorías:

- **True Positives (TP)**: instancias positivas correctamente clasificadas.
- **False Positives (FP)**: instancias negativas clasificadas incorrectamente como positivas.
- **True Negatives (TN)**: instancias negativas correctamente clasificadas.
- **False Negatives (FN)**: instancias positivas clasificadas incorrectamente como negativas.

La diagonal principal de la matriz (TP y TN) refleja la tasa de aciertos del modelo; valores más elevados indican un mejor desempeño.

## 3. Sensibilidad y especificidad

Dos métricas derivadas de la matriz de confusión permiten evaluar la capacidad del modelo para identificar correctamente las clases positivas y negativas:

- **Sensibilidad (Recall)**: mide la proporción de verdaderos positivos respecto al total de instancias realmente positivas. Se calcula como:

[
\text{Sensibilidad} = \frac{TP}{TP + FN}
]

Un valor alto indica que el modelo identifica correctamente la mayoría de las instancias positivas.

- **Especificidad**: mide la proporción de verdaderos negativos respecto al total de instancias realmente negativas. Se calcula como:

[
\text{Especificidad} = \frac{TN}{TN + FP}
]

Un valor alto indica que el modelo discrimina adecuadamente las instancias negativas. Ambas métricas se pueden expresar en porcentaje multiplicando por cien, lo que facilita la interpretación de los resultados.
