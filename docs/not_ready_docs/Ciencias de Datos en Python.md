# Ciencias de Datos en Python

# Ciencias de Datos en Python

Realizado por Daniel Bazo.

Última actualización: 03/05/2021

# Contenido

[TOC]

# Introducción

Una de las librerías de ciencias de datos más utilizadas en Python por su gran ecosistema es:

- **SciPy**

El curso se divide en 4 módulos principales:

1. Organizar los prerrequisitos y repasar conceptos básico de Python.
2. La herramienta Panda, fundamental cuando hacemos ciencia de datos con Python ya que ofrece estructuras de datos que nos permite pensar en los datos de una forma tabular.
3. Maneras avanzadas de consultar y manipular datos con Panda.
4. Análisis básico de estadísticas con NumPy y SciPy y la creación de un proyecto.

## **Fundamentos de la manipulación de datos en Python**

### 1. Introducción

La clase trata de la manipulación de datos mediante la librería Panda. Veremos como adquirir datos, como limpiar datos, como manipular y unir datos juntos etc. **Todas las tareas del curso** r**equieren una calificación del 80 % para pasar** ya que al final del curso se conseguirá hacer un proyecto de ciencia de datos aplicado al mundo real adecuado a ponerlo en un porfolio.

Todos los ejemplos, ejercicios y demás se realizarán en Jupyter.

### 1.1. Leyendo y escribiendo ficheros CSV

Vamos a importar el archivo de datos “*mpg.csv*”, el cual contiene datos de economía de combustible para 234 coches, teniendo en cuenta los parámetros descritos en el cuaderno Jupyter, veremos lo básico para iterar por un archivo .csv, para crear diccionario y recoger estadísticas.

```python
import csv # Librearía CSV para poder leer nuestro archivo CSV# Configurar en 2 la precisión para los decimales%precision 2with open('mpg.csv') as csvfile:    mpg = list(csv.DictReader(csvfile))mpg[:3] # Los 3 primeros diccionarios de la lista.
```

Con la función **.keys()** podemos obtener el nombre de las columnas del fichero csv sabiendo que es la primera fila la contiene dicha información, es decir:

```python
mpg[0].keys()
```

Una vez abierto el fichero, por ejemplo, vamos a determinar las millas por galón promedio en ciudad en todos los vehículos del archivo csv, teniendo en cuenta que los datos están guardados como string, debemos hacer un casting a float para poder operar con ellos:

```python
sum(float(d['cty']) for d in mpg) / len(mpg)
```

De manera similar, podemos calcular el promedio de ahorro de combustible en todos los coches:

```python
sum(float(d['hwy']) for d in mpg) / len(mpg)
```

Usa set() para devolver los valores únicos del número de cilindros que tienen los coches del fichero. Al agruparlo en un set, se guardan los elementos que no se repitan, es decir tendremos un set de datos con la cilindrada de todos los coches del fichero sin que se repita ninguno:

```python
cylinders = set(d['cyl'] for d in mpg)cylinders
```

### 1.2. Fechas y tiempos

Importamos las siguientes librerías:

```python
import datetime as dtimport time as tm# Para obtener la hora desde la época (enero 1 del 1970)
```

- **tm.time()** → Devuelve la hora actual en segundos desde la época. (1 de enero de 1970)
- **dt.datetime.fromtimestamp(tm.time())** → Convierte la marca de tiempo en fecha y hora.
- **variable.year()** → Obtener el año.
- **variable.month()** → Obtener el mes.
- **variable.day()** → Obtener el día.
- **variable.hour()** → Obtener hora.
- **variable.minute()** → Obtener minutos.
- **variable.second()** → Obtener segundos.
- **dt.timedelta(days = 100)** → Es una duración que expresa la diferencia entre dos fechas.
- **dt.date.today()** → Devuelve la fecha local actual.

### 1.3. Objetos avanzados en Python y función Map()

Aquí veremos un ejemplo de mapeo de la función **min** entre dos listas.

La función map es un ejemplo de programación funcional, el primer parámetro es la función que deseas ejecutar, el resto de parámetros son algo sobre lo que se puede iterar. Todos los argumentos iterables se desempaquetan de manera conjunta y son pasados a la función dada. Poniendo un ejemplo, imaginamos que tenemos dos tiendas con productos iguales pero con precios distintos y queremos encontrar el mínimo que tendríamos que pagar si compráramos el articulo mas barato entre las dos tiendas:

```python
store1 = [10.00, 11.00, 12.34, 2.34]store2 = [9.00, 11.10, 12.34, 2.01]cheapest = map(min, store1, store2)# Con map hacemos esa funcion en una sola lineacheapest
```

### 1.4. Funciones Lambda y List Comprehensions (Comprensión de lista)

Ejemplo de función Lambda:

```python
my_function = lambda a, b, c : a + bmy_function(1, 2, 3)
```

Recordar que la función lambda se conoce como función anónima y se debe a que es una funcionalidad que intenta utilizar una sola vez, no le damos un nombre.

Ahora para el caso de los list comprehensions partiremos de una función normal que convertiremos en una list comprehensions:

```python
# FUNCIÓN NORMALmy_list = []for number in range(0, 1000):    if number % 2 == 0:        my_list.append(number)my_list# MISMA FUNCIÓN HACIENDO USO DEL LIST COMPREHENSIONSmy_list = [number for number in range(0,1000) if number % 2 == 0]my_list
```

### 2. Fundamentos para manipulación de datos

Numpy es el paquete fundamental para la computación numérica con Python ya que proporciona formas de crear almacenamiento y manipulación de datos por lo que es fácil integrarlo con una amplia variedad de bases de datos y formatos de datos. A continuación veremos diferentes ejemplos utilizando la librería Numpy.

### 2.1. Librería numérica de Python, NumPy

NumPy es una biblioteca de Python, que añade soporte para grandes matrices y conjuntos multidimensionales, junto con una gran colección de funciones matemáticas de alto nivel para operar en estos conjuntos.

- Web: https://numpy.org

### 2.1.1. Creación de arrays con NumPy

Librerías a importar:

```python
import numpy as np # Así importamos la librería numpyimport math # Librería para operaciones matemáticas varias# CREACIÓN DE UN ARRAYa = np.array([1,2,3])print(a)
```

- **variable.ndim** → Imprime el número de la dimensión de la lista.

Si pasamos una lista de lista en un array de NumPy estaríamos creando una matriz:

```python
b = np.array([[1,2,3],[4,5,6]])
```

- **variable.shape** → Permite imprimir el orden de la matriz.
- **variable.dtype** → Permite verificar el tipo de elementos del array.
- **variable.dtype.name** → Permite quedarme con el nombre del tipo de elementos del array.

Si utilizamos números con coma flotante (con decimales), NumPy automáticamente convierte los enteros en flotantes ya que no existe perdida de precisión. NumPy intentará darle el mejor formato de tipo de datos posible para mantener los datos lo más homogéneos posible.

En algunos casos, nos puede interesar crear una matriz pero sin saber con qué valores rellenarla. NumPy ofrece funciones para este caso, rellenando matrices con 1, 0 e incluso con el valor que queramos:

```python
# Rellena matrices con 0d = np.zeros((2,3))print(d)# Rellena matrices con 1e = np.ones((2,3))print(e)# Valores del 1 al (2 - 1), con tamaño de paso de 0.1 de orden 5x2a = np.arange(1,2,0.1).reshape(5,2)print(a)# Array del tamaño de imagen con valores de 255np.full(imagen_array.shape,255)
```

- **np.random.rand(i,j)** → Permite generar un array con números aleatorios con el orden de i * j, si no se especifica j, este no será nada. Ejemplo:
    
    ```python
    a1 = np.random.rand(4)a2 = np.random.rand(4, 1)a3 = np.array([[1, 2, 3, 4]])a4 = np.arange(1, 4, 1)a5 = np.linspace(1 ,4, 4)a1.shape == a2.shape # Devuelve Falsea5.shape == a1.shape # Devuelve True
    ```
    
- **np.arange(a,b,x)** → Permite crear una secuencia de números en un array, siendo a el limite inicial, b el limite final y x el tamaño de paso (step-size).
- **np.linspace(a,b,x)** → Permite generar una secuencia de números con coma flotante, x números desde a hasta b, ambos inclusive.

### 2.1.2. Operaciones con arrays

Vamos a partir de 2 arrays, por ejemplo:

```python
a = np.array([10,20,30,40])b = np.array([1,2,3,4])
```

- Restar arrays:
    
    ```python
    c = a - bprint(c)
    ```
    
- Multiplicar cada índice del array por los índices de otro array:
    
    ```python
    d = a*bprint(d)
    ```
    

Vamos a poner un ejemplo práctico, suponiendo que tenemos un array de datos con temperaturas en Fahrenheit y las queremos pasar a Celsius, sabiendo que la formula es Celsius = (Fahrenheit - 32) * 5/9, tenemos:

```python
farenheit = np.array([0,-10,-5,-15,0])celcius = (farenheit - 32) * (5/9)print(celcius)
```

Ahora, aparece un concepto muy interesante, **el array booleano**, ya que podemos aplicar operadores en arrays devolviendo True si se cumple la condición por ejemplo en relación con los datos de las temperaturas anteriores, imaginando que queremos comprobar que la temperatura de los Celsius es mayor a -20:

```python
celcius > -20
```

Devolvería: array([ True, False, False, False, True]), lo que hace es iterar en el array y comprobar si los valores son mayores a -20.

Hemos dicho que NumPy admite la manipulación de matrices. Partiendo de 2 matrices, por ejemplo:

```python
A = np.array([[1,1],[0,1]])B = np.array([[2,0],[3,4]])
```

- Producto por número de la matriz (**NO ES EL PRODUCTO PUNTO**):
    
    ```python
    print(A*B)
    ```
    
- Producto punto de matrices:
    
    ```python
    print(A @ B)
    ```
    

Tener en cuenta que al manipular matrices de diferentes tipos, por ejemplo una matriz con elementos de tipo entero y otra con elementos de tipo float, el tipo de la matriz resultante corresponderá al mas general de los 2 tipos. Esto se lama **upcasting**.

Los arrays en NumPy tienen funciones muy interesantes como:

- **variable.sum()** → Devuelve la suma de todos los elementos del array.
- **variable.max()** → Devuelve el valor máximo del array.
- **variable.min()** → Devuelve el mínimo valor del array.
- **variable.mean()** → Devuelve la media.

Muchas veces pensamos en un array multidimensional como una matriz, con sus filas y columnas, pero también podemos pensar en estos arrays multidimensionales como listas ordenadas gigantes de números. EL numero de filas y columnas es una abstracción para un propósito particular. Así es como se almacenan las imágenes básica, ahora veremos un ejemplo de cómo funciona NumPy con imágenes.

Primero, vamos a utilizar la librería de imágenes de Python, PIL y una función para mostrar la imagen en Jupyter:

```python
from PIL import Imagefrom IPython.display import displayimagen = Image.open('nombre_imagen.terminacion')display(imagen) # Mostrar imagen en Jupyter
```

Podemos convertir una imagen en un array de NumPy de la siguiente forma:

- **Seguiremos los nombres de las variables del recuadro anterior.**

```python
imagen_array = np.array(imagen)print(f"Tamaño del array de la imagen: {imagen_array.shape}")imagen_array
```

Al imprimir el array de la imagen, al final aparecerá “*dtype =*”, en el cuaderno Jupyter se usó una imagen en blanco y negro, al mostrar el array obtuvimos “*dtype=uint8*”, el uint8 indica que son enteros sin signo, no hay números negativos, el 8 de uint8 significa 8 bits por byte, por lo que cada valor puede ser de hasta 2⁸ = 256 en tamaño, aunque realmente se queda en 255 porque partimos de 0. Tener en cuenta que para las imágenes en blanco y negro, el negro se almacena a 0 y el blanco se almacena en 255.

Una vez tengamos el array de la imagen podemos realizar cualquier operación con ella, en el cuaderno Jupyter se invierte los colores por ejemplo, pero, lo mejor es que podemos crear un renderizado de este array para componer la fotografía utilizando la función **fromarray()** de la librería de imágenes de Python (***from PIL import Image***):

```python
display(Image.fromarray(array_imagen))
```

### 2.1.3. Indexación, corte e iteración

Son importantes para la manipulación y el análisis de datos, ya que nos permite seleccionar datos en función de unas condiciones y copiar o actualizar los datos.

### 2.1.3.a. Indexado

Primero veremos la indexación de enteros. Un array unidimensional funciona de manera similar a una lista, por lo que para obtener un valor del array lo hacemos del mismo modo que para una lista, mediante los índices, por ejemplo:

```python
a = np.array([1,2,3])print(a[2]) # Se mostraría el 3, recordar que el índice inicial es 0
```

Para el caso de un array multidimensional (matriz) debemos coger el índice de la fila y de la columna, por ejemplo:

```python
a = np.array([[1,2,3],[4,5,6]])a[1][1] # Mostaría el 5
```

Incluso podríamos crear un array unidimensional que almacene varios datos de un array multidimensional, por ejemplo:

```python
valores = np.array([a[0,1],a[0,2],a[1,1]])
```

### 2.1.3.b. Indexación booleana

Permite seleccionar elementos arbitrarios en función de las condiciones, por ejemplo:

```python
a = np.array([[1,2,3],[4,5,6]])valores = np.array([a[0,1],a[0,2],a[1,1]])print(valores>=3)# Devuelve -> [False  True  True]
```

### 2.1.3.c. Corte

El corte es una forma de crear una submatriz basada en la matriz original, esta submatriz funciona de manera similar a una lista, por ejemplo:

```python
a = np.array([1,2,3,4,5,6,7,8,9])# Me muestra todos los elementos de a desde# el primer indice hasta el indice 3 (sin incluir)print(a[:3])
```

O me puedo quedar con partes específicas:

```python
a = np.array([1,2,3,4,5,6,7,8,9])b = a[2:4]print(b)# Devuelve -> [3 4]
```

Para las matrices ocurre lo mismo, teniendo en cuenta que el primer elemento sirve para seleccionar la fila y el segundo para seleccionar la columna, si solo aportamos un solo parámetro devolveremos una fila al completo con todas las columnas. Vamos a definir la siguiente matriz:

```python
a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])print(a)
```

Si me quiero quedar con los elementos hasta la segunda fila:

```python
a[:2]
```

Si quiero elegir las filas y dentro de las filas las columnas que quiero mostrar, haría lo siguiente:

```python
a[:2, 1:3]
```

Es importante darse cuenta de que un segmento de una matriz es una vista en los mismos datos, esto se llama pasar por referencia. Quiere decir que **la modificación de la submatriz modificará en consecuencia la matriz original**.

### 2.1.4. Probando NumPy con sets de datos

Para cargar un conjunto de datos en NumPy, podemos usar la función **genfromtxt()**. Podemos especificar el nombre del archivo de datos, el delimitador (el elemento que separa una columna de otra, por ejemplo: ‘Nombre’ ; ‘Apellidos’ ; ‘Edad’ → Aquí el delimitador sería el ;) que es opcional, pero usado a menudo y el numero de filas a omitir si tenemos una fila de encabezado (suele ser la primera fila que indicar los nombres de las columnas). Por eso, la función **genfromtxt()** tiene un parámetro llamado **d-type** para especificar tipos de datos para cada columna (este parámetro es opcional).

Si tenemos un ser de datos llamado “*winequality-red.csv*”, un fichero con información sobre vinos. Para cargar el conjunto haríamos lo siguiente:

```python
# EL delimitador en este archivo csv son ;vinos = np.genfromtxt("datasets/winequality-red.csv",                delimiter=";", skip_header=1)vinos
```

**¿Qué pasa si queremos varias columnas no consecutivas?**

Podemos colocar los índices de las columnas que queremos en una matriz y pasar esa matriz como el segundo argumento:

```python
# Me voy a quedar con todas las filas, con la columna 0, 2 y 4vinos[:, [0,2,4]]
```

Un ejemplo del procesado de datos para este fichero podría ser el querer averiguar la calidad media del vino tinto, teniendo en cuenta que la última columna indica la calidad del vino, podemos hacer una media de cada vino:

```python
vinos[:,-1].mean()
```

Otro ejemplo, vamos a analizar los datos para la admisión según puntuación para una universidad. Con este conjunto de datos, podemos hacer manipulación de datos y análisis básico para inferir que condiciones están asociadas con mayores posibilidades de admisión. Podemos especificar nombres de campos de datos usando **genfromtext()** mientras carga los datos del fichero csv llamado “*Admission_Predict.csv*”:

```python
admision_graduados = np.genfromtxt('datasets/Admission_Predict.csv',                                            dtype=None, delimiter=',', skip_header=1,                      names=('Serial No','GRE Score', 'TOEFL Score',                                                            'University Rating', 'SOP',                              'LOR','CGPA','Research', 'Chance of Admit'))admision_graduados
```

El resultado es un array unidimensional con 400 tuplas en ellas. Para recuperar una columna específica:

```python
admision_graduados['Nombre_Columna'][:5]# [:5] indica el número de tuplas a mostrar (de la tupla 0 a la 4)
```

Con este set de datos podríamos utilizar la indexación/máscara booleana para averiguar cuántos estudiantes han tenido experiencia en investigación:

- En este set de datos, si es 1 = True = Experiencia en investigación

```python
len(admision_graduados[admision_graduados['Research'] == 1])
```

### 2.2. Manipular texto con expresiones regulares

En este punto veremos **coincidencias de patrones en set de datos usando expresiones regulares**. Las expresiones regulares se escriben en un lenguaje de formato condensado, en general, se pueden pensar como un patrón que le da a un procesador de expresiones regulares algunos datos de origen, a continuación, el procesador analiza los datos de origen utilizando el patrón y devuelve fragmentos de textos para su posterior manipulación.

Hay 3 razones principales para hacerlo:

- Para comprobar si existe un patrón dentro de algunos datos de origen.
- Para obtener todos los casos de un patrón complejo de alguna fuente.
- Para limpiar sus datos de origen usando un patrón generalmente a través de la división de cadenas.

Las expresiones regulares no son triviales, pero son una técnica fundamental para la limpieza de datos en aplicaciones de ciencia de datos ya que nos permite manipular de forma rápida y eficiente los datos.

Primero hay que importar la librería de expresiones regulares:

```python
import re # Librería de expresiones regularestexto = "Hoy es un buen dia"if re.search("buen", texto): # El primer parámetro es el patrón    print("Perfecto!")else:    print("Lo siento :(")
```

- **variable.search()** → **C**omprueba si hay una coincidencia en cualquier lugar de la cadena y devuelve un valor booleano.
- **variable.match()** → Comprueba si hay una coincidencia del patrón al principio de la cadena.

Sabemos que poner separar un string con la función **split()**, ahora, si queremos contar cuantas veces hemos encontrado un patrón podemos usar la función **findall()**:

```python
text = """Amy works diligently. Amy gets good grades.                Our student Amy is succesful."""re.split("Amy", text)re.findall("Amy",text)
```

Existen patrones más complejos. Los caracteres:

- **^** → Significa inicio
- **$** → Significa final

Por ejemplo, para comprobar si un texto empieza con una palabra determinada:

```python
texto = "Hola muy buenas, es principio de año y me llamo Daniel"# Vamos a comprobar si el texto empieza con la palabra Holare.search("^Hola",texto)
```

Vemos que **re.search()** devuelve un nuevo objeto llamo **re.match**, este siempre tiene un valor booleano, True si encuentra algo.

### 2.2.1. Patrones y caracteres de clases

Lo mejor es verlo con un ejemplo y trabajar con él.

Vamos a crear una serie de calificaciones de estudiantes individuales durante un semestre en un solo curso a lo largo de todas sus tareas:

```python
import re# Lista con todas las calificaciones agrupadasgrades="ACAAAABCBCBAA"# Vamos a buscar las calificaciones Bre.findall("B",grades)# Vamos a buscar tanto los calificaciones A como las Bre.findall("[AB]",grades)# Vamos a buscar combinaciones de calificaciones# Se buscarán combinaciones del tipo AB ó ACre.findall("[A][B-C]",grades)# La busqueda anterior también se puede hacer usando operadores lógicosre.findall("AB|AC",grades)# Podemos utilizar la intercalación con el operador de conjunto# para negar nuestros resultados.# Por ejemplo si quisiera analizar solo las calificaciones que no eran A,# haríamos lo siguiente:re.findall("[^A]",grades)# OJO!!! TENER CUIDADO CON EXPRESIONES DEL TIPO:re.findall("^[^A]",grades)# Es una lista vacia, porque la expresion está diciendo# que queremos hacer coincidir cualquier valor al comienzo# de la cadena que no es una A y nuestra cadena comienza con una A,# por lo que no se encuentra ninguna coincidencia.
```

### 2.2.2. Cuantificadores

Los cuantificadores son el número de veces que se desea que el patrón se empareje para poder contar realmente como una coincidencia. El cuantificador mas básicos es: **e{m,n}**, donde e es la expresión que estamos haciendo coincidir, m el mínimo número de veces que desea que se coincida y n es el número máximo de veces que el elemento podría coincidir. Por ejemplo:

```python
import regrades="ACAAAABCBCBAA"re.findall("A{2,10}",grades) # 2 como el minimo y 10 como el maximo# Si quremos ir buscando rachas de 2re.findall("A{1,1}A{1,1}",grades)
```

Es importante tener en cuenta que la sintaxis del cuantificador de expresiones regulares no permite desviarse del patrón {m,n}. En particular, si tiene un espacio extra entre las llaves obtendrá un resultado vacío:

```python
re.findall("A{2, 2}",grades)# Resultado -> []
```

Si no incluimos un cuantificador, el valor predeterminado es 1 y si tiene un número entre llaves, se considera que es tanto el valor m como el valor n, por ejemplo:

```python
re.findall("A{2}",grades)
```

Vamos a ver un ejemplo mas complejo, usando algunos datos de Wikipedia:

```python
import rewith open("datasets/ferpa.txt","r") as file:    wiki=file.read()
```

Al mostrar el fichero podemos ver que todos los encabezados tienen la palabra “*[edit]*” seguidos de una nueva linea de carácter. Si queremos tener una lista con todos los encabezados de este artículo, podríamos hacerlo usando **re.findall**. Podemos estar interesados en buscar todos los caracteres tanto en minúscula como mayúscula de la A a la Z con una cantidad de entre 1-100 caracteres siempre y cuando sean seguido por la palabra “*[edit]*”:

```python
re.findall("[\\w]{1,100}\\[edit\\]",wiki)
```

***, es un meta carácter que indica un patrón especial de cualquier letra o dígito (existen mas meta caracteres). Uno de los más interesantes para acortar la sintaxis sería el asterisco*** **, indica la coincidencia entre 0 o mas veces, esto permite eliminar el limite superior de 100 caracteres seguidos:

```python
re.findall("[\\w]*\\[edit\\]",wiki)
```

Ahora podríamos crear una lista de títulos iterando a través de la función del recuadro anterior y aplicar otra expresión regular:

```python
# Hemos metido un espaciofor title in re.findall("[\\w ]*\\[edit\\]",wiki):print(re.split("[\\[]",title)[0]) # Tomamos el valor intermedio y lo dividimos en corchetes# así solo cogemos el título y no la palabra "*[edit]*"
```

### 2.2.3. Grupos

Hemos estado hablando de expresiones regulares como un único patrón que se coordina pero en realidad podemos hacer coincidir diferentes patrones al mismo tiempo, llamados **grupos** y luego referirnos a estos grupos mas tarde como queramos. Para agrupar patrones, usamos paréntesis:

```python
re.findall("([\\w ]*)(\\[edit\\])",wiki)# Podemos iterar por los elementos que cumplen este patrón# por lo que podemos obtener una lista de objetos usando# **finditer()**for item in re.finditer("([\\w ]*)(\\[edit\\])",wiki):    print(item.groups())
```

Vemos que el método **groups()** devuelve una tupla del grupo, por lo que podemos obtener un grupo individual usando grupos de número, donde el grupo(0) es toda la coincidencia:

```python
for item in re.finditer("([\\w ]*)(\\[edit\\])",wiki):    print(item.group(1))
```

Sería buena idea poner etiquetas o nombrar estos grupos. Al darles una etiqueta y mirar los resultados como un diccionario, sería bastante útil. Para esto utilizamos la sintaxis:

- (?P
    
    )
    

El paréntesis inicia el grupo, la ?P indica que es una extensión, y  es la clave del diccionario que queremos utilizar envuelta en <>, podemos hacer algo así:

```python
for item in re.finditer("(?P<title>[\\w ]*)(?P<edit_link>\\[edit\\])",wiki): # (?P<edit_link>\\[edit\\]), le pone la etiqueta de "edit_link" a lo que vaya delante de [edit]    # podemos obtener el diccionario de vuelta usando groupdict()    print(item.groupdict()['title'])
```

### 2.2.4. Look-ahead and Look-behind

En este caso el patrón que se hace coincidir con el motor de expresiones regulares es para el texto de antes ó después del texto que estamos tratando de aislar. En el ejemplo del punto 2.2.3 queríamos aislar el texto que viene antes de “*[edit]*” pero en realidad no nos importa el texto de “*[edit]*”.

Si queremos usarlo para que coincida el patrón, pero no queremos capturarlos, podríamos ponerlos en un grupo y usar **look-ahead** en vez de la sintaxis ?=, por ejemplo:

```python
for item in re.finditer("(?P<title>[\\w ]+)(?=\\[edit\\])",wiki):        # El primer llamado "title", el segundo grupo es un grupo desechable        # que está mirando hacia adelante (look-ahead)    # Esta expresión regular pretende que coincida con 2 grupos,        # el primero se nombrará "title",    # tendremos espacios en blanco o carácteres de palabra normales,        # el segundo será la edición de carácteres    # pero en realidad no queremos que este elemento        # esté en nuestros objetos de coincidencia en la salida    print(item)
```

Incluso podemos crear patrones en múltiples líneas, por ejemplo:

```python
patron = """(?P<title>.*)        #Nombre universidad(–\\ located\\ in\\ )   #indicación localización(?P<city>\\w*)        #ciudad en la que se encuentra la universidad(,\\ )                #separador(?P<state>\\w*)       #estado de la ciudad"""# Como es en multilinea usamos **re.VERBOSE**for item in re.finditer(patron,wiki,re.VERBOSE):    print(item.groupdict())
```

### 2.2.5. Documentación

- Operaciones expresiones regulares: https://docs.python.org/3/library/re.html
- Para hacer debugs de patrones y comprobar que son correctos podemos usar webs como: https://regex101.com/

## **Procesado básico de datos**

### 1. Introducción a Pandas y datos en serie

En esta semana, vamos a ver cómo Python se puede usar para manipular, limpiar y consultar datos mirando el kit de herramientas que implementa la librería Pandas.

Tener en cuenta otro tipo de herramientas como **Stack Overflow**, muy empleado en el desarrollo de software para publicar preguntas sobre programación o en este caso, para preguntar sobre diferentes herramientas relacionadas con Pandas, es el recurso número 1 .

Otro recurso muy útil es el libro: “*Python for Data Analysis*” de Wes McKinney.

También existe un blog: https://www.planetpython.org/ , un blog actualizado con las últimas noticiones relacionada con la ciencias de datos en Python.

### 1.1. Estructura de datos en serie

Aquí veremos la estructura de la serie de Pandas. Deberemos estar familiarizados con cómo almacenar y manipular datos de unidimensional en una serie. La serie, es una de las estructura centrales en Pandas, es una especie de cruce entre una lista y un diccionario.

Todos los objetos se almacenan en un orden y hay etiquetas con las que podemos recuperar estos objetos. Una forma de visualizar esto es con 2 columnas de datos, la primera columna es el índice especial (lo que sería las claves (keys) en un diccionario) y la segunda columna serían los datos reales (los valores (values) del diccionario)