# Front end development

Utilizamos HTML para la colocación de los objetos dentro de la web, colocar títulos,
botones, etc. Mientras que utilizamos css para darle formato y estilo al contenido de
HTML. Utilizamos java Script por ejemplo para procesar entradas del usuario, realizar
conexiones de APIS, o similares. Lo que viene siendo la interación del usuario.

La parte del backend se encarga de la parte de interacción con bases de datos, APIs,
sevidores webs.

Full staks debería tener conocimiento en: planning, arquitectura, diseño, desarrollo,
despliegue y manteniemiento.

Existe el modelo fundamental que es el modelo cliente-servidor, donde tenemos múltiples
ordenadores que son los clientes, que se comunican con internet, y el servidor que es el
centro de datos, aqui existe una transmision de datos, donde los clientes realizan
peticiones a través de internet, estas son procesadas por el servidor, mandando una
respuesta que llega al usuario.

Existe el web requests que son peticiones que manda el usuario atraves de Internet hasta
el servidor, esta se procesa y posteriormente se da una respuesta que se devuelve al
usuario.

Las paginas se procesan de manera secuencial y en orden, cada linea de HTML se
interpreta y el navegador lo renderiza (Page Rendering)

El protocolo usado entre el cliente y servidor es quest y responde.

## Hosting

Es el servidr donde alojamos nuestro servicio web

Tenemos el hosting compartido, donde los recursos de un servidor se reaparten entre
multiples clietnes (no usuarios) entonces otras web consumen de los mismo recursos. El
coste es menor, por lo que para paginas pequeñas, poco demandatens

Tenemos el hosting compartido, donde los recursos de un servidor se reparten entre
multiples clientes, que no son los usuarios finales en sí. Entonces otras webs consumen
de los mismos recursos. El coste es menor, por lo que para páginas pequeñas y pocos
demandantes, para desarrollo, suelen estar bien. El coste es menor, por lo que para
páginas pequeñas y poco demandantes, eso está bien. Luego están los hostings dedicados,
donde los recursos son exclusivos. Esto al final tiene la ventaja de la flexibilidad,
pero generalmente tiene un mayor coste. Existen códigos para reconocer los estados de
HTTP, como 404, Not Found, No Encontrado o similares. Y se pueden agrupar por diferentes
categorías. La información, que va desde 100 hasta 199. El grupo de correcto, que va
desde 200 hasta 299. La redirección, que va de 300 a 399. Cliente, que va de 400
hasta 499. Servidor, que va desde 500 hasta 599. Utilizamos HTTPS para comunicaciones
seguras mediante encriptación, antes de mandar el contenido. Tenemos webpages, que son
solo una página. Websites, que son un conjunto de webpages, lo que suelen ser mayor a
una página. Y las aplicaciones webs, como pueden ser Spotify, donde la idea es que sea
más dinámico e interactivo. Habría que diferenciar lo que son las librerías de los
frameworks, donde las librerías son funcionalidades reducidas que hacen cosas
específicas, de manera reducida. Un framework agrupa un conjunto de librerías, que es
algo más grande, que puede alterar la forma en la que trabajamos. Por ejemplo, podemos
instalar una librería para realizar validación de datos, utilizando el gestor de
paquetes de Node, que se conoce como npm. Un framework sería algo como, endokusaurus,
typescript, astro... Luego tenemos las APIs, que son Application Programming Interface,
que actúa como un middleware entre lo que viene siendo el usuario y el servicio en sí. Y
es un servicio, aplicación o interfaz, que ofrece funcionalidades avanzadas sobre lo que
estamos haciendo en el taxi. Puede ser un browser API, que entiende funcionalidades del
navegador, extiende las funcionalidades del navegador, por ejemplo, Fetch, Canvas,
History API. Y luego tenemos REST API, que es un conjunto de principios para construir
APIs eficientes. Y por último, los Sensor Based APIs, que son APIs para sensores
físicos, para dispositivos IoT. Luego tenemos que HTML es Hypertext Markup Language.
Está compuesto por tags y elementos. Hypertext significa texto que contiene enlaces a
otros textos. HTML tiene una especificación que está mantenida por la W3C. Ahí se
especifican los cambios y peculiaridades de cada versión de HTML.

## Tags

- <p></p>: parrafos.
- <br>: no necesita cerra </br>, es un salto de linea.
- <strong></strong>: importante
- <b></b>: negrita
- <em></em>: enfatizar
- <i></i>: italica
- lista: <ul><li>Elemento 1</li><li>Elemento 2</li></ul>
- lista ordenada: <ol><li>Elemento 1</li><li>Elemento 2</li></ol>
- <div></div>: define divisiones en el contexto HTML sin añadir estilo. Por ejemplo, <div><h1>Hola</h1></div>

- Linkado: <a href="pagina.html">Hola</a>
- Imagenes: <img src="image.png" height="300" width="300"/>

Input Tags, para elegir formatos de entrada y similares en HTML, como formularios,
campos a rellenar, etc. Luego tenemos los DOM, que es Document Object Module, que
permite hacer una interpretación de la página HTML para mandar una query, por ejemplo, a
Javascript y actualizar la página web en sí. Nosotros tenemos la página HTML que genera
un DOM, que es el Document Object Model, y ese genera o recarga o actualiza la página
HTML. El DOM es una estructura en árbol de los objetos de la web. Es como hacer una
compilación de los métodos o tags de HTML, donde se crea una estructura en árbol. Por
ejemplo, primero tendríamos HTML, y tendríamos luego de ese nodo principal que parten
dos nodos secundarios, el Head y el Body. El Head tendría el título, y por ejemplo,
luego debajo del título, el texto. Y en el caso del Body, pues podremos tener diferentes
tips. Esto sería un ejemplo de DOM con los elementos de una página simple de HTML, que
dependerá de los elementos que tenemos definidos, pero sería estático. Gracias a
Javascript podemos alterar el DOM para hacer que nuestra web sea dinámica, y es lo que
hacen frameworks como React. Es buena práctica tener siempre en consideración la
accesibilidad en el código. Para ello podemos acudir a la WAI, que es el Web
Accessibility Initiative, y el ARIA, que es Accessible Rich Internet Application.

# CSS

h1 { propiedad: valor; }

h1 es el selector del elemento a modificar, dentro de las llaves del selecto, son todas
las propiedades y valores, se conoce como bloque de declaracion. Podemos instalar Live
Preview para visualizar las webs en VSCode, que es una extensión. Luego definimos el CSS
en el head de HTML. Utilizamos Link-rel-style-sheet y hacemos href al CSS.

<head>
    <link rel="stylesheet" href="style.css">
</head>

Podemos definir IDs para estilos en CSS, para así no tener que aplicar el mismo estilo a
todos los elementos de la web. Por ejemplo, podemos definir el CSS para h1 definiendo un
color, por ejemplo el morado, que luego hace referencia a ese identificador de la clase
dentro de HTML para ese tag.

css:

h1 { color: purple; }

o css:

#header1 { color: blue; }

 <h1 id="header1"></h1>

Selector elementos:

<p>hola</p>

p { color: blue; }

ID Selector:

<p id="hola">hola</p>

#hola { color: blue; }

Class selector:

<p class="clase">hola</p>

.clase { color:blue; }

Elemento con class selector:

<p class="clase">hola</p>

p.clase { color: blue; }

Selector descendente:

<div id="blog">
    <p>hola</p>
</div>

#blog p { color: blue; }

Pseudo-class son estados posibles de un objeto, por ejemplo, cuando estamos ecnima de un
boton, cuando hacemos click, etc.

a:hover { color: blue; }

Si tenemos elementos dentro de otros (actuando el elemento superior como padre y el otro
como hijo), <h2><span></span></h2>, y queremos dar formato al hijo dentro del padre:

h2 > span { color: blue; }

Utilizamos el modelo box para indicar la ubicación, que son los rectángulos, de los
objetos dentro de la web. Cada contenido, es decir, cada box, está compuesto por el
contenido en sí, luego tiene un padding o relleno, tiene un border y un margen. El
contenido es la información del objeto en sí. El tamaño del contenido es alto y ancho,
pues repercute en el tamaño del box, donde podemos utilizar el width, el minimum width,
el max width, el height, min height, height y max height, son la altura y la anchura,
respectivamente, y maximum minimum. El relleno viene marcado por el padding box, width y
height, y el relleno interno, que afecta al contenido, viene por el padding top, padding
bottom, padding left y padding right. Entonces el padding box width se calcula como el
content width, más el padding left, más el padding right. Y el padding box height se
calcula como el content height, más padding top, más padding bottom. Para los bordes,
tenemos diferentes tipos como solid, dashed, que es alineado, dotted, que es con puntos,
o dabbled, que es doble. Luego tenemos diferentes tipos de elementos HTML, donde los
elementos HTML pueden separarse en dos puntos principales. Tenemos elementos de bloque,
que ocupan todo el width de la página y tienen una línea de separación entre elementos
de bloque, y elementos de línea, que ocupan el tamaño del contenido. Por ejemplo, los
elementos de bloque son div, los form, h1, h2, y los elementos de línea son la A, el
image, input, bold. Otros conceptos son el bundle app, que es la división de
dependencias, como GOOL o webpacks. La idea es unificar dependencias de un proyecto en
un fichero o en varios, dependiendo de la complejidad. El diseño responsive consiste en
adaptarse a diferentes pantallas, resoluciones, y que está compuesto por una serie de
diseños y métodos principales para tener un diseño responsive. Y es tener flexible
grids, fluid images y media queries, que son mallas flexibles, imágenes que sean fluidas
y queries de media. Un ejemplo de librería es Bootstrap. Bootstrap es una librería de
CSS y JavaScript que se puede combinar para crear webs. Define con múltiples componentes
reusables con grids responsive. Define clases preexistentes de CSS, por lo que es muy
útil para hacer prototipos, por ejemplo. Bootstrap tiene estilos que se pueden alterar
en el identificador de la clase. Por ejemplo, podemos definir el class como col-lg-6,
donde el lg es un breakpoint que se utiliza para pantallas grandes, para mayor o igual a
992 píxeles para ajuste de pantallas, y 6 es el número máximo de columnas que van a
ocupar, el column script. Recordar que los breakpoints son puntos específicos en el
diseño de una página web donde el diseño cambia para adaptarse a diferentes tamaños de
pantalla. El sistema de mallas de Bootstrap cuenta con una jerarquía de 12 columnas que
pueden ser fijas o fluidas. El sistema siempre contiene un contenedor, que es la clase
container, filas, que es la clase row, y las columnas, que es la clase col. Por ejemplo,
podemos tener un div class, columna-12, y columna-lg-6. Por defecto, Bootstrap es una
librería que tiene en cuenta el desarrollo mobile first. Por eso las definiciones de las
clases para tipo móvil no tienen un identificador como el lg, que es para pantallas
grandes, ya que es el valor por defecto. Con la definición anterior del div class, pues
con esto decimos que para pantallas pequeñas de Bootstrap, no tiene identificar para
ello porque el valor por defecto es mobile first, pues va a ocupar el ancho máximo de 12
columnas. Para pantallas largas, mayor o igual a 992 píxeles, pues la columna se
ajustará para ocupar 6 columnas. Al darle en inspeccionar elemento o a la herramienta de
desarrollo del navegador, pues podemos elegir la opción de simular una web de manera
responsive o alterando entre dispositivos. En cuanto a páginas relevantes para el diseño
de CSS, pues tenemos pure.css. Luego tenemos la diferencia entre las páginas dinámicas y
las estáticas, donde el contenido dinámico se genera en la aplicación del servidor, que
manda al servidor web el navegador al usuario. Para reducir el número de peticiones,
pues utilizamos la cacheta. Luego tenemos librerías como React, donde React es una
librería en JavaScript, no es un framework, y se utiliza en un conjunto con otras
librerías para reducir o simplificar el desarrollo. Podemos definir varias componentes o
módulos, lo que permite testear los módulos por separado y reutilizar esas componentes.
Por ejemplo, una imagen de un usuario se puede reutilizar en varias partes del código.
Compila su propio DOM, conocido como Virtual DOM, en memoria. Es una representación en
memoria del DOM de la web. Lo que hace es realiza una comparativa entre el DOM
actualizado y el previo, para ver qué elementos del árbol han cambiado. Lo que hace
React es que actualiza el Virtual DOM y lo compara con la versión anterior del Virtual
DOM. Si se produce algún cambio, solo se actualiza ese cambio en la DOM de la web.
