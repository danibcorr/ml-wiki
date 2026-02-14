# Cómo contribuir a wiki-personal

¡Hola y bienvenido! Nos alegra que estés interesado en contribuir al proyecto
**wiki-personal**.

Apreciamos todo tipo de contribuciones, ya sea contenido, reportes de errores o nuevas
funciones.

## ¿Quieres unirte al equipo?

Si te interesa colaborar más estrechamente, no dudes en contactarnos.

## Cómo contribuir

Existen dos formas principales de contribuir:

- Solicitar una **nueva función** o reportar un **error**
- Enviar una **contribución de contenido o código** (documentación, mejoras del sitio,
  correcciones, etc.)

### Paso 1: Abrir un issue

Comienza abriendo un **issue** para describir tu propuesta o el problema que
encontraste. Esto nos ayuda a evaluar y guiar el trabajo antes de hacer cambios.

> Si tu cambio es pequeño (por ejemplo, corregir un error tipográfico), puedes omitir
> este paso y abrir directamente un pull request.

### Paso 2: Realizar los cambios

1. Haz un **fork** del repositorio.
2. Clónalo localmente.
3. Crea una nueva rama para tu cambio.
4. Configura tu entorno de desarrollo (ver [Configurar entorno](#configurar-entorno)).
5. Realiza tus cambios y verifica que todo funcione correctamente.

### Paso 3: Enviar un Pull Request

Cuando tus cambios estén listos, abre un **pull request (PR)** desde tu rama al branch
`main`.

### Paso 4: Revisión de código/contenido

Un mantenedor revisará tu PR y puede dejar comentarios. Es posible que te pidan:

- Ajustar el contenido o el formato
- Corregir errores
- Mejorar la estructura de la documentación

> Asegúrate de corregir cualquier problema antes de solicitar una nueva revisión.

### Paso 5: Merge

Una vez aprobado, un mantenedor hará el **merge** del pull request.

## Configurar entorno

Puedes configurar tu entorno de desarrollo de dos formas:

### Opción 1: Contenedor de desarrollo

Soportamos **Visual Studio Code**, **GitHub Codespaces** y **JetBrains IDEs** con
contenedores de desarrollo preconfigurados para un setup fácil.

### Opción 2: Configuración local

Para trabajar localmente:

1. Instala `git` y `node.js`.
2. Clona el repositorio.
3. Ejecuta:

```bash linenums="1"
make install
```

Esto instalará todas las dependencias necesarias.

## Comandos útiles

El proyecto usa un **Makefile** para automatizar tareas comunes:

| Comando           | Descripción                                     |
| ----------------- | ----------------------------------------------- |
| `make dev`        | Inicia el servidor de desarrollo                |
| `make build`      | Construye el sitio para producción              |
| `make serve`      | Sirve el sitio construido localmente            |
| `make format`     | Formatea el código (JS, CSS, MD)                |
| `make typecheck`  | Revisa tipos con TypeScript                     |
| `make setup-i18n` | Configura traducciones                          |
| `make clean`      | Limpia cache y dependencias                     |
| `make all`        | Ejecuta `build` y todas las tareas relacionadas |

> Recomendamos usar `make dev` para previsualizar cambios mientras editas documentación
> o contenido.

## Estilo de documentación

Un buen contenido/documentación debe ser:

- Claro y conciso
- Bien estructurado con títulos, subtítulos y listas
- Consistente en formato y estilo

Si estás agregando funciones o componentes de Docusaurus, incluye ejemplos y
descripciones completas.

## Otras formas de contribuir

Si no quieres escribir código, aún puedes ayudar:

- Mejorar la documentación o ejemplos
- Corregir errores tipográficos
- Sugerir nuevas funciones o mejoras de usabilidad
- Ayudar a organizar y responder issues

Gracias por ser parte de la aventura de **wiki-personal**. ¡Nos alegra tenerte aquí! Si
tienes preguntas, no dudes en contactarnos o abrir un issue.
