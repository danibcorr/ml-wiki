---
sidebar_position: 9
authors:
  - name: Daniel Bazo Correa
description: Crea y almacena tus artefactos en repositorios.
title: Creacion de APIs
toc_max_heading_level: 3
---

Arquitectura monolítica es cuando creamos software combinado todo en un mismo grupo
incluso cuando su funcionalidad es variada. El problema es la escalabilidad, la
flexibilidad y el mantenimiento. Una solución es utilizar micro servicios que consiste
en crear un código específico para cada cosa está modularidad, pues permite resolver los
fallos y los problemas anteriores aquí por ejemplo tendríamos una aplicación que podría
tener por ejemplo un modelo para la inferencia y luego un modelo que se crearía durante
el proceso de entrenamiento, pues la idea es separar la lógica de ambos. Créame un
código específico que puede ser puesto en producción, de manera independiente.

También utilizar arquitecturas basadas en micro servicios, pues permite utilizar
tecnologías diferentes para cada tipo de micro servicios, por ejemplo para la creación
de un modelo de Machine Lerning. Podríamos utilizar Python para inferencia y crear una
API. Podríamos utilizar Go o para Pipeline de datos. Podríamos utilizar escala con los
micros de servicios, pues al final podemos escalar componentes de manera independiente,
reduce el impacto de fallo, ya que un código no afecta otro para la comunicación entre
servicios, pues se usan las apps que definen reglas y protocolos para dicha
comunicación, sin importar el lenguaje de programación utilizado para ello utilizan
protocolos de comunicación como puede ser GRPC, Rest o HTTP.

Las Apis pueden ser de dos maneras de manera síncrona o asíncrona, y sin embargo, existe
una serie de limitaciones las Apis síncronas, pues es cuando los usuarios tienen que
esperar peticiones de los usuarios de adelante entonces si yo tengo hasta acá usuarios,
el usuario que tiene que esperar en el segundo es de latencia donde K sería el usuario
carísimo. Esa latencia es el tiempo de respuesta desde que el usuario mando una petición
hasta que he resuelto. Al final esto supone un problema de escalabilidad, y para ello
están las Apis asíncronas que permiten ejecutar ciertas acciones de la aplicación de
manera concurrente. En vez de secuencial una API puede tener dos procesos en general
tareas de transferencia de datos, lo que se conoce como el I/O que son operaciones de
entrada y salida o operaciones de escritura y tenemos tareas de procesado que están
relacionadas con la CPU al final las limitaciones en la entrada y salida, pues depende
de lo rápido que seas para transferir datos acceder a espacio de memoria de la capacidad
de Internet, y luego tenemos la limitaciones en la CPU que son a nivel de hardware. Por
tanto, la programación asíncrona no depende de la velocidad de la CPU y se utiliza
sobretodo para optimizar procesos.

