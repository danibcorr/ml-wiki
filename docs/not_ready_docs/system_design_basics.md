- CAP theorem para sistemas distribuidos.

- Los load balancer se pueden colocar entre el usuario y el servidor, entre servidores y
  aplicaciones y entre aplicaciones y bases de datos. Existen varios algoritmos para los
  balanceadores de carga. Podemos poner varios load balancer por redundancia, ya que
  pueden ser tambien un punto de fallo.

- Para reducir el compute/coste de operaciones petitivas utilizamos caché que suele ser
  más rápida de acceder que a una base de datos. Eso se conoce como caché de
  aplicaciones.

- Content Delivery Network (CDN) es muy útil para servir contenido estático, cachea
  información cerca del usuario, lo que permite reducir latencia.

- El problema de la caché es cuando la información que almacena no corresponde con la
  información (la última información más actual) de la base de datos. Para solventarlo,
  existen sistemas de invalidación de caché. También existen mecanismos para eliminar
  información de la caché.

- En SQL podemos mejorar rendimiento de queries utilizando indices, p.ej. para
  identificar personas de manera única, en vez de recorrer todas las filas. Esto mejora
  el rendimiento de lectura pero empeora el rendimiento de escritura.

- Podemos dividir bases de datos grandes, creando particiones. Para ello existen técnicas
  de partición.
