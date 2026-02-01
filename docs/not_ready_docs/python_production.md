- Podemos utisar valores dimales:

```python
from decimal import Decimal

RATES = {
    ("USD", "EUR"): Decimal("0.91")
}
```

- Podemos utilizar PathLib (Path) para definir directorios a archivos porque es más
  limpio, y es multi plataforma.

- Los dataclasses de dataclass definen internamente los métodos: **init**, **repr**,
  **eq**

- Podemos añadir notas a las excepciones capturadas en Python:

```python
try:
    ...
except Exception as e:
    e.add_note("Nota")
```

# Python Dataclass

- Modificar atributos de un dataclass que esta frozen (inmutable):

```python
@dataclass(frozen=True)
class User:
    name: str

    def __post_init__(self):
        norm_name = self.name.lower()
        object.__setattr__(self, "name", norm_name)
```

Que sea frozen significa que solo aplica a los atributos.

- Cuando tengo un método de una clase que devuelve el mismo tipo de la clase podemos
  usar:

```python
from typing import Self

class User:
    def metodo() -> Self:
        return objeto_de_tipo_User
```

Tambien en el decorador de los dataclasses podemos utilizar:

@dataclass(frozen=True, order=True, slots=True)

- El order permite realizar comparativas entre objetos del dataclass, p.ej. u1 < u2
- Los slots permite eliminar el diccionario **dict** que se crea para almacenar los
  atributos de la clase, lo que lo hace más rápido, pero impide la creacion de nuevo
  atributos no declarados previamente.

Tambien podemos crear clases abstractas para dataclasses
