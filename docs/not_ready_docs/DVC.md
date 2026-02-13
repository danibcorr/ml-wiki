# DVC

- Para inicializar un proyecto: dvc init
- Para desactivar el uso de analíticas: dvc config core.analytics false
- Cuando iniciamos el proyecto con DVC se añadirán nuevos ficheros a la carpeta .dvc, un .gitignore y un config → Hacemos un git add, un commit y un push al repo.
- Forzar la reproduuccion de un pipeline incluso si ya esta cacheado: dvc repro --force, en caso contrario: dvc repro
-