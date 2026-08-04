# Extraer `pyndicators` a su propio repositorio

`pyndicators` vive ahora como subcarpeta del repo `TensorFlow-stocks-…`. Antes de
publicar en PyPI conviene moverlo a su propio repo para que:

- los workflows de `.github/` se activen (GitHub Actions solo lee `.github/` de la
  raíz del repositorio),
- la URL del proyecto en PyPI apunte a un repo limpio,
- el `pip install` y el `git clone` sean directos.

## Comprobado

Un `git subtree split --prefix=pyndicators` produce una historia con el contenido de
`pyndicators/` **en la raíz** (`src/`, `pyproject.toml`, `.github/`…) y conserva los
commits que tocaron la carpeta. Verificado en este repo.

## Pasos

1. Crea un repositorio vacío en GitHub, p. ej. `Leci37/pyndicators` (sin README ni
   licencia, para no chocar).

2. Desde la raíz del repo actual, extrae la historia de la carpeta a una rama:

   ```bash
   git subtree split --prefix=pyndicators -b pyndicators-only
   ```

3. Publícala como `main` del repo nuevo:

   ```bash
   git push https://github.com/Leci37/pyndicators.git pyndicators-only:main
   ```

4. Clona el repo nuevo y verifica que todo funciona en la raíz:

   ```bash
   git clone https://github.com/Leci37/pyndicators.git
   cd pyndicators
   pip install -e ".[dev]"
   pytest -q
   python -m build && twine check dist/*
   ```

5. (Opcional) borra la rama temporal en el repo original:

   ```bash
   git branch -D pyndicators-only
   ```

## Después

- En el repo nuevo, `.github/workflows/ci.yml` y `release.yml` ya funcionan.
- Configura *Trusted Publishing* en PyPI apuntando a `Leci37/pyndicators` +
  `release.yml` (ver [releasing](docs/releasing.md)).
- Sube el tag `v0.1.0` para publicar.
- En el repo original puedes dejar `pyndicators/` como estaba o sustituirlo por un
  puntero al repo nuevo en el README.
