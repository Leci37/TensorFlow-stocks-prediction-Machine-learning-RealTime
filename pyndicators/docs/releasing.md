# Publicar en PyPI

Estado actual: el paquete **ya compila y pasa `twine check`** (wheel + sdist), se
instala en entorno limpio y funciona solo con `numpy`+`pandas`. Lo único que falta
es la **subida**, que requiere tu cuenta de PyPI.

## Comprobación local (ya verificada)

```bash
pip install build twine
python -m build                 # genera dist/*.whl y dist/*.tar.gz
twine check dist/*              # valida metadatos y README -> PASSED
```

## Opción A — subida manual con token

1. Crea cuenta en [pypi.org](https://pypi.org) (y en [test.pypi.org](https://test.pypi.org)).
2. Genera un **API token** (Account settings → API tokens).
3. Prueba primero en TestPyPI:

   ```bash
   twine upload --repository testpypi dist/*
   pip install --index-url https://test.pypi.org/simple/ pyndicators
   ```

4. Sube a PyPI de verdad:

   ```bash
   twine upload dist/*
   ```

Listo: `pip install pyndicators` funciona para todo el mundo.

## Opción B — Trusted Publishing (recomendada, sin tokens)

PyPI puede confiar en este repositorio y publicar desde GitHub Actions al crear un
tag, sin guardar tokens.

1. En PyPI: proyecto → *Publishing* → añade un *trusted publisher* con el repo, el
   workflow `release.yml` y el *environment* `pypi`.
2. Sube el tag de versión:

   ```bash
   git tag v0.1.0
   git push origin v0.1.0
   ```

El workflow `.github/workflows/release.yml` construye y publica solo.

## Checklist por versión

- [ ] Subir `version` en `pyproject.toml` (semver).
- [ ] Actualizar `CHANGELOG.md`.
- [ ] `python -m build && twine check dist/*`.
- [ ] Tag `vX.Y.Z` (o subida manual).
- [ ] Verificar `pip install pyndicators==X.Y.Z` en un entorno limpio.
