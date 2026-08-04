# Handover — publicar `pyndicators` en PyPI

Documento de traspaso para llevar `pyndicators` de "carpeta dentro de otro repo" a
librería publicada e instalable con `pip install pyndicators`.

---

## 0. Punto de partida (estado actual)

- **Dónde está el código:** carpeta `pyndicators/` dentro del repo
  `Leci37/TensorFlow-stocks-prediction-Machine-learning-RealTime`, rama
  `claude/stocks-prediction-review-bud91u` (commit `37b909e` o posterior).
- **Qué es:** 160 indicadores técnicos en puro `numpy`+`pandas`, verificados
  contra TA-Lib / py_ti y point-in-time. Docs en `pyndicators/docs/`.
- **Verificado ya:** compila (`python -m build`), pasa `twine check`, instala en
  entorno limpio solo con numpy+pandas, `catalog.csv` y `py.typed` van dentro del
  wheel. Tests: 129 en verde.
- **Qué falta:** 3 pasos manuales (crear repo → extraer → publicar), abajo.

Tiempo estimado total: **~20-30 min**.

---

## 1. Prerrequisitos (una sola vez)

- Cuenta de GitHub (usuario `Leci37`).
- Cuenta de [PyPI](https://pypi.org) y de [TestPyPI](https://test.pypi.org)
  (recomendado para probar antes).
- `git`, Python ≥ 3.9, y las herramientas de build:
  ```bash
  pip install build twine
  ```

---

## 2. Paso 1 — Crear el repo vacío en GitHub

1. GitHub → New repository.
2. Nombre: **`pyndicators`** (owner `Leci37`).
3. **Sin** README, **sin** .gitignore, **sin** licencia (para no chocar con la
   historia que vamos a empujar).
4. Crear. Anota la URL: `https://github.com/Leci37/pyndicators.git`.

> Si prefieres otro nombre, cámbialo también en `pyproject.toml`
> (`[project.urls]`) y en `name = "..."`. Verifica antes que está libre en PyPI:
> `https://pypi.org/project/<nombre>/` debe dar 404.

---

## 3. Paso 2 — Extraer `pyndicators` con su historia y subirlo

Desde la raíz del repo actual (`TensorFlow-stocks-…`):

```bash
# 1) extrae la historia de la carpeta pyndicators/ a una rama (queda en la RAÍZ)
git subtree split --prefix=pyndicators -b pyndicators-only

# 2) empújala como main del repo nuevo
git push https://github.com/Leci37/pyndicators.git pyndicators-only:main

# 3) borra la rama temporal
git branch -D pyndicators-only
```

**Verificación** (clona el repo nuevo y prueba que funciona en la raíz):

```bash
cd /tmp
git clone https://github.com/Leci37/pyndicators.git
cd pyndicators
pip install -e ".[dev]"
pytest -q                      # deben pasar todos
python -m build && twine check dist/*   # ambos PASSED
python -c "import pyndicators as pt; print(len(pt.catalog()))"   # ~330
```

Al estar `pyndicators` en la raíz del repo nuevo, `.github/workflows/ci.yml` y
`release.yml` ya se activan solos.

---

## 4. Paso 3 — Publicar en PyPI

### Opción A (recomendada): Trusted Publishing, sin tokens

1. En PyPI: crea el proyecto o ve a *Your projects* → **Publishing** → *Add a
   pending publisher*:
   - PyPI Project Name: `pyndicators`
   - Owner: `Leci37`  ·  Repository: `pyndicators`
   - Workflow: `release.yml`  ·  Environment: `pypi`
2. En el repo nuevo, sube el tag de versión:
   ```bash
   git tag v0.1.0
   git push origin v0.1.0
   ```
3. El workflow `.github/workflows/release.yml` construye y publica automáticamente.
   Míralo en la pestaña *Actions* del repo.

### Opción B: subida manual con token

```bash
# (recomendado) probar primero en TestPyPI
twine upload --repository testpypi dist/*
pip install --index-url https://test.pypi.org/simple/ pyndicators   # smoke test

# publicar de verdad
twine upload dist/*     # pide usuario __token__ y tu API token de PyPI
```

---

## 5. Verificación final (post-publicación)

```bash
cd /tmp && python -m venv v && v/bin/pip install pyndicators
v/bin/python -c "import pyndicators as pt; print(pt.__version__, len(pt.catalog()))"
```

Comprueba también la ficha pública: `https://pypi.org/project/pyndicators/`.

---

## 6. Problemas típicos

| Síntoma | Causa / arreglo |
|---|---|
| `File already exists` al subir | Esa versión ya está en PyPI. Sube `version` en `pyproject.toml` y reconstruye (no se puede sobrescribir una versión). |
| El workflow de release no arranca | El tag debe empezar por `v` (`v0.1.0`). Y `pyndicators` debe estar en la raíz del repo nuevo. |
| `twine check` avisa del README | Asegura `readme = "README.md"` en `pyproject.toml` (ya está). |
| Trusted publisher rechazado | El owner/repo/workflow/environment deben coincidir EXACTOS con lo configurado en PyPI. |
| Falta `catalog.csv` al instalar | Ya cubierto por `[tool.setuptools.package-data]`; verifica que sigue en `pyproject.toml`. |

---

## 7. Siguientes versiones (para más adelante)

- `v0.1.x`: subir cobertura de indicadores (quedan velas, statistics, cycles) y
  refinar `factor_style` por indicador.
- Modo streaming incremental O(1) (ver `docs/realtime.md`).
- Los indicadores Hilbert/DSP (SAR, MAMA, HT_TRENDLINE) tras el extra `[talib]`.

Checklist por release en `docs/releasing.md`. Cómo añadir indicadores en
`docs/contributing.md`.
