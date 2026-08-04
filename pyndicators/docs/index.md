# pyndicators

Indicadores técnicos para **machine learning financiero**, en puro `numpy` + `pandas`.

- **Point-in-time** — ningún indicador registrado mira al futuro.
- **Verificado** — cada uno con paridad byte-a-byte contra TA-Lib (o py_ti).
- **Instalación limpia** — núcleo sin dependencias que compilar.

```python
import pyndicators as pt
features = pt.compute(df)                 # todos
features = pt.compute(df, family="Candle")  # solo velas
```

## Mapa de la documentación

| Página | Para qué |
|---|---|
| [Instalación](installation.md) | `pip install`, extras opcionales |
| [Guía de uso](usage.md) | cómo ejecutar y **cómo elegir qué generar** (todo / por familia / por tipo / point-in-time) |
| [Tiempo real](realtime.md) | generar al cerrar cada vela (1m, 5m, 15m…), streaming |
| [Catálogo](catalog.md) | los 3 ejes de clasificación y cómo explorar los ~160 indicadores |
| [Diseño interno](design.md) | registro, motor, grafo de dependencias, validación PIT |
| [Contribuir](contributing.md) | cómo añadir un indicador nuevo |
| [Publicar](releasing.md) | construir y subir a PyPI |
