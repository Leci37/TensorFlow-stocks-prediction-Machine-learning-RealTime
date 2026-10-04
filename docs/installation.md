# Installation

The tutorial pipeline needs **Python 3.9–3.11**: the code is written for TensorFlow 2.15 and Keras 2
(it saves weights as `.tf` and uses `tensorflow_addons`), and TensorFlow 2.15 does not install on
Python 3.12.

```bash
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .                   # makes stockpred/ importable from pipeline/
```

`tf.keras.utils.plot_model` (step [9] of `pipeline/1_train.py`) also needs the Graphviz program:
`sudo apt install graphviz` on Debian/Ubuntu, `brew install graphviz` on macOS, or the installer
from graphviz.org on Windows.

## Two packages that changed since the code was written

- **pandas_ta**: version 0.3.14b0, which the indicator code was written for, is no longer on PyPI
  or GitHub. `requirements.txt` installs `pandas-ta-openbb`, a maintained fork that imports as
  `pandas_ta` and works with Python 3.11 and numpy 1.x.
- **TA-Lib**: recent versions (0.5 and later) ship wheels that include the C library, so
  `pip install` is enough. With an older version you must install the C library first.

## Check the install

```bash
python tools/check_imports.py      # needs nothing installed: every stockpred import resolves
python pipeline/1_train.py         # trains on data/sample/ (a few minutes on a laptop CPU)
```

## API keys for the downloaders

The downloaders read their keys from the environment, never from the code:

```bash
export ALPACA_API_KEY=...   ALPACA_API_SECRET=...      # pipeline/0_download_alpaca.py
export ALPHAVANTAGE_API_KEYS=KEY1                      # pipeline/0_download_alphavantage.py
```

The setup guide of the old real-time pipeline (Python 3.8, PyCharm) is in
`legacy/realtime_v1/docs/Readme_setup_installation.md`.
