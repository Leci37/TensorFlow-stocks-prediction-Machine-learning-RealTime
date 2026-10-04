"""Where the pipeline reads and writes.

Every path is anchored at the repository root, so the scripts in pipeline/ work from any
working directory. The output folders are created on import, because outputs/ is not in git.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA = ROOT / "data"            # price history (gitignored, except data/sample/)
SAMPLE = DATA / "sample"        # the RIVN 5-min file the tutorial runs on
ALPACA = DATA / "alpaca"        # written by pipeline/0_download_alpaca.py
RAW_ALPHA = DATA / "RAW_alpha"  # written by pipeline/0_download_alphavantage.py
OUTPUTS = ROOT / "outputs"      # everything the pipeline writes (gitignored)

MODELS = OUTPUTS / "models"                  # trained .h5 models
PLOTS = OUTPUTS / "plots"                    # model diagrams and training curves
VAL_DATA = OUTPUTS / "val_data"              # held-out windows saved by 2_tune for 3_evaluate
MODEL_INFO = OUTPUTS / "model_info"          # training history per model
COLUMNS_SELECT = OUTPUTS / "columns_select"  # chosen indicator columns (.json)
SCALERS = OUTPUTS / "scalers"                # fitted MinMax scalers (.pkl)
CLASS_WEIGHT = OUTPUTS / "class_weight"      # class weights (.pkl)
INITIAL_WEIGHTS = OUTPUTS / "initial_weights"
LOGS = OUTPUTS / "logs"                      # TensorBoard hparams logs

for _d in (DATA, ALPACA, RAW_ALPHA, MODELS, PLOTS, VAL_DATA, MODEL_INFO, COLUMNS_SELECT, SCALERS, CLASS_WEIGHT, INITIAL_WEIGHTS, LOGS):
    _d.mkdir(parents=True, exist_ok=True)
