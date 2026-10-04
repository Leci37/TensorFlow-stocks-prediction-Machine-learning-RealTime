# legacy/

Everything the tutorial pipeline does not use, kept for reference. Nothing here is imported by
`stockpred/` or `pipeline/`, and the code inside was moved, not edited.

| Folder | What it is |
|---|---|
| [`realtime_v1/`](realtime_v1/README.md) | The original system (2022–2023): ~600 indicators, 36 models per stock, live scoring every 15 minutes, alerts to Telegram and Twitter. Moved as one block with its own `_KEYS_DICT.py`, `Utils/`, data folders and `requirements.txt`, so it still runs from its own folder. Its old docs are in `realtime_v1/docs/` |
| `reinforcement_learning/` | Reinforcement-learning trading environments and agents (a MetaTrader simulator, gym environments) |
| `research/` | Side experiments: news sentiment (`news_sentiment/`, `d_sentiment/`), model fine-tuning (`model_finetune/`), Dukascopy history (`dukascopy/`), three unused helpers from `utilW3/`, an older copy of the indicator package, and the tutorial's `Get_technical_indicators.py`, which imports a `features_W3` package that was never in this repository |
| `useless/` | The author's own folder of abandoned code (XTB and investing.com APIs, early sklearn/XGBoost tests) |

Deleted in the 2026 reorganization (still in git history): 75 images no document used (mostly
photos from the iaCarry project), an empty TA-Lib wheel, a duplicate `requirements_x.y.z.txt`,
the unprocessed `tutorial_RIVN_5min.csv` that nothing read, the placeholder `gitkeep.txt` files of
the output folders, and two CI workflows copied from another project that could not run.
