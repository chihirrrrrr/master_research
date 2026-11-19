# src/ のコード一覧(研究で使うコードはすべてここ)

実行は、リポジトリのルートから `.venv/bin/python -m src.…`。テストは `tests/`、設定は `configs/`、まとめ実行は `Makefile`。

| 役割 | ファイル | 実行(Makefile) |
|---|---|---|
| 価格の取得(yfinance) | `src/data/fetch_prices.py` | `make fetch` |
| 生データの整合確認 | `src/data/verify_raw.py` | `make verify` |
| 研究1の下準備(S&P500のOHLC、金利) | `src/data/make_market_base.py` | `make market-base` |
| 研究2の数値特徴量・目的変数 | `src/features/numeric.py` | `make features` |
| 研究1の数値特徴量18個・目的変数 | `src/features/market_level.py` | `make market-features` |
| NYT見出しの感情(FinBERT)5個 | `src/features/nyt_text.py` | `make nyt-features` |
| 分割(walk-forward、内側の検証) | `src/eval/walk_forward.py` | `study1.py` から呼ばれる |
| 指標(R²、AUC、Clark–West など) | `src/eval/metrics.py` | 同上 |
| **研究1の学習・予測・評価の本体** | `src/models/study1.py` | `make study1-dev` / `study1-all` |
| 結果の年別表・期間別表 | `src/analysis/summarize_runs.py` | `make analysis` |
| 特徴量の分布 | `src/analysis/feature_distribution.py` | `make analysis` |
| 目的変数の上昇・下降の割合 | `src/analysis/label_balance.py` | `make analysis` |

分析メモ(`docs/analysis/`)の表の数値は、`src/analysis/` のスクリプトで再現できる。
| 累積リターン(5・10・20日)の期間比較 | `src/analysis/compare_horizons.py` | `make analysis` |
| 平均リターンと累積リターンの関係(5・10・20日) | `src/analysis/target_relation.py` | `make analysis` |
