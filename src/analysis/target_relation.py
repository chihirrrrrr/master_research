"""目的変数どうしの関係: 「h日の日次リターンの平均」と「h日累積リターン」が、どれだけ同じ情報か。

使い方(リポジトリのルートから):
    python -m src.analysis.target_relation [--out results/analysis/target_relation.md]

入力: data/interim/market_base.parquet(S&P500の終値)
- 相関が1に近く、符号の一致率が高ければ、目的変数を取り替えても予測の中身は変わらない。
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis._md import table

ROOT = Path(__file__).resolve().parents[2]


def relation(close: pd.Series, horizons=(5, 10, 20)) -> pd.DataFrame:
    r = close.pct_change()
    rows = []
    for h in horizons:
        mean_h = sum(r.shift(-k) for k in range(1, h + 1)) / h * 1e4      # t+1〜t+h の日次リターンの平均 [bp]
        cum_h = (close.shift(-h) / close - 1) * 1e4                        # 終値_{t+h} / 終値_t − 1 [bp]
        d = pd.DataFrame({"mean": mean_h, "cum": cum_h}).dropna()
        rows.append({"期間": f"{h}日", "日数": len(d), "相関": round(float(d.corr().iloc[0, 1]), 5),
                     "比(累積/平均、中央値)": round(float((d["cum"] / d["mean"]).median()), 3),
                     "符号の一致": round(float((np.sign(d["mean"]) == np.sign(d["cum"])).mean()), 4),
                     "累積の標準偏差(bp)": round(float(d["cum"].std()), 1)})
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    close = pd.read_parquet(ROOT / "data" / "interim" / "market_base.parquet").set_index("date")["spx_close"]
    text = "# 日次リターンの平均 と 累積リターン の関係\n\n" + table(relation(close), index=False)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text, encoding="utf-8")
        print("保存:", a.out)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
