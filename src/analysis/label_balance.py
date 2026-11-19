"""目的変数の符号(上昇/下降)の割合を、期間別・年別に数える(クラスの偏り)。

使い方(リポジトリのルートから):
    python -m src.analysis.label_balance [--target spx_ret_fwd5] [--out results/analysis/label_balance.md]

「いつも上昇」と答えたときの正解率の基準が、これ(上昇の割合)。
"""
import argparse
from pathlib import Path

import pandas as pd

from src.analysis._md import table


def balance(df: pd.DataFrame, target: str, other: str = "spx_ret_fwd1") -> pd.DataFrame:
    """年ごと(と全体)の、日数・上昇・下降・上昇割合・上昇:下降。other は比較用(翌日リターン)の列。"""
    def row(label, q):
        up = int((q[target] > 0).sum())
        down = len(q) - up
        r = {"year": label, "n": len(q), "up": up, "down": down, "up_share": up / len(q), "ratio": up / down if down else float("nan")}
        if other in q:
            r["other_up_share"] = float((q[other].dropna() > 0).mean())
        return r
    rows = [row(y, q) for y, q in df.groupby(df["date"].dt.year)] + [row("全体", df)]
    return pd.DataFrame(rows)


def main() -> int:
    from src.models import study1
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="spx_ret_fwd5")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    df, _ = study1.load_dataset("numeric+nyt", a.target, "common")
    b = balance(df, a.target)
    b["up_share"] = (b["up_share"] * 100).round(1).astype(str) + "%"
    text = f"# 目的変数 {a.target} の上昇・下降の割合(共通 {len(df):,} 行)\n\n{table(b.round(3), index=False)}\n"
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text, encoding="utf-8")
        print("保存:", a.out)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
