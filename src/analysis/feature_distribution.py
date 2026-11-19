"""研究1の特徴量の分布(平均・標準偏差・最小・最大・歪度・尖度・時期のずれ・相関・外れ値)を表にする。

使い方(リポジトリのルートから):
    python -m src.analysis.feature_distribution [--out results/analysis/feature_distribution.md]

- shift = (2013年以降の平均 − 2012年以前の平均) / 全体の標準偏差。絶対値が1前後なら、時期で平均が1標準偏差ずれている。
- 行は、数値+NYTの全特徴量が揃う共通行(src.models.study1.load_dataset の common)。
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis._md import table


def describe(df: pd.DataFrame, cols, split_year: int = 2013) -> pd.DataFrame:
    x = df[cols]
    d = x.describe(percentiles=[0.01, 0.5, 0.99]).T[["mean", "std", "min", "1%", "50%", "99%", "max"]]
    d["skew"], d["kurt"] = x.skew(), x.kurt()
    early, late = df[df["date"].dt.year < split_year][cols], df[df["date"].dt.year >= split_year][cols]
    d["shift"] = (late.mean() - early.mean()) / x.std()
    d["n_outlier_5sd"] = ((x - x.mean()).abs() / x.std() > 5).sum()
    return d


def high_correlations(df: pd.DataFrame, cols, top: int = 8) -> pd.Series:
    c = df[cols].corr().abs()
    upper = c.where(np.triu(np.ones(c.shape), 1).astype(bool))
    return upper.stack().sort_values(ascending=False).head(top)


def yearly_mean(df: pd.DataFrame, cols) -> pd.DataFrame:
    return df.groupby(df["date"].dt.year)[list(cols)].mean()


def main() -> int:
    from src.models import study1
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    df, cols = study1.load_dataset("numeric+nyt", "spx_ret_fwd5", "common")
    text = (f"# 特徴量の分布(共通 {len(df):,} 行、{df['date'].min().date()}〜{df['date'].max().date()})\n\n"
            f"{table(describe(df, cols).round(4))}\n\n## 相関が高い組\n\n{table(high_correlations(df, cols).round(3).rename('相関').to_frame())}\n\n"
            f"## 水準系の年別平均\n\n{table(yearly_mean(df, ['tnx_level', 'term_spread', 'vix_level', 'spx_rv_20d']).round(3))}\n")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text, encoding="utf-8")
        print("保存:", a.out)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
