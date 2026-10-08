"""時間順の walk-forward(拡張窓)の分割。

年ごとに、「その年より前のすべて」で学習し、その年をテストする。学習とテストの境界に embargo 行の空きを置く
(目的変数が t+1〜t+5 の価格を使うため、直前の embargo 行のラベルは、テスト期間の価格を含む)。
学習の内側にも、時間順の検証を作る(最後の share を検証、その前との境界にも空きを置く)。

行は日付の昇順に並んでいる前提。インデックスは、その並びでの位置。
"""
import numpy as np
import pandas as pd


def make_folds(dates, test_years, embargo: int) -> list:
    """[{'year', 'train', 'test'}, ...]。train/test は位置の配列。"""
    idx = pd.DatetimeIndex(dates)
    if not idx.is_monotonic_increasing:
        raise ValueError("日付は昇順に並べておくこと")
    folds = []
    for year in test_years:
        test = np.where(idx.year == year)[0]
        if len(test) == 0:
            continue
        end = max(test[0] - embargo, 0)  # 学習は、テスト開始の embargo 行前まで
        folds.append({"year": int(year), "train": np.arange(0, end), "test": test})
    return folds


def inner_split(n: int, share: float, embargo: int) -> tuple:
    """学習の内側: (内側の学習, 検証)。検証は最後の share。両者の間に embargo 行の空き。"""
    n_val = int(round(n * share))
    val = np.arange(n - n_val, n)
    train = np.arange(0, max(n - n_val - embargo, 0))
    return train, val
