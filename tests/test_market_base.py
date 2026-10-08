"""研究1の土台の表(S&P500のOHLC + 金利 + 為替)のテスト。

合成データのテスト(補完しない・営業日に並べる)は常に動く。実データのテストは、作成前ならスキップする。
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data import make_market_base as mb

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data" / "interim" / "market_base.parquet"
needs_base = pytest.mark.skipif(not BASE.exists(), reason="未作成(python -m src.data.make_market_base)")


def synthetic_macro() -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-01", periods=10)
    rows = []
    for ticker in [mb.SPX] + list(mb.CLOSE_ONLY):
        for i, d in enumerate(dates):
            v = 100.0 + i
            rows.append({"ticker": ticker, "date": d, "open": v, "high": v + 1, "low": v - 1, "close": v + 0.5})
    df = pd.DataFrame(rows)
    # 金利 ^TNX の 3 日目を欠けさせる(債券市場だけ休み)。^TYX には S&P500 の営業日にない週末の行を足す
    df = df[~((df["ticker"] == "^TNX") & (df["date"] == dates[2]))]
    weekend = pd.DataFrame([{"ticker": "^TYX", "date": pd.Timestamp("2020-01-04"), "open": 1.0, "high": 1.0,
                             "low": 1.0, "close": 999.0}])
    return pd.concat([df, weekend], ignore_index=True)


def test_aligned_to_spx_days_without_imputation():
    macro = synthetic_macro()
    out = mb.build_market_base(macro, pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-14"))
    assert list(out.columns) == mb.COLUMNS
    assert len(out) == 10 and out["date"].is_monotonic_increasing
    assert out.loc[2, "tnx_10y"] != out.loc[2, "tnx_10y"]  # 欠けた日は欠損のまま(補完しない)
    assert out["tnx_10y"].isna().sum() == 1
    assert not (out["tyx_30y"] == 999.0).any()  # S&P500の営業日にない日(週末)は入らない
    assert out["spx_close"].iloc[0] == 100.5 and out["spx_open"].iloc[0] == 100.0  # 値はそのまま


def test_period_is_cut_to_the_requested_range():
    out = mb.build_market_base(synthetic_macro(), pd.Timestamp("2020-01-03"), pd.Timestamp("2020-01-08"))
    assert out["date"].min() >= pd.Timestamp("2020-01-03") and out["date"].max() <= pd.Timestamp("2020-01-08")


@needs_base
def test_real_table_contract():
    t = pd.read_parquet(BASE)
    assert list(t.columns) == mb.COLUMNS
    assert t["date"].is_unique and t["date"].is_monotonic_increasing
    # 期間はNYTの期間
    start, end = mb.nyt_period()
    assert t["date"].min() >= start and t["date"].max() <= end
    assert t["date"].min() == pd.Timestamp("2008-01-02") and t["date"].max() == pd.Timestamp("2021-12-31")
    assert len(t) == 3526 and t.shape[1] == 9
    # S&P500のOHLCは欠損なし・整合
    assert t[mb.SPX_COLUMNS].notna().all().all()
    top = t[["spx_open", "spx_close", "spx_low"]].max(axis=1)
    bottom = t[["spx_open", "spx_close", "spx_high"]].min(axis=1)
    assert (t["spx_high"] >= top - 1e-9).all() and (t["spx_low"] <= bottom + 1e-9).all()
    # 金利の欠けは少ない(米国債の休場日)
    assert t[["irx_3m", "fvx_5y", "tnx_10y", "tyx_30y"]].isna().sum().max() <= 5


@needs_base
def test_values_are_unmodified_from_raw():
    t = pd.read_parquet(BASE)
    src = mb.latest_prices_dir()
    macro = pd.read_parquet(src / "macro.parquet")
    spx = macro[macro["ticker"] == mb.SPX].set_index("date").sort_index()
    m = t.set_index("date")
    for c in ("open", "high", "low", "close"):
        assert np.array_equal(m["spx_" + c].to_numpy(), spx.loc[m.index, c].to_numpy())
    for ticker, name in mb.CLOSE_ONLY.items():
        raw = macro[macro["ticker"] == ticker].set_index("date")["close"].reindex(m.index)
        assert np.array_equal(m[name].to_numpy(), raw.to_numpy(), equal_nan=True)


def test_spike_detector_finds_a_one_day_jump_only():
    s = pd.Series([100.0, 100.5, 120.0, 100.2, 100.4, 100.1],
                  index=pd.bdate_range("2020-01-01", periods=6))
    assert mb.spike_dates(s, 0.03) == ["2020-01-03"]  # 前後は同じなのに、その日だけ+20%
    trend = pd.Series([100.0, 105.0, 110.0, 115.0, 120.0], index=pd.bdate_range("2020-01-01", periods=5))
    assert mb.spike_dates(trend, 0.03) == []  # 続けて動く(トレンド)は飛び値ではない
    gapped = s.copy()
    gapped.iloc[1] = float("nan")  # 欠損の日は飛ばして比べる
    assert mb.spike_dates(gapped, 0.03) == ["2020-01-03"]


def test_fx_is_not_in_the_table():
    # 為替は含めない(2026-10-08の決定。2008年のドル円・ユーロドルに取得元の日付の崩れがあるため)
    assert not {"dxy", "usdjpy", "eurusd"} & set(mb.COLUMNS)
    assert "為替" in " ".join(mb.EXCLUDED)
