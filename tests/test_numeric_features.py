"""数値特徴量のリーク防止テスト(合成データ。実データが無くても動く)。

最重要: 「取引日 T 以降の価格を書き換えても、行 T までの特徴量は変わらない」(= 特徴量は前日までの情報だけ)。
"""
import numpy as np
import pandas as pd
import pytest

from src.features import numeric as nm

N = 420
DATES = pd.bdate_range("2018-01-01", periods=N)
STOCKS = ["AAA", "BBB", "CCC"]


def make_stocks(seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frames = []
    for t in STOCKS:
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, N)))
        open_ = close * np.exp(rng.normal(0, 0.003, N))
        frames.append(pd.DataFrame({
            "ticker": t, "date": DATES, "open": open_, "high": np.maximum(open_, close) * 1.005,
            "low": np.minimum(open_, close) * 0.995, "close": close, "adj_close": close * 0.98,
            "volume": rng.integers(1_000_000, 2_000_000, N).astype(float), "dividends": 0.0, "stock_splits": 0.0,
        }))
    return pd.concat(frames, ignore_index=True)


def make_macro(seed=1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frames = []
    for t in list(nm.SYMBOLS.values()) + ["SPY"]:
        close = 50 * np.exp(np.cumsum(rng.normal(0, 0.01, N)))
        open_ = close * np.exp(rng.normal(0, 0.003, N))
        frames.append(pd.DataFrame({
            "ticker": t, "date": DATES, "open": open_, "high": np.maximum(open_, close), "low": np.minimum(open_, close),
            "close": close, "adj_close": close, "volume": 0.0, "dividends": 0.0, "stock_splits": 0.0,
        }))
    return pd.concat(frames, ignore_index=True)


def perturb(df: pd.DataFrame, from_date: pd.Timestamp) -> pd.DataFrame:
    """from_date 以降の価格・出来高を大きく書き換える。"""
    out = df.copy()
    m = out["date"] >= from_date
    for c in ("open", "high", "low", "close", "adj_close", "volume"):
        out.loc[m, c] = out.loc[m, c] * 3.0
    return out


def test_feature_count_and_no_target_names():
    assert len(nm.FEATURE_COLUMNS) == 27
    assert not set(nm.FEATURE_COLUMNS) & set(nm.TARGET_COLUMNS)
    assert not any(c.startswith("ret_next") or c.startswith("spy_ret_") for c in nm.FEATURE_COLUMNS)


def test_features_do_not_look_ahead():
    stocks, macro = make_stocks(), make_macro()
    T = DATES[300]
    base, _ = nm.build_numeric(stocks, macro)
    pert, _ = nm.build_numeric(perturb(stocks, T), perturb(macro, T))
    key = ["ticker", "date"]
    past_base = base[base["date"] <= T].sort_values(key).reset_index(drop=True)
    past_pert = pert[pert["date"] <= T].sort_values(key).reset_index(drop=True)
    # 行 T までの特徴量は、T 以降の価格に依存しない(行 T は T-1 の終値までの値)
    pd.testing.assert_frame_equal(past_base, past_pert)


def test_features_do_use_previous_day():
    # 対照: T を書き換えると、翌日(T+1)の行の特徴量は変わる(= テストが空振りしていない)
    stocks, macro = make_stocks(), make_macro()
    T = DATES[300]
    base, _ = nm.build_numeric(stocks, macro)
    pert, _ = nm.build_numeric(perturb(stocks, T), perturb(macro, T))
    nxt = DATES[301]
    a = base[base["date"] == nxt].sort_values("ticker")[nm.STOCK_FEATURES + nm.MACRO_FEATURES]
    b = pert[pert["date"] == nxt].sort_values("ticker")[nm.STOCK_FEATURES + nm.MACRO_FEATURES]
    assert not np.allclose(a.to_numpy(), b.to_numpy(), equal_nan=True)


def test_targets_use_the_right_prices():
    stocks, macro = make_stocks(), make_macro()
    _, targ = nm.build_numeric(stocks, macro)
    s = stocks[stocks["ticker"] == "AAA"].reset_index(drop=True)
    f = s["adj_close"] / s["close"]
    adj_open = s["open"] * f
    t = targ[targ["ticker"] == "AAA"].reset_index(drop=True)
    i = 100
    assert t.loc[i, "ret_next"] == pytest.approx(s.loc[i + 1, "adj_close"] / s.loc[i, "adj_close"] - 1)
    assert t.loc[i, "ret_day"] == pytest.approx(s.loc[i, "adj_close"] / s.loc[i - 1, "adj_close"] - 1)
    assert t.loc[i, "ret_gap"] == pytest.approx(adj_open[i] / s.loc[i - 1, "adj_close"] - 1)
    assert t.loc[i, "ret_intraday"] == pytest.approx(s.loc[i, "adj_close"] / adj_open[i] - 1)
    assert np.isnan(t["ret_next"].iloc[-1])  # 最終日は翌日がない


def test_calendar_features_from_the_date_itself():
    feats, _ = nm.build_numeric(make_stocks(), make_macro())
    assert (feats["dow"] == pd.to_datetime(feats["date"]).dt.dayofweek).all()
    assert (feats["month"] == pd.to_datetime(feats["date"]).dt.month).all()


def test_safe_pct_negative_price_is_nan():
    s = pd.Series([10.0, 5.0, -3.0, 4.0, 6.0])  # WTI 2020-04 のような負の価格
    r = nm.safe_pct(s, 1)
    assert r.iloc[1] == pytest.approx(-0.5)
    assert np.isnan(r.iloc[2]) and np.isnan(r.iloc[3])  # 負の価格を含む区間は欠損
    assert r.iloc[4] == pytest.approx(0.5)
