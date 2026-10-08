"""研究1の日単位の説明変数・目的変数のテスト(合成データ)。リーク防止と、定義の確認。"""
import numpy as np
import pandas as pd
import pytest

from src.features import market_level as ml
from tests.test_numeric_features import DATES, make_macro, perturb


def test_feature_columns_are_18_without_fx():
    assert len(ml.MARKET_FEATURES) == 18
    assert not {"dxy_ret_5d", "usdjpy_ret_5d"} & set(ml.MARKET_FEATURES)
    assert not set(ml.MARKET_FEATURES) & set(ml.TARGET_COLUMNS)


def test_features_do_not_look_ahead():
    macro = make_macro()
    T, T1 = DATES[300], DATES[301]
    base, _ = ml.build_market_level(macro)
    pert, _ = ml.build_market_level(perturb(macro, T1))  # T+1 以降を書き換える
    a = base[base["date"] <= T].reset_index(drop=True)
    b = pert[pert["date"] <= T].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)  # 行 T までは変わらない(T の終値までの情報)


def test_features_use_the_same_day():
    macro = make_macro()
    T = DATES[300]
    base, _ = ml.build_market_level(macro)
    pert, _ = ml.build_market_level(perturb(macro, T))
    a = base.loc[base["date"] == T, ml.OHLC_FEATURES + ["spx_ret_1d"]].to_numpy(float)
    b = pert.loc[pert["date"] == T, ml.OHLC_FEATURES + ["spx_ret_1d"]].to_numpy(float)
    assert not np.allclose(a, b, equal_nan=True)  # T 自身を変えると、行 T は変わる


def test_ohlc_features_formulas():
    macro = make_macro()
    feats, _ = ml.build_market_level(macro)
    s = macro[macro["ticker"] == "^GSPC"].reset_index(drop=True)
    i = 50
    f = feats.reset_index(drop=True)
    assert f.loc[i, "spx_range_1d"] == pytest.approx((s.loc[i, "high"] - s.loc[i, "low"]) / s.loc[i, "close"])
    assert f.loc[i, "spx_gap_1d"] == pytest.approx(s.loc[i, "open"] / s.loc[i - 1, "close"] - 1)
    loc = (s.loc[i, "close"] - s.loc[i, "low"]) / (s.loc[i, "high"] - s.loc[i, "low"])
    assert f.loc[i, "spx_close_loc"] == pytest.approx(loc)


def test_targets_are_forward_means_in_bp():
    macro = make_macro()
    _, targ = ml.build_market_level(macro)
    s = macro[macro["ticker"] == "^GSPC"].reset_index(drop=True)
    r = s["close"].pct_change()
    t = targ.reset_index(drop=True)
    i = 100
    assert t.loc[i, "spx_ret_fwd5"] == pytest.approx(r.iloc[i + 1:i + 6].mean() * 1e4)  # t+1〜t+5。t は含まない
    assert t.loc[i, "spx_ret_fwd5"] != pytest.approx(r.iloc[i:i + 5].mean() * 1e4)
    assert t.loc[i, "spx_ret_fwd1"] == pytest.approx(r.iloc[i + 1] * 1e4)
    assert t["spx_ret_fwd5"].iloc[-5:].isna().all() and not np.isnan(t["spx_ret_fwd5"].iloc[-6])
    assert t["spx_ret_fwd1"].iloc[-1:].isna().all()
