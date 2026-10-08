"""研究1の実験パイプラインのテスト(合成データ。XGBoostの代わりにテスト用のモデルで、仕組みを確かめる)。"""
import numpy as np
import pandas as pd
import pytest

from src.features import nyt_text as nt
from src.models import study1

SPLIT = {"embargo_days": 4, "development": {"first": 2013, "last": 2017}, "confirmation": {"first": 2018, "last": 2021},
         "inner_validation": {"share": 0.2, "order": "time"}}
MODEL = {"target": "spx_ret_fwd5", "model": {
    "n_estimators_max": 60, "early_stopping_rounds": 10, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8,
    "reg_lambda": 1, "grid": {"max_depth": [2], "min_child_weight": [20]}, "seeds": [0, 1]}}
COLS = ["x0", "x1", "x2"]


def synthetic(signal: float, seed: int = 0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2008-01-31", "2021-12-23")
    X = rng.normal(size=(len(dates), 3))
    y = signal * X[:, 0] + rng.normal(size=len(dates))
    return pd.DataFrame({"date": dates, "x0": X[:, 0], "x1": X[:, 1], "x2": X[:, 2], "spx_ret_fwd5": y}), COLS


def run(signal, periods="dev", final=False, kind="hgb"):
    return study1.run("numeric", periods, final=final, kind=kind, row_set="numeric", out_root=False,
                      dataset=synthetic(signal), split=SPLIT, model=MODEL)


def test_only_development_years_are_evaluated():
    r = run(1.0)
    assert [f["year"] for f in r["folds"]] == [2013, 2014, 2015, 2016, 2017]
    assert all(f["n_train"] < g["n_train"] for f, g in zip(r["folds"], r["folds"][1:]))  # 拡張窓


def test_confirmation_is_locked_without_final():
    with pytest.raises(SystemExit):
        run(1.0, periods="confirm", final=False)
    r = run(1.0, periods="confirm", final=True)  # --final があれば、最後の確認は実行できる
    assert [f["year"] for f in r["folds"]] == [2018, 2019, 2020, 2021]


def test_all_periods_also_needs_final_and_splits_dev_and_confirm():
    with pytest.raises(SystemExit):
        run(1.0, periods="all", final=False)
    r = run(1.0, periods="all", final=True)
    assert [f["year"] for f in r["folds"]] == list(range(2013, 2022))
    assert r["pooled_dev"]["n_test"] + r["pooled_confirm"]["n_test"] == r["pooled"]["n_test"]


def test_model_learns_a_real_signal_and_not_noise():
    assert run(1.0)["pooled"]["r2_os"] > 0.15          # 信号が強ければ、履歴平均より大きく良い
    assert run(0.0)["pooled"]["r2_os"] < 0.02          # 信号がなければ、履歴平均を超えない(過学習していない)


def test_nyt_aggregation_uses_only_past_days():
    dates = pd.bdate_range("2020-01-01", periods=12)
    s = pd.DataFrame({"date": dates, "pos": np.linspace(0.1, 0.6, 12), "neg": np.linspace(0.5, 0.1, 12), "neu": 0.2})
    a = nt.aggregate(s)
    assert a["nyt_net_ma5"].iloc[:4].isna().all() and not np.isnan(a["nyt_net_ma5"].iloc[4])
    assert a["nyt_net_ma5"].iloc[6] == pytest.approx((s["pos"] - s["neg"]).iloc[2:7].mean())  # 当日を含む直近5日
    s2 = s.copy()
    s2.loc[8:, ["pos", "neg"]] = 0.99  # 後の日を書き換えても、前の日の特徴は変わらない
    b = nt.aggregate(s2)
    pd.testing.assert_frame_equal(a.iloc[:8].reset_index(drop=True), b.iloc[:8].reset_index(drop=True))


def test_nyt_build_keeps_missing_days_missing():
    nyt = pd.DataFrame({"date": pd.bdate_range("2020-01-01", periods=8),
                        "news": ["a"] * 3 + [None] + ["b"] * 4})
    scorer = lambda texts: (np.tile([0.6, 0.3, 0.1], (len(texts), 1)), {"model": "dummy"})
    out, info = nt.build(nyt, scorer=scorer)
    assert np.isnan(out.loc[3, "nyt_pos"]) and out.loc[0, "nyt_pos"] == pytest.approx(0.6)
    assert out["nyt_net_ma5"].iloc[3:7].isna().all()  # 欠けた日を含む窓は欠損


def test_xgboost_smoke_if_available():
    try:
        from xgboost import XGBRegressor  # noqa: F401
        XGBRegressor(n_estimators=2).fit(np.zeros((6, 2)), np.zeros(6))
    except Exception:
        pytest.skip("XGBoostが動かない環境(libomp が未インストール)")
    assert run(1.0, kind="xgboost")["pooled"]["r2_os"] > 0.1
