"""分析スクリプト(src/analysis/)のテスト。小さい合成データで、計算の定義を確かめる。"""
import json

import numpy as np
import pandas as pd
import pytest

from src.analysis import feature_distribution as fd
from src.analysis import label_balance as lb
from src.analysis import summarize_runs as sr


def test_label_balance_counts_up_and_down_by_year():
    df = pd.DataFrame({"date": pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06", "2021-01-04"]),
                       "y": [1.0, -1.0, 2.0, -3.0]})
    b = lb.balance(df, "y", other="none").set_index("year")
    assert b.loc[2020, "up"] == 2 and b.loc[2020, "down"] == 1 and b.loc[2020, "ratio"] == pytest.approx(2.0)
    assert b.loc["全体", "n"] == 4 and b.loc["全体", "up_share"] == pytest.approx(0.5)


def test_feature_distribution_flags_a_level_shift_between_periods():
    dates = pd.bdate_range("2010-01-01", "2016-12-30")
    level = np.where(dates.year < 2013, 3.0, 1.0) + np.random.default_rng(0).normal(0, 0.1, len(dates))
    df = pd.DataFrame({"date": dates, "lvl": level, "noise": np.random.default_rng(1).normal(size=len(dates))})
    d = fd.describe(df, ["lvl", "noise"])
    assert d.loc["lvl", "shift"] < -1 and abs(d.loc["noise", "shift"]) < 0.2


def test_summarize_runs_per_year_matches_a_hand_calculation(tmp_path):
    p = pd.DataFrame({"date": pd.bdate_range("2020-01-01", periods=6), "year": 2020,
                      "y": [1.0, -1.0, 2.0, -2.0, 1.0, 3.0], "pred": [1.0, -1.0, 2.0, -2.0, 1.0, 3.0], "hist": 0.0})
    (tmp_path / "artifacts").mkdir()
    p.to_parquet(tmp_path / "artifacts" / "predictions.parquet")
    (tmp_path / "metrics.json").write_text(json.dumps({"folds": [{"year": 2020, "n_train": 10, "n_test": 6}]}), encoding="utf-8")
    r = sr.per_year(tmp_path).iloc[0]
    assert r["r2"] == pytest.approx(1.0) and r["auc"] == pytest.approx(1.0) and r["hit"] == pytest.approx(1.0)
    assert r["always_up"] == pytest.approx(4 / 6)


def test_compare_horizons_builds_a_table_from_runs(tmp_path):
    from src.analysis import compare_horizons as ch
    rng = np.random.default_rng(0)
    for name in ("20261119_study1_numeric_reg1_cum10_all", "20261119_study1_numeric_nyt_reg1_cum10_all"):
        d = tmp_path / name
        (d / "artifacts").mkdir(parents=True)
        yrs = np.repeat([2017, 2018], 50)
        y = rng.normal(size=100)
        pd.DataFrame({"date": pd.bdate_range("2017-01-02", periods=100), "year": yrs, "y": y,
                      "pred": rng.normal(size=100) * 0.1, "hist": 0.0}).to_parquet(d / "artifacts" / "predictions.parquet")
        (d / "metrics.json").write_text(json.dumps({
            "n_rows": 100, "folds": [{"year": 2017, "n_train": 10, "n_test": 50}, {"year": 2018, "n_train": 60, "n_test": 50}],
            "pooled": {"clark_west_vs_hist": {"p_one_sided": 0.5}, "direction": {"balanced_accuracy": 0.5}},
            "shap_mean_abs_test": {"tnx_level": 1.0, "vix_level": 0.5, "x": 0.1, "y": 0.0}}), encoding="utf-8")
    res = ch.build(tmp_path)
    assert (10, "A0m", "reg1") in res and (5, "A0m", "reg1") not in res
    assert res[(10, "A0m", "reg1")]["shap_top"] == ["tnx_level", "vix_level", "x"]
    assert "10日" in ch.to_markdown(res) and "A1m − A0m" in ch.to_markdown(res)


def test_target_relation_cumulative_is_about_h_times_the_mean():
    from src.analysis import target_relation as tr
    close = pd.Series(100 * np.cumprod(1 + np.random.default_rng(0).normal(0.0003, 0.01, 400)), index=pd.bdate_range("2020-01-01", periods=400))
    rel = tr.relation(close, horizons=(5,)).iloc[0]
    assert rel["相関"] > 0.99 and 4.8 < rel["比(累積/平均、中央値)"] < 5.2 and rel["符号の一致"] > 0.95
