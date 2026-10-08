"""walk-forward の分割と、評価指標のテスト。"""
import numpy as np
import pandas as pd
import pytest

from src.eval import metrics as mt
from src.eval import walk_forward as wf

DATES = pd.bdate_range("2008-01-31", "2021-12-23")


def test_folds_are_expanding_and_leave_an_embargo():
    folds = wf.make_folds(DATES, range(2013, 2018), embargo=4)
    assert [f["year"] for f in folds] == [2013, 2014, 2015, 2016, 2017]
    prev = 0
    for f in folds:
        tr, te = f["train"], f["test"]
        assert tr[0] == 0 and np.array_equal(tr, np.arange(len(tr)))  # 先頭から連続(拡張窓)
        assert te[0] - tr[-1] - 1 == 4                                  # 境界に4行の空き
        assert (DATES[te].year == f["year"]).all()                      # テストはその年だけ
        assert DATES[tr].max() < DATES[te].min()                        # 学習は、テストより前だけ
        assert len(tr) > prev                                           # 学習は年々増える
        prev = len(tr)


def test_no_test_row_is_ever_in_train():
    for f in wf.make_folds(DATES, range(2013, 2022), embargo=4):
        assert not set(f["train"]) & set(f["test"])


def test_inner_split_keeps_time_order_and_a_gap():
    tr, va = wf.inner_split(1000, 0.2, 4)
    assert len(va) == 200 and va[-1] == 999 and va[0] == 800
    assert va[0] - tr[-1] - 1 == 4 and tr[0] == 0


def test_unsorted_dates_are_rejected():
    with pytest.raises(ValueError):
        wf.make_folds(DATES[::-1], [2013], embargo=4)


def test_r2_os_reference_points():
    y = np.array([1.0, -1.0, 2.0, 0.5])
    assert mt.r2_os(y, np.full(4, y.mean()), y.mean()) == pytest.approx(0.0)
    assert mt.r2_os(y, y, y.mean()) == pytest.approx(1.0)
    assert mt.r2_os(y, -y, y.mean()) < 0  # 逆向きの予測は、履歴平均より悪い


def test_hit_rate_ignores_zero_realizations():
    assert mt.hit_rate([1, -1, 0, 2], [1, 1, 5, 2]) == pytest.approx(2 / 3)


def test_clark_west_detects_a_real_improvement_and_not_noise():
    rng = np.random.default_rng(0)
    n = 600
    x = rng.normal(size=n)
    y = 0.5 * x + rng.normal(size=n)
    bench = np.zeros(n)
    good = 0.5 * x                       # 本当に情報がある予測
    noise = rng.normal(scale=0.3, size=n)  # 無関係な予測
    assert mt.clark_west(y, bench, good)["p_one_sided"] < 0.01
    assert mt.clark_west(y, bench, noise)["p_one_sided"] > 0.05


def test_direction_metrics_flags_a_constant_up_predictor():
    y = np.array([1.0, 2.0, -1.0, 3.0, -2.0, 1.5])
    d = mt.direction_metrics(y, np.full(6, 5.0))  # いつも上昇と予測
    assert d["accuracy"] == pytest.approx(d["always_up_accuracy"]) and d["pred_up_share"] == 1.0
    assert d["balanced_accuracy"] == pytest.approx(0.5) and d["auc"] == pytest.approx(0.5)
    assert d["pred_std"] == 0.0 < d["y_std"]
    good = mt.direction_metrics(y, y * 2)  # 実現どおりの符号・順序
    assert good["accuracy"] == 1.0 and good["auc"] == 1.0
