"""取得した価格(data/raw/prices_yfinance)と、作成した数値データ(data/processed)のデータ契約テスト。

取得・作成がまだの環境ではスキップする。
"""
from pathlib import Path

import pandas as pd
import pytest

from src.features import numeric as nm

ROOT = Path(__file__).resolve().parents[1]
PRICE_ROOT = ROOT / "data" / "raw" / "prices_yfinance"
PROCESSED = ROOT / "data" / "processed"
NPE = ROOT / "data" / "raw" / "news_price_events" / "news_price_events.csv"

dirs = sorted(p for p in PRICE_ROOT.glob("*") if (p / "stocks.parquet").exists()) if PRICE_ROOT.exists() else []
needs_prices = pytest.mark.skipif(not dirs, reason="価格が未取得(python -m src.data.fetch_prices)")
needs_processed = pytest.mark.skipif(
    not (PROCESSED / "numeric_features.parquet").exists(), reason="数値データが未作成(python -m src.features.numeric)")


@pytest.fixture(scope="module")
def stocks():
    return pd.read_parquet(dirs[-1] / "stocks.parquet")


@pytest.fixture(scope="module")
def macro():
    return pd.read_parquet(dirs[-1] / "macro.parquet")


@needs_prices
def test_stocks_contract(stocks):
    # 列の順序は契約ではない(保存されている順序は date, ticker, ...)。名前の集合で確認する
    assert set(stocks.columns) == {"ticker", "date", "open", "high", "low", "close", "adj_close", "volume",
                                   "dividends", "stock_splits"}
    assert stocks["ticker"].nunique() == 220
    assert not stocks.duplicated(["ticker", "date"]).any()
    assert (stocks[["open", "high", "low", "close", "adj_close"]] > 0).all().all()
    assert stocks["date"].min() == pd.Timestamp("2008-01-02") and stocks["date"].max() == pd.Timestamp("2021-12-31")
    # 取得できないのは、買収された3銘柄だけ
    npe_tickers = set(pd.read_csv(NPE, usecols=["stock"])["stock"]) if NPE.exists() else set()
    if npe_tickers:
        assert npe_tickers - set(stocks["ticker"]) == {"AET", "ESRX", "TWX"}


@needs_prices
def test_macro_contract(macro):
    need = set(nm.SYMBOLS.values()) | {"SPY"}
    assert need <= set(macro["ticker"])
    assert not macro.duplicated(["ticker", "date"]).any()
    n = macro.groupby("ticker").size()
    assert n[list(need)].min() > 3_400


@needs_processed
def test_processed_shapes_and_alignment():
    f = pd.read_parquet(PROCESSED / "numeric_features.parquet")
    t = pd.read_parquet(PROCESSED / "numeric_targets.parquet")
    assert list(f.columns) == ["ticker", "date"] + nm.FEATURE_COLUMNS
    assert list(t.columns) == ["ticker", "date"] + nm.TARGET_COLUMNS
    assert (f[["ticker", "date"]].values == t[["ticker", "date"]].values).all()
    assert not f[nm.FEATURE_COLUMNS].isin([float("inf"), float("-inf")]).any().any()


@needs_processed
@pytest.mark.skipif(not NPE.exists(), reason="news_price_events が無い環境")
def test_reproduces_authors_label():
    # 自分で作った翌日の絶対リターンが、news_price_events の forward_return と一致する(価格の整合の確認)
    t = pd.read_parquet(PROCESSED / "numeric_targets.parquet")
    n = pd.read_csv(NPE, usecols=["stock", "trade_date", "forward_return"], low_memory=False)
    n = n.drop_duplicates(["stock", "trade_date"])
    n["trade_date"] = pd.to_datetime(n["trade_date"])
    m = n.merge(t[["ticker", "date", "ret_next"]], left_on=["stock", "trade_date"], right_on=["ticker", "date"]).dropna()
    assert len(m) > 150_000
    assert ((m["ret_next"].abs() - m["forward_return"]).abs() < 1e-3).mean() > 0.999
