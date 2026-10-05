"""data/raw のデータ契約テスト。

生データが意図せず変わっていないか、加工が前提にしている性質(列・値域・時刻の保持)が成り立つかを確認する。
データが無い環境(別のマシンなど)ではスキップする。
"""
from pathlib import Path

import pandas as pd
import pytest

from src.data import verify_raw

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
NPE = RAW / "news_price_events" / "news_price_events.csv"
NYT = RAW / "nyt_headlines" / "dataset_NYTimes.csv"

pytestmark = pytest.mark.skipif(not NPE.exists(), reason="data/raw が無い環境")


@pytest.fixture(scope="module")
def npe() -> pd.DataFrame:
    return pd.read_csv(NPE, low_memory=False)


def test_checksums_match():
    assert verify_raw.main() == 0


def test_news_price_events_schema_and_size(npe):
    assert list(npe.columns) == [
        "stock", "time_window", "aggregated_text", "headline_count", "original_first_date",
        "clean_text", "trade_date", "Close", "forward_return", "market_impact",
    ]
    assert len(npe) == 219_971
    assert npe["stock"].nunique() == 223


def test_label_contract(npe):
    # forward_return は絶対値(方向の情報がない)。market_impact は3つのほぼ等しい分位。
    assert (npe["forward_return"] >= 0).all()
    assert set(npe["market_impact"].unique()) == {0, 1, 2}
    share = npe["market_impact"].value_counts(normalize=True)
    assert share.between(0.32, 0.35).all()
    assert npe["Close"].gt(0).all()


def test_trade_date_range(npe):
    d = pd.to_datetime(npe["trade_date"])
    assert d.min() == pd.Timestamp("2009-04-29")
    assert d.max() == pd.Timestamp("2020-06-12")


def test_original_time_is_kept(npe):
    # 寄り前ニュースを識別するには、元の時刻(分まで)が残っている必要がある。
    ts = pd.to_datetime(npe["original_first_date"].str[:19])
    assert ts.dt.minute.nunique() > 30
    tw = pd.to_datetime(npe["time_window"].str[:19])
    open_window = tw.dt.hour == 9
    td = pd.to_datetime(npe["trade_date"])
    pre_open = open_window & (ts < td + pd.Timedelta(hours=9, minutes=30))
    # 09時窓の約92%は、元の時刻が当日9:30より前(寄り前・時間外)
    assert 0.90 < pre_open.sum() / open_window.sum() < 0.95


@pytest.mark.skipif(not NYT.exists(), reason="NYT が無い環境")
def test_nyt_contract():
    nyt = pd.read_csv(NYT)
    assert list(nyt.columns) == ["Date", "Label", "News"]
    assert len(nyt) == 3_522
    assert set(nyt["Label"].unique()) == {0, 1}
