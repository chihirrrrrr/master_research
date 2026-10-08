"""研究1(市場レベル)の、日単位の説明変数(数値)と目的変数を作る。

使い方(リポジトリのルートから):
    python -m src.features.market_level

入力: data/raw/prices_yfinance/<最新の取得日>/macro.parquet
出力: data/processed/market_features.parquet   説明変数18個(行 = S&P500の営業日 t。t の終値までの情報)
      data/processed/market_targets.parquet    目的変数(t+1 以降の価格だけを使う。特徴量とは別ファイル)
      data/processed/market_meta.json

説明変数(18個。docs/VARIABLES.md §6):
  市場・マクロ13 : numeric.py の macro_features(S&P500・VIX・金利・商品・SOX−NASDAQ)
  カレンダー2    : dow, month
  S&P500のOHLC由来3 : spx_range_1d(日中の値幅/終値)、spx_gap_1d(始値/前日終値−1)、spx_close_loc(終値の位置)
目的変数(単位 bp = 1/10000):
  spx_ret_fwd5 : S&P500指数(^GSPC の終値)の、t+1〜t+5 の日次リターンの平均(t は含まない)
  spx_ret_fwd1 : t+1 の日次リターン
"""
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.features import numeric as nm

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "processed"

OHLC_FEATURES = ["spx_range_1d", "spx_gap_1d", "spx_close_loc"]
MARKET_FEATURES = nm.MACRO_FEATURES + nm.CALENDAR_FEATURES + OHLC_FEATURES
TARGET_COLUMNS = ["spx_ret_fwd5", "spx_ret_fwd1"]
BP = 1e4


def spx_frame(macro: pd.DataFrame) -> pd.DataFrame:
    s = macro[macro["ticker"] == nm.SYMBOLS["spx"]].set_index("date").sort_index()
    return s[["open", "high", "low", "close"]]


def ohlc_features(spx: pd.DataFrame) -> pd.DataFrame:
    """S&P500のOHLCから作る3個。行 t は、t の始値・高値・安値・終値と前日終値だけから計算する。"""
    rng = spx["high"] - spx["low"]
    f = pd.DataFrame(index=spx.index)
    f["spx_range_1d"] = rng / spx["close"]
    f["spx_gap_1d"] = spx["open"] / spx["close"].shift(1) - 1
    f["spx_close_loc"] = ((spx["close"] - spx["low"]) / rng).where(rng > 0)
    return f


def targets(spx: pd.DataFrame) -> pd.DataFrame:
    """目的変数。t+1 以降の終値だけを使う。窓に欠損があれば欠損(末尾の5日)。"""
    r = spx["close"].pct_change()  # r_s = 前日終値 → 終値
    t = pd.DataFrame(index=spx.index)
    t["spx_ret_fwd5"] = sum(r.shift(-k) for k in range(1, 6)) / 5 * BP
    t["spx_ret_fwd1"] = r.shift(-1) * BP
    return t


def build_market_level(macro: pd.DataFrame):
    """(説明変数, 目的変数)を返す。どちらも 'date' 列 + 値の列。行 = S&P500の営業日。"""
    spx = spx_frame(macro)
    feats = nm.macro_features(macro).set_index("date").join(ohlc_features(spx), how="left")
    feats["dow"] = feats.index.dayofweek
    feats["month"] = feats.index.month
    feats = feats[MARKET_FEATURES]
    feats.index.name = "date"
    targ = targets(spx).reindex(feats.index)
    targ.index.name = "date"
    return feats.reset_index(), targ.reset_index()


def main() -> int:
    src = nm.latest_prices_dir()
    macro = pd.read_parquet(src / "macro.parquet")
    feats, targ = build_market_level(macro)
    OUT.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(OUT / "market_features.parquet", index=False)
    targ.to_parquet(OUT / "market_targets.parquet", index=False)
    complete = feats.dropna().merge(targ.dropna(), on="date")
    (OUT / "market_meta.json").write_text(json.dumps({
        "built_at": datetime.now().isoformat(timespec="seconds"), "prices_dir": str(src.relative_to(ROOT)),
        "rows": len(feats), "usable_rows": len(complete),
        "first_usable": str(complete["date"].min().date()), "last_usable": str(complete["date"].max().date()),
        "feature_columns": MARKET_FEATURES, "target_columns": TARGET_COLUMNS,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"入力: {src.relative_to(ROOT)}")
    print(f"説明変数: {len(feats):,} 行 × {len(MARKET_FEATURES)} 列 / 目的変数 {len(TARGET_COLUMNS)} 列")
    print(f"全部が揃う行: {len(complete):,} ({complete['date'].min().date()} 〜 {complete['date'].max().date()})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
