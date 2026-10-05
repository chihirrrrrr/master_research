"""数値の特徴量(A0)と、目的変数の候補を作る。

使い方(リポジトリのルートから):
    python -m src.features.numeric

入力: data/raw/prices_yfinance/<最新の取得日>/{stocks,macro}.parquet
出力: data/processed/numeric_features.parquet   特徴量(行 = 銘柄 × 取引日 t)
      data/processed/numeric_targets.parquet    目的変数の候補(特徴量とは別ファイル。リーク防止)
      data/processed/numeric_meta.json          どの取得データから作ったか

**時刻のルール**: 取引日 t の寄り前に予測する。特徴量はすべて「t-1 の終値までの情報」で、
行 t の値は 1 営業日前に計算した値をずらしたもの。曜日・月だけは t の日付そのもの(事前に分かる)。
目的変数は t 以降の価格を使う(特徴量には入れない)。

特徴量(27個。RESEARCH_PLAN.md §3-2):
  銘柄(own_)  : ret_1d, ret_5d, ret_20d, rv_5d, rv_20d, range_1d, gap_1d, volume_ratio, dist_ma50, dist_52wk_high
  市場         : spx_ret_1d, spx_ret_5d, spx_rv_20d, vix_level, vix_chg_1d, vix_term
  金利         : tnx_level, tnx_chg_5d, term_spread
  為替         : dxy_ret_5d, usdjpy_ret_5d
  商品         : wti_ret_5d, copper_ret_5d, gold_ret_5d
  テック       : sox_minus_nasdaq_5d
  カレンダー   : dow, month
目的変数(§4-2): ret_gap(前日終値→始値)、ret_intraday(始値→終値)、ret_day(前日終値→終値)、ret_next(終値→翌日終値)と、
                SPYの同じ4つ(spy_ret_*。超過リターン用)。値はすべて配当・分割調整済み。
"""
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PRICE_ROOT = ROOT / "data" / "raw" / "prices_yfinance"
OUT = ROOT / "data" / "processed"

STOCK_FEATURES = ["own_ret_1d", "own_ret_5d", "own_ret_20d", "own_rv_5d", "own_rv_20d", "own_range_1d",
                  "own_gap_1d", "own_volume_ratio", "own_dist_ma50", "own_dist_52wk_high"]
MARKET_FEATURES = ["spx_ret_1d", "spx_ret_5d", "spx_rv_20d", "vix_level", "vix_chg_1d", "vix_term"]
RATE_FEATURES = ["tnx_level", "tnx_chg_5d", "term_spread"]
FX_FEATURES = ["dxy_ret_5d", "usdjpy_ret_5d"]
COMMODITY_FEATURES = ["wti_ret_5d", "copper_ret_5d", "gold_ret_5d"]
TECH_FEATURES = ["sox_minus_nasdaq_5d"]
CALENDAR_FEATURES = ["dow", "month"]
MACRO_FEATURES = MARKET_FEATURES + RATE_FEATURES + FX_FEATURES + COMMODITY_FEATURES + TECH_FEATURES
FEATURE_COLUMNS = STOCK_FEATURES + MACRO_FEATURES + CALENDAR_FEATURES
TARGET_COLUMNS = ["ret_gap", "ret_intraday", "ret_day", "ret_next",
                  "spy_ret_gap", "spy_ret_intraday", "spy_ret_day", "spy_ret_next"]

SYMBOLS = {"spx": "^GSPC", "vix": "^VIX", "vix3m": "^VIX3M", "tnx": "^TNX", "irx": "^IRX", "dxy": "DX-Y.NYB",
           "jpy": "JPY=X", "wti": "CL=F", "copper": "HG=F", "gold": "GC=F", "sox": "^SOX", "nasdaq": "^IXIC"}


def safe_pct(s: pd.Series, n: int = 1) -> pd.Series:
    """n日変化率。価格が0以下(2020年4月のWTIなど)の区間は欠損にする。"""
    prev = s.shift(n)
    return (s / prev - 1).where((s > 0) & (prev > 0))


def _clean(x):
    return x.replace([np.inf, -np.inf], np.nan)


def _adjust(df: pd.DataFrame) -> pd.DataFrame:
    """調整係数(adj_close/close)で open/high/low も配当・分割調整する。"""
    f = df["adj_close"] / df["close"]
    out = df.copy()
    for c in ("open", "high", "low"):
        out["adj_" + c] = df[c] * f
    return out


def stock_features(stocks: pd.DataFrame) -> pd.DataFrame:
    """銘柄ごとの特徴量。行 t の値は、1行前(前営業日)の終値までから計算した値。"""
    s = _adjust(stocks.sort_values(["ticker", "date"]).reset_index(drop=True))
    g = s.groupby("ticker", sort=False)
    lr = np.log(s["adj_close"]).groupby(s["ticker"]).diff()
    vol_ma20 = g["volume"].transform(lambda x: x.shift(1).rolling(20).mean())
    f = pd.DataFrame({"ticker": s["ticker"], "date": s["date"]})
    f["own_ret_1d"] = g["adj_close"].pct_change()
    f["own_ret_5d"] = g["adj_close"].pct_change(5)
    f["own_ret_20d"] = g["adj_close"].pct_change(20)
    f["own_rv_5d"] = lr.groupby(s["ticker"]).transform(lambda x: x.rolling(5).std())
    f["own_rv_20d"] = lr.groupby(s["ticker"]).transform(lambda x: x.rolling(20).std())
    f["own_range_1d"] = (s["adj_high"] - s["adj_low"]) / s["adj_close"]
    f["own_gap_1d"] = s["adj_open"] / g["adj_close"].shift(1) - 1
    f["own_volume_ratio"] = _clean(s["volume"] / vol_ma20)
    f["own_dist_ma50"] = s["adj_close"] / g["adj_close"].transform(lambda x: x.rolling(50).mean()) - 1
    f["own_dist_52wk_high"] = s["adj_close"] / g["adj_close"].transform(lambda x: x.rolling(252).max()) - 1
    f[STOCK_FEATURES] = _clean(f[STOCK_FEATURES])
    # 行 t には、1行前(t-1)で計算した値を入れる
    f[STOCK_FEATURES] = f.groupby("ticker", sort=False)[STOCK_FEATURES].shift(1)
    return f


def macro_features(macro: pd.DataFrame) -> pd.DataFrame:
    """市場・マクロの特徴量(日付ごと)。S&P500の営業日を基準にし、1営業日ずらす。"""
    wide = macro.pivot(index="date", columns="ticker", values="close").sort_index()
    cal = wide[SYMBOLS["spx"]].dropna().index
    w = wide.reindex(cal).ffill(limit=5)
    c = {k: w[t] for k, t in SYMBOLS.items()}
    lr_spx = np.log(c["spx"]).diff()
    f = pd.DataFrame(index=cal)
    f["spx_ret_1d"] = safe_pct(c["spx"], 1)
    f["spx_ret_5d"] = safe_pct(c["spx"], 5)
    f["spx_rv_20d"] = lr_spx.rolling(20).std()
    f["vix_level"] = c["vix"]
    f["vix_chg_1d"] = np.log(c["vix"]).diff()
    f["vix_term"] = c["vix3m"] / c["vix"]
    f["tnx_level"] = c["tnx"]
    f["tnx_chg_5d"] = c["tnx"].diff(5)
    f["term_spread"] = c["tnx"] - c["irx"]
    f["dxy_ret_5d"] = safe_pct(c["dxy"], 5)
    f["usdjpy_ret_5d"] = safe_pct(c["jpy"], 5)
    f["wti_ret_5d"] = safe_pct(c["wti"], 5)
    f["copper_ret_5d"] = safe_pct(c["copper"], 5)
    f["gold_ret_5d"] = safe_pct(c["gold"], 5)
    f["sox_minus_nasdaq_5d"] = safe_pct(c["sox"], 5) - safe_pct(c["nasdaq"], 5)
    f = _clean(f[MACRO_FEATURES]).shift(1)  # 行 t は t-1 の終値までの値
    f.index.name = "date"
    return f.reset_index()


def targets(stocks: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    """目的変数の候補(t 以降の価格を使う。特徴量には入れない)。"""
    def returns(df: pd.DataFrame) -> pd.DataFrame:
        s = _adjust(df.sort_values(["ticker", "date"]).reset_index(drop=True))
        g = s.groupby("ticker", sort=False)["adj_close"]
        out = pd.DataFrame({"ticker": s["ticker"], "date": s["date"]})
        out["ret_gap"] = s["adj_open"] / g.shift(1) - 1
        out["ret_intraday"] = s["adj_close"] / s["adj_open"] - 1
        out["ret_day"] = g.pct_change()
        out["ret_next"] = g.shift(-1) / s["adj_close"] - 1
        return out
    t = returns(stocks)
    spy = returns(macro[macro["ticker"] == "SPY"]).drop(columns="ticker")
    spy = spy.rename(columns={c: "spy_" + c for c in spy.columns if c != "date"})
    t = t.merge(spy, on="date", how="left")
    t[TARGET_COLUMNS] = _clean(t[TARGET_COLUMNS])
    return t[["ticker", "date"] + TARGET_COLUMNS]


def build_numeric(stocks: pd.DataFrame, macro: pd.DataFrame):
    """(特徴量, 目的変数) を返す。どちらも行 = 銘柄 × 取引日 t。"""
    feats = stock_features(stocks).merge(macro_features(macro), on="date", how="left")
    d = pd.to_datetime(feats["date"])
    feats["dow"] = d.dt.dayofweek
    feats["month"] = d.dt.month
    feats = feats[["ticker", "date"] + FEATURE_COLUMNS].sort_values(["ticker", "date"]).reset_index(drop=True)
    return feats, targets(stocks, macro).sort_values(["ticker", "date"]).reset_index(drop=True)


def latest_prices_dir() -> Path:
    dirs = sorted(p for p in PRICE_ROOT.iterdir() if p.is_dir() and (p / "stocks.parquet").exists())
    if not dirs:
        raise FileNotFoundError("data/raw/prices_yfinance/ に取得済みの価格がない(python -m src.data.fetch_prices)")
    return dirs[-1]


def main() -> int:
    src = latest_prices_dir()
    stocks = pd.read_parquet(src / "stocks.parquet")
    macro = pd.read_parquet(src / "macro.parquet")
    feats, targ = build_numeric(stocks, macro)
    OUT.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(OUT / "numeric_features.parquet", index=False)
    targ.to_parquet(OUT / "numeric_targets.parquet", index=False)
    (OUT / "numeric_meta.json").write_text(json.dumps({
        "built_at": datetime.now().isoformat(timespec="seconds"), "prices_dir": str(src.relative_to(ROOT)),
        "rows": len(feats), "tickers": int(feats["ticker"].nunique()),
        "feature_columns": FEATURE_COLUMNS, "target_columns": TARGET_COLUMNS,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"入力: {src.relative_to(ROOT)}")
    print(f"特徴量: {len(feats):,} 行 × {len(FEATURE_COLUMNS)} 列 / {feats['ticker'].nunique()} 銘柄 / {feats['date'].min().date()} 〜 {feats['date'].max().date()}")
    print(f"目的変数: {len(targ):,} 行 × {len(TARGET_COLUMNS)} 列")
    return 0


if __name__ == "__main__":
    sys.exit(main())
