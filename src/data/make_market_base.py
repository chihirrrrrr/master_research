"""研究1(市場レベル)の土台になる表を作る: S&P500のOHLC + 金利。期間はNYTの期間。

使い方(リポジトリのルートから):
    python -m src.data.make_market_base

入力: data/raw/prices_yfinance/<最新の取得日>/macro.parquet
      data/raw/nyt_headlines/20XX.csv(期間を決めるためだけに使う)
出力: data/interim/market_base.parquet / market_base.csv   行 = S&P500の営業日(NYTの期間内)
      data/interim/market_base_meta.json                    どの取得データから作ったか・欠けた日

**加工はしない。** 値はyfinanceで取得したそのまま。S&P500の営業日に並べるだけで、欠けた日は欠損のまま(補完しない)。
列:
  spx_open / spx_high / spx_low / spx_close : S&P500(^GSPC)の始値・高値・安値・終値
  irx_3m / fvx_5y / tnx_10y / tyx_30y       : 米国債の利回り(終値、%)。^IRX(3か月)、^FVX(5年)、^TNX(10年)、^TYX(30年)

**為替は含めない**(2026-10-08の決定)。2008年のドル円・ユーロドルが、取得元(Yahoo)で日付が崩れていたため(EXCLUDED)。

注意: 金利の「日付 t の終値」は、米国株の引け(16:00 ET)と同じ時刻とは限らない。
      米国債の利回りは債券市場が休みで欠ける日がある(株式市場だけ開いている日)。
"""
import glob
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PRICE_ROOT = ROOT / "data" / "raw" / "prices_yfinance"
NYT_DIR = ROOT / "data" / "raw" / "nyt_headlines"
OUT = ROOT / "data" / "interim"

SPX = "^GSPC"
CLOSE_ONLY = {"^IRX": "irx_3m", "^FVX": "fvx_5y", "^TNX": "tnx_10y", "^TYX": "tyx_30y"}
SPX_COLUMNS = ["spx_open", "spx_high", "spx_low", "spx_close"]
EXCLUDED = {
    "為替(ドル指数 dxy、ドル円 usdjpy、ユーロドル eurusd)":
        "含めない(2026-10-08の決定)。2008年のドル円・ユーロドルに、取得元(Yahoo)の日付の崩れがあるため"
        "(月日が同じ数字の日に別の日の値が入り、8月の大半と4/1・5/1・7/1が欠ける。日と月が入れ替わった行とみられる)。"
        "2009年以降は問題なし。ドル指数には異常がない。必要なら取得済みのデータ(macro.parquet)から追加できる。",
}
COLUMNS = ["date"] + SPX_COLUMNS + list(CLOSE_ONLY.values())


def nyt_period() -> tuple:
    """NYTの期間(年別ファイルの最初と最後の日付)。"""
    dates = pd.concat([pd.read_csv(f, usecols=["Date"], parse_dates=["Date"])
                       for f in sorted(glob.glob(str(NYT_DIR / "20*.csv")))])["Date"]
    return dates.min(), dates.max()


def build_market_base(macro: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """S&P500の営業日(start〜end)に、金利の終値を並べる。補完はしない。"""
    spx = macro[macro["ticker"] == SPX].set_index("date").sort_index().loc[start:end]
    out = pd.DataFrame(index=spx.index)
    for col in ("open", "high", "low", "close"):
        out["spx_" + col] = spx[col]
    close = macro[macro["ticker"].isin(CLOSE_ONLY)].pivot(index="date", columns="ticker", values="close")
    for ticker, name in CLOSE_ONLY.items():
        out[name] = close[ticker].reindex(out.index) if ticker in close else float("nan")
    out.index.name = "date"
    return out.reset_index()[COLUMNS]


def spike_dates(s: pd.Series, thr: float = 0.03) -> list:
    """前後の値はほぼ同じなのに、その日だけ前後の平均から thr 以上ずれている日(飛び値)の日付。欠損の日は飛ばして比べる。"""
    x = s.dropna()
    prev, nxt = x.shift(1), x.shift(-1)
    dev = x / ((prev + nxt) / 2) - 1
    gap = (prev / nxt - 1).abs()
    return [str(d.date()) for d in x.index[(dev.abs() > thr) & (gap < thr / 2)]]


def latest_prices_dir() -> Path:
    dirs = sorted(p for p in PRICE_ROOT.iterdir() if p.is_dir() and (p / "macro.parquet").exists())
    if not dirs:
        raise FileNotFoundError("data/raw/prices_yfinance/ に取得済みの価格がない(python -m src.data.fetch_prices)")
    return dirs[-1]


def main() -> int:
    src = latest_prices_dir()
    start, end = nyt_period()
    base = build_market_base(pd.read_parquet(src / "macro.parquet"), start, end)
    OUT.mkdir(parents=True, exist_ok=True)
    base.to_parquet(OUT / "market_base.parquet", index=False)
    base.to_csv(OUT / "market_base.csv", index=False, date_format="%Y-%m-%d")
    missing = {c: [str(d.date()) for d in base.loc[base[c].isna(), "date"]] for c in COLUMNS[1:] if base[c].isna().any()}
    (OUT / "market_base_meta.json").write_text(json.dumps({
        "built_at": datetime.now().isoformat(timespec="seconds"), "prices_dir": str(src.relative_to(ROOT)),
        "nyt_period": [str(start.date()), str(end.date())], "rows": len(base),
        "first_date": str(base["date"].min().date()), "last_date": str(base["date"].max().date()),
        "columns": COLUMNS, "missing_dates": missing,
        "excluded": EXCLUDED,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"入力: {src.relative_to(ROOT)} / NYTの期間: {start.date()} 〜 {end.date()}")
    print(f"出力: data/interim/market_base.parquet(と .csv) {len(base):,} 行 × {len(COLUMNS)} 列 / {base['date'].min().date()} 〜 {base['date'].max().date()}")
    print("欠損:", {c: len(v) for c, v in missing.items()} or "なし")
    return 0


if __name__ == "__main__":
    sys.exit(main())
