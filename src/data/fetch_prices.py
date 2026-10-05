"""yfinance から価格を取得して data/raw/prices_yfinance/<取得日>/ に保存する。

使い方(リポジトリのルートから):
    python -m src.data.fetch_prices

- open/high/low/close(分割調整済み・配当は未調整)と adj_close(配当・分割とも調整済み)、配当・分割をそのまま保存する。
- 調整済み価格は、配当・分割で後から値が変わる。**同じ場所を上書きしない**(出力先が既にあれば止まる)。
- 取得日・版・引数・取れなかった銘柄を FETCH_LOG.md に記録する(これはgit管理する)。
"""
import argparse
import hashlib
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml
import yfinance as yf

ROOT = Path(__file__).resolve().parents[2]
PRICE_ROOT = ROOT / "data" / "raw" / "prices_yfinance"
COLUMNS = {
    "Open": "open", "High": "high", "Low": "low", "Close": "close", "Adj Close": "adj_close",
    "Volume": "volume", "Dividends": "dividends", "Stock Splits": "stock_splits",
}


def md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def macro_tickers(cfg: dict) -> dict:
    """{ティッカー: (名前, グループ)}"""
    return {t: (name, group) for group, items in cfg["macro"].items() for t, name in items.items()}


def stock_tickers(cfg: dict) -> list:
    s = pd.read_csv(ROOT / cfg["stocks"]["source"], usecols=[cfg["stocks"]["column"]])
    return sorted(s[cfg["stocks"]["column"]].unique())


def _tidy(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    df = df.copy()
    for src in COLUMNS:
        if src not in df.columns:
            df[src] = float("nan")
    df = df[list(COLUMNS)].rename(columns=COLUMNS)
    df = df.dropna(subset=["close"])
    idx = pd.DatetimeIndex(df.index)
    df.index = (idx.tz_localize(None) if idx.tz is not None else idx).normalize()
    df.index.name = "date"
    df.insert(0, "ticker", ticker)
    return df.reset_index()


def download(tickers: list, start: str, end: str, batch_size: int, adjust: bool, actions: bool):
    """tickers をバッチで取得し、(長い形式のDataFrame, 取得できなかった銘柄) を返す。"""
    frames, missing = [], []
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        raw = None
        for attempt in range(3):
            try:
                raw = yf.download(batch, start=start, end=end, interval="1d", auto_adjust=adjust,
                                  actions=actions, group_by="ticker", threads=True, progress=False)
                break
            except Exception as e:  # ネットワーク・レート制限
                print(f"  retry {attempt + 1}: {str(e)[:80]}", file=sys.stderr)
                time.sleep(5 * (attempt + 1))
        for t in batch:
            try:
                df = raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw
                tidy = _tidy(df, t)
            except Exception:
                tidy = pd.DataFrame()
            if len(tidy):
                frames.append(tidy)
            else:
                missing.append(t)
        print(f"  {min(i + batch_size, len(tickers))}/{len(tickers)}", flush=True)
        time.sleep(1)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return out, missing


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "prices.yaml"))
    ap.add_argument("--out", default=None, help="出力先(既定: data/raw/prices_yfinance/<YYYYMMDD>)")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    now = datetime.now()
    out = Path(args.out) if args.out else PRICE_ROOT / now.strftime("%Y%m%d")
    if out.exists() and any(out.iterdir()):
        print(f"出力先が既にある(上書きしない): {out}", file=sys.stderr)
        return 1
    out.mkdir(parents=True, exist_ok=True)

    start, end = cfg["period"]["start"], cfg["period"]["end"]
    yfc = cfg["yfinance"]
    stocks = stock_tickers(cfg)
    macro = macro_tickers(cfg)
    print(f"株式 {len(stocks)} 銘柄 / 市場・マクロ {len(macro)} 系列 / {start} 〜 {end}(endは含まない)")

    s_df, s_missing = download(stocks, start, end, yfc["batch_size"], yfc["auto_adjust"], yfc["actions"])
    m_df, m_missing = download(list(macro), start, end, yfc["batch_size"], yfc["auto_adjust"], yfc["actions"])

    files = {"stocks.parquet": s_df, "macro.parquet": m_df}
    for name, df in files.items():
        df.to_parquet(out / name, index=False)

    # 記録
    lines = [
        "# 取得ログ(yfinance)", "",
        f"- 取得日時: {now:%Y-%m-%d %H:%M:%S}",
        f"- yfinance {yf.__version__} / pandas {pd.__version__} / Python {platform.python_version()}",
        f"- 期間: {start} 〜 {end}(endは含まない) / interval=1d / auto_adjust={yfc['auto_adjust']} / actions={yfc['actions']}",
        f"- 設定: `configs/prices.yaml`(md5 {md5(Path(args.config))[:12]})",
        "- 列: ticker, date, open, high, low, close(分割調整済み・配当は未調整), adj_close(配当・分割とも調整済み), volume, dividends, stock_splits",
        "", "## 株式(news_price_events の銘柄)",
        f"- 要求 {len(stocks)} 銘柄 / 取得 {s_df['ticker'].nunique()} 銘柄 / {len(s_df):,} 行",
        f"- **取得できなかった銘柄**: {', '.join(s_missing) if s_missing else 'なし'}",
        "", "## 市場・マクロ", f"- **取得できなかった系列**: {', '.join(m_missing) if m_missing else 'なし'}", "",
        "| ティッカー | 名前 | グループ | 開始 | 終了 | 行数 |", "|---|---|---|---|---|---|",
    ]
    for t, (name, group) in macro.items():
        g = m_df[m_df["ticker"] == t]
        if len(g):
            lines.append(f"| `{t}` | {name} | {group} | {g['date'].min().date()} | {g['date'].max().date()} | {len(g):,} |")
        else:
            lines.append(f"| `{t}` | {name} | {group} | - | - | 0 |")
    lines += ["", "## ファイル", "| ファイル | 行数 | md5 |", "|---|---|---|"]
    sums = []
    for name, df in files.items():
        h = md5(out / name)
        lines.append(f"| {name} | {len(df):,} | {h} |")
        sums.append(f"{h}  {name}")
    (out / "FETCH_LOG.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "CHECKSUMS.txt").write_text("\n".join(sums) + "\n", encoding="utf-8")
    for p in out.iterdir():
        os.chmod(p, 0o444)  # 読み取り専用(生データは書き換えない)
    print(f"保存: {out}")
    print(f"取得できなかった株式: {s_missing} / 系列: {m_missing}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
