"""NYT見出し(世界情勢)を、感情の数値に変換する(FinBERT)。研究1の「+NYT」の入力になる。

使い方(リポジトリのルートから。torch・transformers・FinBERTのモデルが必要):
    python -m src.features.nyt_text

入力: data/raw/nyt_headlines/dataset_NYTimes.csv(Date, News。営業日 3,522日)
出力: data/processed/nyt_features.parquet   行 = NYTの日付 D(= 予測の起点 t)。D の見出しだけから作る
      data/processed/nyt_meta.json          使ったモデル・版・欠損

特徴量(5列):
  nyt_pos / nyt_neg / nyt_neu : その日の見出し全体の、FinBERTの確率(正・負・中立)
  nyt_net_ma5                 : (正 − 負) の、直近5営業日(その日を含む)の平均
  nyt_neg_ma5                 : 負の確率の、直近5営業日の平均
過去の日付の分だけを使う(未来は見ない)。窓に欠損があれば欠損。
FinBERTは固定のまま(ファインチューニングしない)。入力は512トークンで切り詰める(1日約84語なので収まる)。
"""
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "nyt_headlines" / "dataset_NYTimes.csv"
OUT = ROOT / "data" / "processed"
MODEL = "ProsusAI/finbert"
NYT_COLUMNS = ["nyt_pos", "nyt_neg", "nyt_neu", "nyt_net_ma5", "nyt_neg_ma5"]


def aggregate(scores: pd.DataFrame) -> pd.DataFrame:
    """日ごとの確率(date, pos, neg, neu)から、5列の特徴量を作る。過去の日付だけを使う。"""
    s = scores.sort_values("date").reset_index(drop=True)
    net = s["pos"] - s["neg"]
    return pd.DataFrame({
        "date": s["date"], "nyt_pos": s["pos"], "nyt_neg": s["neg"], "nyt_neu": s["neu"],
        "nyt_net_ma5": net.rolling(5, min_periods=5).mean(),
        "nyt_neg_ma5": s["neg"].rolling(5, min_periods=5).mean(),
    })


def finbert_scores(texts, batch_size: int = 16) -> tuple:
    """FinBERTで、各テキストの (正, 負, 中立) の確率を返す。(確率の配列, モデルの情報)"""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL).eval()
    labels = {v.lower(): k for k, v in model.config.id2label.items()}  # 例: {'positive': 0, 'negative': 1, 'neutral': 2}
    order = [labels["positive"], labels["negative"], labels["neutral"]]
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            enc = tok(list(texts[i:i + batch_size]), padding=True, truncation=True, max_length=512, return_tensors="pt")
            out.append(torch.softmax(model(**enc).logits, dim=-1)[:, order].numpy())
    info = {"model": MODEL, "id2label": {str(k): v for k, v in model.config.id2label.items()}}
    return np.concatenate(out), info


def build(nyt: pd.DataFrame, scorer=finbert_scores):
    """nyt: date, news。見出しが欠けた日は欠損のまま。scorer(texts) -> (確率の配列, 情報)。"""
    has = nyt["news"].notna() & (nyt["news"].astype(str).str.strip() != "")
    probs = np.full((len(nyt), 3), np.nan)
    info = {}
    if has.any():
        p, info = scorer(nyt.loc[has, "news"].astype(str).tolist())
        probs[has.to_numpy()] = p
    scores = pd.DataFrame({"date": nyt["date"].to_numpy(), "pos": probs[:, 0], "neg": probs[:, 1], "neu": probs[:, 2]})
    return aggregate(scores), info


def main() -> int:
    nyt = pd.read_csv(RAW, parse_dates=["Date"]).rename(columns={"Date": "date", "News": "news"})
    feats, info = build(nyt[["date", "news"]])
    OUT.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(OUT / "nyt_features.parquet", index=False)
    (OUT / "nyt_meta.json").write_text(json.dumps({
        "built_at": datetime.now().isoformat(timespec="seconds"), "rows": len(feats), **info,
        "missing_text_days": int(nyt["news"].isna().sum()), "columns": NYT_COLUMNS,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"NYTの特徴量: {len(feats):,} 行 × {len(NYT_COLUMNS)} 列 / 見出しが欠けた日 {int(nyt['news'].isna().sum())} / モデル {info.get('model')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
