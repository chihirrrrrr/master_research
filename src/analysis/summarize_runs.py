"""実験(results/runs/<名前>/)の結果を、年別の表と、期間別の比較表にまとめる。分析メモ(docs/analysis/)の表の元。

使い方(リポジトリのルートから):
    python -m src.analysis.summarize_runs results/runs/20261008_study1_numeric_reg1_all [他のrun ...] [--out results/analysis/xxx.md]

入力: run のフォルダ(metrics.json と artifacts/predictions.parquet)
- 年別 : 年, 学習行, テスト行, R², 相関, 的中率(いつも上昇), AUC, 上昇と予測した割合, 予測sd/実現sd
- 期間別: 開発(〜2017)と確認(2018〜)の R², AUC(予測をつないだ値。年ごとの再学習の評価ではなく、参考)
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.eval import metrics as mt


def _auc(y, pred) -> float:
    yc = np.asarray(y) > 0
    return float(roc_auc_score(yc, pred)) if 0 < yc.mean() < 1 else float("nan")


def per_year(run_dir) -> pd.DataFrame:
    """年ごとの結果(その年のテスト行だけで計算)。物差しは、その回の学習データの平均(hist)。"""
    run_dir = Path(run_dir)
    m = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    p = pd.read_parquet(run_dir / "artifacts" / "predictions.parquet")
    rows = []
    for f in m["folds"]:
        q = p[p["year"] == f["year"]]
        y, pr = q["y"].to_numpy(), q["pred"].to_numpy()
        rows.append({"year": f["year"], "n_train": f["n_train"], "n_test": f["n_test"],
                     "r2": mt.r2_os(y, pr, q["hist"].to_numpy()), "corr": mt.corr(y, pr),
                     "hit": float(np.mean((pr > 0) == (y > 0))), "always_up": float(np.mean(y > 0)),
                     "auc": _auc(y, pr), "up_share": float(np.mean(pr > 0)),
                     "pred_sd": float(np.std(pr)), "y_sd": float(np.std(y))})
    return pd.DataFrame(rows)


def by_period(run_dir, split_year: int = 2018) -> dict:
    """開発(split_year より前)と確認(split_year 以降)に分けて、予測をつないだ R² と AUC を出す。"""
    p = pd.read_parquet(Path(run_dir) / "artifacts" / "predictions.parquet")
    out = {}
    for name, q in (("dev", p[p["year"] < split_year]), ("confirm", p[p["year"] >= split_year])):
        if len(q):
            out[name] = {"n": int(len(q)), "r2": mt.r2_os(q["y"], q["pred"], q["hist"]), "auc": _auc(q["y"], q["pred"])}
    return out


def to_markdown(df: pd.DataFrame) -> str:
    head = "| 年 | 学習行 | テスト行 | R² | 相関 | 的中率(いつも上昇) | AUC | 上昇と予測した割合 | 予測sd / 実現sd(bp) |\n|---|---|---|---|---|---|---|---|---|\n"
    body = "".join(
        f"| {r.year} | {r.n_train:,} | {r.n_test} | {r.r2:+.4f} | {r.corr:+.3f} | {r.hit:.3f} ({r.always_up:.3f}) | "
        f"{r.auc:.3f} | {r.up_share:.0%} | {r.pred_sd:.1f} / {r.y_sd:.1f} |\n" for r in df.itertuples())
    return head + body


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    text = ""
    for r in a.runs:
        text += f"## {Path(r).name}\n\n{to_markdown(per_year(r))}\n"
        for k, v in by_period(r).items():
            text += f"- {k}: {v['n']}日 R² {v['r2']:+.4f} / AUC {v['auc']:.3f}\n"
        text += "\n"
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text, encoding="utf-8")
        print("保存:", a.out)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
