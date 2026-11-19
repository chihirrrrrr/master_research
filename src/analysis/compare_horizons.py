"""累積リターンの期間(5・10・20日)ごとの結果を、同じ表にして比べる。

使い方(リポジトリのルートから):
    python -m src.analysis.compare_horizons [--out results/analysis/horizon_compare.md]

入力: results/runs/ の、各期間 × (数値のみ/+NYT) × (reg1/reg2) の run(`--periods all` の実行)
- R² は予測をつないだ値(開発 2013〜2017 / 確認 2018〜2021)。物差しは、その回の学習データの平均。
- 正解率は「いつも上昇」と比べる。予測sd は実現値の標準偏差と比べ、小さいほど「ほぼ定数」の予測。
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis import summarize_runs as sr
from src.analysis._md import table
from src.eval import metrics as mt

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "results" / "runs"
# 期間 → run 名の接尾辞(5日は 11-05 に作った名前)
SUFFIX = {5: "cum_all", 10: "cum10_all", 20: "cum20_all"}
INPUTS = {"A0m": "numeric", "A1m": "numeric_nyt"}


def find_run(horizon: int, inp: str, setting: str, runs_dir=RUNS):
    hits = sorted(Path(runs_dir).glob(f"*_study1_{INPUTS[inp]}_{setting}_{SUFFIX[horizon]}"))
    return hits[-1] if hits else None


def summarize(run_dir) -> dict:
    run_dir = Path(run_dir)
    m = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    p = pd.read_parquet(run_dir / "artifacts" / "predictions.parquet")
    y, pr = p["y"].to_numpy(), p["pred"].to_numpy()
    per = sr.by_period(run_dir)
    shap = m.get("shap_mean_abs_test") or {}
    d = m["pooled"]["direction"]
    return {"rows": m["n_rows"], "n_test": len(p),
            "r2_dev": per["dev"]["r2"], "auc_dev": per["dev"]["auc"],
            "r2_conf": per["confirm"]["r2"], "auc_conf": per["confirm"]["auc"],
            "acc": float(np.mean((pr > 0) == (y > 0))), "always_up": float(np.mean(y > 0)),
            "auc": sr._auc(y, pr), "up_share": float(np.mean(pr > 0)),
            "pred_sd": float(np.std(pr)), "y_sd": float(np.std(y)),
            "cw_p": m["pooled"]["clark_west_vs_hist"]["p_one_sided"], "bal_acc": d["balanced_accuracy"],
            "shap_top": list(shap)[:3], "shap": shap, "per_year": sr.per_year(run_dir)}


def build(runs_dir=RUNS) -> dict:
    out = {}
    for h in SUFFIX:
        for inp in INPUTS:
            for st in ("reg1", "reg2"):
                r = find_run(h, inp, st, runs_dir)
                if r is not None:
                    out[(h, inp, st)] = summarize(r)
    return out


def to_markdown(res: dict) -> str:
    rows = [{"期間": f"{h}日", "入力": inp, "設定": st, "行": v["rows"],
             "開発R²": f"{v['r2_dev']:+.4f}", "確認R²": f"{v['r2_conf']:+.4f}",
             "AUC(開発/確認)": f"{v['auc_dev']:.3f} / {v['auc_conf']:.3f}",
             "正解率(いつも上昇)": f"{v['acc']:.3f} ({v['always_up']:.3f})", "上昇と予測": f"{v['up_share']:.0%}",
             "予測sd/実現sd": f"{v['pred_sd']:.0f} / {v['y_sd']:.0f}", "SHAP上位3": ", ".join(v["shap_top"])}
            for (h, inp, st), v in sorted(res.items())]
    t1 = table(pd.DataFrame(rows), index=False)
    drows = []
    for h in SUFFIX:
        for st in ("reg1", "reg2"):
            a, b = res.get((h, "A0m", st)), res.get((h, "A1m", st))
            if a and b:
                diff = b["per_year"]["r2"].to_numpy() - a["per_year"]["r2"].to_numpy()
                drows.append({"期間": f"{h}日", "設定": st, "開発ΔR²": f"{b['r2_dev'] - a['r2_dev']:+.4f}",
                              "確認ΔR²": f"{b['r2_conf'] - a['r2_conf']:+.4f}",
                              "改善した年": f"{int((diff > 0).sum())} / {len(diff)}", "ΔAUC(開発)": f"{b['auc_dev'] - a['auc_dev']:+.3f}"})
    return "## 期間別の結果\n\n" + t1 + "\n## +NYT − 数値のみ(A1m − A0m)\n\n" + table(pd.DataFrame(drows), index=False)


def yearly_table(res: dict, inp: str = "A0m", setting: str = "reg1") -> str:
    """期間ごとの、年別の R² と AUC(指定した入力・設定)。"""
    rows = []
    for h in SUFFIX:
        v = res.get((h, inp, setting))
        if v is None:
            continue
        py = v["per_year"]
        rows.append({"期間": f"{h}日", "指標": "R²", **{str(int(y)): f"{r:+.3f}" for y, r in zip(py["year"], py["r2"])}})
        rows.append({"期間": f"{h}日", "指標": "AUC", **{str(int(y)): f"{r:.2f}" for y, r in zip(py["year"], py["auc"])}})
    return table(pd.DataFrame(rows), index=False)


def shap_table(res: dict, inp: str = "A0m", setting: str = "reg1", top: int = 5) -> str:
    """期間ごとの SHAP の上位(テスト行の平均|寄与|、bp)と、`tnx_level` が寄与全体に占める割合。"""
    rows = []
    for h in SUFFIX:
        v = res.get((h, inp, setting))
        if v is None or not v["shap"]:
            continue
        sh, tot = v["shap"], sum(v["shap"].values())
        rows.append({"期間": f"{h}日", f"上位{top}(bp)": ", ".join(f"{k} {x:.1f}" for k, x in list(sh.items())[:top]),
                     "tnx_levelの割合": f"{sh.get('tnx_level', 0) / tot:.0%}"})
    return table(pd.DataFrame(rows), index=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    res = build()
    text = (to_markdown(res) + "\n## 年別の R² と AUC(A0m reg1)\n\n" + yearly_table(res) +
            "\n## SHAP(A0m reg1)\n\n" + shap_table(res))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text, encoding="utf-8")
        print("保存:", a.out)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
