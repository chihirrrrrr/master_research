"""研究1: S&P500の将来リターンを、XGBoostで予測する。数値のみ / 数値+NYT。

使い方(リポジトリのルートから):
    python -m src.models.study1 --features numeric --periods dev
    python -m src.models.study1 --features numeric+nyt --periods dev

- 時間順の walk-forward(拡張窓)。年ごとに、その年より前で学習して、その年をテストする(configs/split.yaml)。
- **開発(dev)の期間だけ**を評価する。**最終確認(confirm)は、`--final` を付けたときだけ**(最後に1回だけ見るため)。
- 数値のみと +NYT は、同じ行(NYTの特徴が揃う行)・同じ分割・同じ設定で比べる。
- 結果は results/runs/<日付>_<名前>/ に保存する(config.yaml、metrics.json、notes.md、git_commit.txt、artifacts/)。
"""
import argparse
import json
import subprocess
import sys
from datetime import date
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.eval import metrics as mt
from src.eval import walk_forward as wf
from src.features import market_level as ml
from src.features import nyt_text as nt

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"


def load_configs(model_config=None):
    split = yaml.safe_load((ROOT / "configs" / "split.yaml").read_text(encoding="utf-8"))["study1"]
    path = Path(model_config) if model_config else ROOT / "configs" / "study1_xgb.yaml"
    model = yaml.safe_load(path.read_text(encoding="utf-8"))
    return split, model


def load_dataset(feature_set: str, target: str, row_set: str):
    """(行ごとの表, 使う説明変数の列)。row_set='common' なら、NYTの特徴が揃う行に、全条件でそろえる。"""
    df = pd.read_parquet(PROC / "market_features.parquet").merge(
        pd.read_parquet(PROC / "market_targets.parquet"), on="date", how="inner")
    cols = list(ml.MARKET_FEATURES)
    need = cols + [target]
    if row_set == "common" or feature_set == "numeric+nyt":
        df = df.merge(pd.read_parquet(PROC / "nyt_features.parquet"), on="date", how="left")
        need = need + nt.NYT_COLUMNS
    if feature_set == "numeric+nyt":
        cols = cols + nt.NYT_COLUMNS
    return df.dropna(subset=need).sort_values("date").reset_index(drop=True), cols


def make_model(kind: str, cfg: dict, depth: int, mcw: int, seed: int, n_estimators=None, early=False):
    if kind == "xgboost":
        from xgboost import XGBRegressor
        return XGBRegressor(
            objective="reg:squarederror", n_estimators=n_estimators or cfg["n_estimators_max"],
            learning_rate=cfg["learning_rate"], max_depth=depth, min_child_weight=mcw, subsample=cfg["subsample"],
            colsample_bytree=cfg["colsample_bytree"], reg_lambda=cfg["reg_lambda"], tree_method="hist",
            random_state=seed, n_jobs=1, importance_type="gain", reg_alpha=cfg.get("reg_alpha", 0), gamma=cfg.get("gamma", 0),
            early_stopping_rounds=cfg["early_stopping_rounds"] if early else None)
    if kind == "hgb":  # パイプラインのテスト用の代用品(研究では使わない)
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(max_depth=depth, min_samples_leaf=mcw, learning_rate=0.05,
                                             max_iter=n_estimators or 60, random_state=seed)
    raise ValueError(kind)


def fit_predict(Xtr, ytr, Xte, cfg: dict, kind: str, share: float, embargo: int):
    """学習期間の内側(時間順)で設定を選び、学習期間の全体で学び直して、テストを予測する。"""
    inner_tr, inner_va = wf.inner_split(len(Xtr), share, embargo)
    best, best_score = None, np.inf
    for depth, mcw in product(cfg["grid"]["max_depth"], cfg["grid"]["min_child_weight"]):
        m = make_model(kind, cfg, depth, mcw, cfg["seeds"][0], early=(kind == "xgboost"))
        if kind == "xgboost":
            m.fit(Xtr[inner_tr], ytr[inner_tr], eval_set=[(Xtr[inner_va], ytr[inner_va])], verbose=False)
            n_best = int(m.best_iteration) + 1
        else:
            m.fit(Xtr[inner_tr], ytr[inner_tr])
            n_best = None
        score = mt.rmse(ytr[inner_va], m.predict(Xtr[inner_va]))
        if score < best_score:
            best, best_score = {"max_depth": depth, "min_child_weight": mcw, "n_estimators": n_best}, score
    preds, importance = [], None
    for k, seed in enumerate(cfg["seeds"]):
        m = make_model(kind, cfg, best["max_depth"], best["min_child_weight"], seed, n_estimators=best["n_estimators"])
        m.fit(Xtr, ytr)
        preds.append(m.predict(Xte))
        if k == 0 and hasattr(m, "feature_importances_"):
            importance = np.asarray(m.feature_importances_, dtype=float)
    return np.mean(preds, axis=0), {**best, "inner_val_rmse": float(best_score)}, importance


def pooled_metrics(p: pd.DataFrame) -> dict:
    """予測を全部つなげた指標(p: y, pred, hist の列を持つ)。"""
    return {"n_test": int(len(p)), "r2_os": mt.r2_os(p["y"], p["pred"], p["hist"]),
            "rmse_model": mt.rmse(p["y"], p["pred"]), "rmse_hist": mt.rmse(p["y"], p["hist"]),
            "corr": mt.corr(p["y"], p["pred"]), "hit_model": mt.hit_rate(p["y"], p["pred"]),
            "hit_hist": mt.hit_rate(p["y"], p["hist"]), "up_share": float((p["y"] > 0).mean()),
            "clark_west_vs_hist": mt.clark_west(p["y"], p["hist"], p["pred"], lag=5),
            "direction": mt.direction_metrics(p["y"], p["pred"])}


def git_state() -> str:
    try:
        h = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout.strip())
        return f"{h}{' (dirty: 未コミットの変更あり)' if dirty else ''}"
    except Exception:
        return "unknown"


def run(feature_set: str, periods: str, final: bool = False, kind: str = "xgboost", row_set=None, name=None,
        out_root=None, dataset=None, split=None, model=None, model_config=None) -> dict:
    if periods in ("confirm", "all") and not final:
        raise SystemExit("最終確認(confirm)の期間を含む実行には --final が要る(開けると、未見の確認データではなくなる)。")
    split_cfg, model_cfg = (split, model) if split is not None else load_configs(model_config)
    target = model_cfg["target"]
    if dataset is not None:
        df, cols = dataset
    else:
        if row_set is None:
            row_set = "common" if (PROC / "nyt_features.parquet").exists() else "numeric"
        df, cols = load_dataset(feature_set, target, row_set)
    dev, conf = split_cfg["development"], split_cfg["confirmation"]
    first, last = {"dev": (dev["first"], dev["last"]), "confirm": (conf["first"], conf["last"]),
                   "all": (dev["first"], conf["last"])}[periods]
    years = list(range(first, last + 1))
    folds = wf.make_folds(df["date"], years, split_cfg["embargo_days"])
    X, y = df[cols].to_numpy(float), df[target].to_numpy(float)
    share = split_cfg["inner_validation"]["share"]

    rows, per_fold, imps = [], [], []
    for f in folds:
        pred, best, imp = fit_predict(X[f["train"]], y[f["train"]], X[f["test"]], model_cfg["model"], kind, share,
                                      split_cfg["embargo_days"])
        hist = float(y[f["train"]].mean())
        yt = y[f["test"]]
        rows.append(pd.DataFrame({"date": df["date"].iloc[f["test"]].to_numpy(), "year": f["year"], "y": yt,
                                  "pred": pred, "hist": hist}))
        per_fold.append({"year": f["year"], "n_train": int(len(f["train"])), "n_test": int(len(f["test"])), "best": best,
                         "r2_os": mt.r2_os(yt, pred, hist), "rmse_model": mt.rmse(yt, pred), "rmse_hist": mt.rmse(yt, hist),
                         "corr": mt.corr(yt, pred), "hit_model": mt.hit_rate(yt, pred), "hit_hist": mt.hit_rate(yt, np.full_like(yt, hist))})
        if imp is not None:
            imps.append(imp)
    p = pd.concat(rows, ignore_index=True)
    pooled = pooled_metrics(p)
    imp_top = None
    if imps:
        mean_imp = np.mean(imps, axis=0)
        order = np.argsort(-mean_imp)[:8]
        imp_top = {cols[i]: float(mean_imp[i]) for i in order}
    result = {"feature_set": feature_set, "periods": periods, "model": kind, "target": target, "n_features": len(cols),
              "features": cols, "row_set": row_set, "n_rows": int(len(df)),
              "first_date": str(df["date"].min().date()), "last_date": str(df["date"].max().date()),
              "folds": per_fold, "pooled": pooled, "importance_top": imp_top}
    if periods == "all":  # 開発(〜2017)と最終確認(2018〜)を、別々にも集計する
        result["pooled_dev"] = pooled_metrics(p[p["year"] <= dev["last"]])
        result["pooled_confirm"] = pooled_metrics(p[p["year"] >= conf["first"]])

    if out_root is not False:
        out_root = Path(out_root) if out_root else ROOT / "results" / "runs"
        run_dir = out_root / f"{date.today():%Y%m%d}_{name or f'study1_{feature_set}_{kind}_{periods}'.replace('+', '_')}"
        k = 2
        while run_dir.exists():
            run_dir = run_dir.with_name(f"{run_dir.name.rsplit('_v', 1)[0]}_v{k}")
            k += 1
        (run_dir / "artifacts").mkdir(parents=True)
        (run_dir / "config.yaml").write_text(yaml.safe_dump(
            {"args": {"features": feature_set, "periods": periods, "final": final, "model": kind, "row_set": row_set,
                      "model_config": str(model_config) if model_config else "configs/study1_xgb.yaml"},
             "split": split_cfg, "model": model_cfg}, allow_unicode=True, sort_keys=False), encoding="utf-8")
        (run_dir / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        (run_dir / "git_commit.txt").write_text(git_state() + "\n", encoding="utf-8")
        p.to_parquet(run_dir / "artifacts" / "predictions.parquet", index=False)
        pooled_r2 = pooled["r2_os"]
        (run_dir / "notes.md").write_text(
            f"# {run_dir.name}\n\n- 入力: {feature_set}({len(cols)}列) / モデル: {kind} / 期間: {periods}"
            f"{ {'confirm': '(最終確認)', 'all': '(開発 + 最終確認。2018〜2021を開けた。事前登録は docs/DECISIONS.md)', 'dev': '(開発)'}[periods] } / 行: {len(df):,}\n"
            f"- 全体の標本外R²(履歴平均に対して): {pooled_r2:+.4f} / 相関 {pooled['corr']:+.3f} / 符号の的中率 {pooled['hit_model']:.3f}"
            f"(履歴平均 {pooled['hit_hist']:.3f})\n\n## 考察(要記入)\n\n(結果を見て、気づきと次の仮説を書く)\n", encoding="utf-8")
        result["run_dir"] = str(run_dir)
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", choices=["numeric", "numeric+nyt"], required=True)
    ap.add_argument("--periods", choices=["dev", "confirm", "all"], default="dev")
    ap.add_argument("--final", action="store_true", help="最終確認(confirm)を実行する(最後に1回だけ)")
    ap.add_argument("--model", choices=["xgboost", "hgb"], default="xgboost")
    ap.add_argument("--row-set", choices=["numeric", "common"], default=None)
    ap.add_argument("--name", default=None)
    ap.add_argument("--model-config", default=None, help="XGBoostの設定ファイル(既定: configs/study1_xgb.yaml)")
    a = ap.parse_args()
    r = run(a.features, a.periods, a.final, a.model, a.row_set, a.name, model_config=a.model_config)
    pl = r["pooled"]
    print(f"入力 {a.features}({r['n_features']}列) / 行 {r['n_rows']:,}({r['first_date']}〜{r['last_date']}、row_set={r['row_set']}) / {a.periods}")
    print(f"{'年':>5} {'学習':>6} {'テスト':>6} {'R2_os':>8} {'相関':>7} {'的中':>6} {'(履歴平均)':>9}  設定")
    for f in r["folds"]:
        b = f["best"]
        print(f"{f['year']:>5} {f['n_train']:>6} {f['n_test']:>6} {f['r2_os']:>+8.4f} {f['corr']:>+7.3f} {f['hit_model']:>6.3f} {f['hit_hist']:>9.3f}  深さ{b['max_depth']} 葉{b['min_child_weight']} 木{b['n_estimators']}")
    cw = pl["clark_west_vs_hist"]
    print(f"全体: R2_os {pl['r2_os']:+.4f} / 相関 {pl['corr']:+.3f} / 的中 {pl['hit_model']:.3f}(履歴平均 {pl['hit_hist']:.3f}、上昇の割合 {pl['up_share']:.3f}) / Clark-West t={cw['t']:+.2f} p(片側)={cw['p_one_sided']:.3f}")
    d = pl["direction"]
    print(f"方向(上昇/下降)として: accuracy {d['accuracy']:.3f}(いつも上昇 {d['always_up_accuracy']:.3f}) / balanced acc {d['balanced_accuracy']:.3f} / F1(上昇) {d['f1_up']:.3f}(いつも上昇 {d['always_up_f1']:.3f}) / AUC {d['auc']:.3f}(0.5=偶然)")
    print(f"予測のばらつき: 標準偏差 {d['pred_std']:.1f}bp(実現値 {d['y_std']:.1f}bp) / 『上昇』と予測した割合 {d['pred_up_share']:.3f}  ← 予測の標準偏差が実現値よりずっと小さいと、ほぼ定数(平均)の予測")
    for key, label in (("pooled_dev", "うち開発(〜2017)"), ("pooled_confirm", "うち最終確認(2018〜)")):
        if key in r:
            q = r[key]
            print(f"  {label}: {q['n_test']}日 R2_os {q['r2_os']:+.4f} / 相関 {q['corr']:+.3f} / AUC {q['direction']['auc']:.3f}")
    print("保存:", r["run_dir"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
