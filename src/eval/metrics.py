"""評価指標。予測の良さを、「履歴平均(物差し)」と比べて測る。

- r2_os        : 標本外R² = 1 − Σ(y−予測)² / Σ(y−履歴平均)²。0 より大きければ、履歴平均より良い。
- clark_west   : 入れ子のモデル(物差し ⊂ 対象)の比較に使う検定(Clark & West, 2007)。5日平均の重なりを、HAC(Newey-West)で補正する。
"""
import numpy as np
import statsmodels.api as sm
from scipy.stats import norm
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score


def _arr(x) -> np.ndarray:
    return np.asarray(x, dtype=float)


def r2_os(y, pred, bench) -> float:
    y, pred, bench = _arr(y), _arr(pred), _arr(bench)
    return float(1.0 - np.sum((y - pred) ** 2) / np.sum((y - bench) ** 2))


def rmse(y, pred) -> float:
    return float(np.sqrt(np.mean((_arr(y) - _arr(pred)) ** 2)))


def corr(y, pred) -> float:
    y, pred = _arr(y), _arr(pred)
    if np.std(pred) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(y, pred)[0, 1])


def hit_rate(y, pred) -> float:
    """予測の符号が、実現の符号と一致する割合(実現が0の行は除く)。"""
    y, pred = _arr(y), _arr(pred)
    m = y != 0
    return float(np.mean(np.sign(pred[m]) == np.sign(y[m])))


def clark_west(y, bench, alt, lag: int = 5) -> dict:
    """物差し(bench)に対して、対象(alt)が良いかを検定する(片側)。

    f_t = (y−bench)² − [(y−alt)² − (bench−alt)²] の平均が 0 より大きければ、alt が良い。
    """
    y, bench, alt = _arr(y), _arr(bench), _arr(alt)
    f = (y - bench) ** 2 - ((y - alt) ** 2 - (bench - alt) ** 2)
    res = sm.OLS(f, np.ones(len(f))).fit(cov_type="HAC", cov_kwds={"maxlags": lag})
    t = float(res.tvalues[0])
    return {"mean": float(f.mean()), "t": t, "p_one_sided": float(1.0 - norm.cdf(t)), "lag": lag}


def direction_metrics(y, pred) -> dict:
    """方向(上昇/下降)の分類として見た指標。リターンの符号を、上昇(1)/下降(0)とみなす。

    上昇の割合が高い(約62%)ので、accuracy と F1(上昇)は「いつも上昇」でも高い。基準は「いつも上昇」と比べること。
    AUC は、予測値の大小で上昇日を当てる力(0.5 = 偶然)。balanced accuracy は、上昇・下降を同じ重みで見た的中率。
    pred_up_share が 1.0 に近い、または pred_std が実現値の標準偏差よりずっと小さいときは、予測が「ほぼ定数」。
    """
    y, pred = _arr(y), _arr(pred)
    yc, pc = (y > 0).astype(int), (pred > 0).astype(int)
    return {"accuracy": float(np.mean(yc == pc)), "balanced_accuracy": float(balanced_accuracy_score(yc, pc)),
            "f1_up": float(f1_score(yc, pc, zero_division=0)),
            "auc": float(roc_auc_score(yc, pred)) if 0 < yc.mean() < 1 else float("nan"),
            "pred_up_share": float(pc.mean()), "always_up_accuracy": float(yc.mean()),
            "always_up_f1": float(f1_score(yc, np.ones_like(yc), zero_division=0)),
            "pred_std": float(np.std(pred)), "y_std": float(np.std(y))}
