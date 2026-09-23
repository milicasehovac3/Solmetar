"""metrics.py - metrike tacnosti (oznake kao u radu: y, y_hat, N, P_nom)."""
import numpy as np
from scipy import stats


def mae(y, yh):
    return float(np.mean(np.abs(y - yh)))


def rmse(y, yh):
    return float(np.sqrt(np.mean((y - yh) ** 2)))


def mbe(y, yh):
    return float(np.mean(yh - y))


def r2(y, yh):
    return float(1 - np.sum((y - yh) ** 2) / np.sum((y - np.mean(y)) ** 2))


def all_metrics(y, yh, p_nom, rmse_ref=None):
    out = {
        "MAE": mae(y, yh), "RMSE": rmse(y, yh), "MBE": mbe(y, yh), "R2": r2(y, yh),
        "nMAE": 100 * mae(y, yh) / p_nom, "nRMSE": 100 * rmse(y, yh) / p_nom,
    }
    if rmse_ref is not None:
        out["SS"] = 100 * (1 - out["RMSE"] / rmse_ref)
    return out


def diebold_mariano(y, yh1, yh2, lags=24):
    """Diebold-Mariano test (kvadratni gubitak), dvostrani. Varijansa srednje razlike
    procjenjuje se Newey-West (HAC) procjeniteljem sa 'lags' pomaka (Bartlettove tezine),
    jer su satne greske autokorelisane. Pozitivna statistika znaci da model 2 ima manju gresku."""
    d = (y - yh1) ** 2 - (y - yh2) ** 2
    n = len(d)
    dm = d.mean()
    gamma0 = np.mean((d - dm) ** 2)
    var = gamma0
    for k in range(1, lags + 1):
        w = 1 - k / (lags + 1)
        var += 2 * w * np.sum((d[k:] - dm) * (d[:-k] - dm)) / n
    stat = dm / np.sqrt(var / n)
    p = 2 * (1 - stats.norm.cdf(abs(stat)))
    return float(stat), float(p)
