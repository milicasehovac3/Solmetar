"""
transfer.py - primjena modela naucenih u Limburgu na gradove u Bosni i Hercegovini.

Za svaki grad racunaju se satne osobine iz meteoroloskih podataka Open-Meteo i solarne
geometrije lokacije, a modeli procjenjuju faktor iskoristenja kappa_hat. Suma kappa_hat
kroz godinu je specificna godisnja proizvodnja u kWh po kWp instalisane snage, koja se
uporedjuje sa procjenom PVGIS (nezavisan izvor, satelitski podaci SARAH3).

Pokretanje:  python -m solar_nn.transfer
"""
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from . import prep, data, models

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
YEARS = [2021, 2022, 2023, 2024, 2025]
MODELS = ["Linearna regresija", "Gradijentno pojačanje", "MLP", "LSTM"]


def load_models():
    L = json.load(open(RES / "metrics.json", encoding="utf-8"))
    b = L["best"]
    mlp = models.MLP(len(prep.FEATURES), tuple(b["mlp"]["hidden"]), b["mlp"]["dropout"])
    mlp.load_state_dict(torch.load(RES / "mlp.pt"))
    lstm = models.LSTMReg(len(prep.SEQ_FEATURES), b["lstm"]["hidden"], b["lstm"]["layers"])
    lstm.load_state_dict(torch.load(RES / "lstm.pt"))
    with open(RES / "baselines.pkl", "rb") as f:
        bl = pickle.load(f)
    return {"mlp": mlp, "lstm": lstm, **bl}


def predict_frame(df, M):
    """Satne procjene kappa_hat (0 nocu i gdje osobine nisu potpune) za sve modele."""
    Xs, src = data.transform(df, M["scaler"])
    ok = prep.valid_rows(df, prep.FEATURES) & data.seq_valid(df) & prep.daytime(df)
    rows = np.where(ok)[0]
    X = Xs[rows].astype(np.float32)
    out = pd.DataFrame(0.0, index=df.index, columns=MODELS)
    out.loc[df.index[rows], "Linearna regresija"] = M["ridge"].predict(X)
    out.loc[df.index[rows], "Gradijentno pojačanje"] = M["gbm"].predict(X)
    out.loc[df.index[rows], "MLP"] = models.predict(M["mlp"], X)
    out.loc[df.index[rows], "LSTM"] = models.predict(M["lstm"], data.sequences(src, rows))
    out = out.clip(0, 1)
    out["valid"] = ok | ~prep.daytime(df)
    return out


def annual(pred, col):
    s = pred[col]
    return {int(y): float(s[s.index.year == y].sum()) for y in YEARS}


def monthly_mean(pred, col):
    s = pred[col][(pred.index.year >= YEARS[0]) & (pred.index.year <= YEARS[-1])]
    m = s.groupby([s.index.year, s.index.month]).sum()
    return {int(mo): float(m.xs(mo, level=1).mean()) for mo in range(1, 13)}


def main():
    M = load_models()
    meteo = prep.load_meteo()
    pv = prep.load_pvgis()
    res = {"years": YEARS, "locations": {}}

    # Limburg: izmjereno vs modeli vs PVGIS
    lim = prep.load_limburg(meteo)
    pl = predict_frame(lim, M)
    pl["measured"] = lim["kappa"].fillna(0)
    rng = {c: (float(lim.loc[lim.index < prep.VAL_END, c].min()), float(lim.loc[lim.index < prep.VAL_END, c].max()))
           for c in prep.FEATURES}
    res["train_range"] = rng
    r = {"PVGIS_E_y": float(pv.loc["Limburg", "E_y"]),
         "PVGIS_E_m": [float(pv.loc["Limburg", f"E_m{i}"]) for i in range(1, 13)],
         "measured_annual": annual(pl, "measured"), "measured_monthly": monthly_mean(pl, "measured")}
    for k in MODELS:
        r[k] = {"annual": annual(pl, k), "monthly": monthly_mean(pl, k)}
    res["locations"]["Limburg"] = r
    meas_mean = np.mean(list(r["measured_annual"].values()))
    q = meas_mean / r["PVGIS_E_y"]
    res["q_Limburg"] = float(q)

    hourly = {}
    for city in prep.CITIES:
        df = prep.location_frame(city, meteo)
        df = df[(df.index >= prep.TRAIN_START) & (df.index < "2026-09-24")]
        p = predict_frame(df, M)
        hourly[city] = p
        day = prep.daytime(df) & prep.valid_rows(df, prep.FEATURES)
        ood = {c: float(100 * np.mean((df[c].values[day] < rng[c][0]) | (df[c].values[day] > rng[c][1])))
               for c in prep.FEATURES}
        r = {"PVGIS_E_y": float(pv.loc[city, "E_y"]),
             "PVGIS_E_m": [float(pv.loc[city, f"E_m{i}"]) for i in range(1, 13)],
             "PVGIS_slope": float(pv.loc[city, "slope"]),
             "ood_pct": ood, "ood_any_pct": float(100 * np.mean(np.any([(df[c].values[day] < rng[c][0]) |
                                                                           (df[c].values[day] > rng[c][1]) for c in prep.FEATURES], axis=0))),
             "G_annual_kWh_m2": {int(y): float(df["G"][df.index.year == y].sum() / 1000) for y in YEARS},
             "T_day_mean": float(df["T_a"].values[day].mean())}
        for k in MODELS:
            r[k] = {"annual": annual(p, k), "monthly": monthly_mean(p, k)}
        res["locations"][city] = r
        print(city, {k: round(np.mean(list(r[k]["annual"].values())), 1) for k in MODELS},
              "PVGIS", r["PVGIS_E_y"], "PVGIS*q", round(r["PVGIS_E_y"] * q, 1), "ood", round(r["ood_any_pct"], 2), flush=True)

    # sazetak: poredjenje relativnih odnosa (grad / Limburg) i korelacija medju gradovima
    summ = {}
    for k in MODELS + ["measured"]:
        pass
    cities = list(prep.CITIES)
    pvg = np.array([res["locations"][c]["PVGIS_E_y"] for c in cities])
    for k in MODELS:
        mod = np.array([np.mean(list(res["locations"][c][k]["annual"].values())) for c in cities])
        lim_mod = np.mean(list(res["locations"]["Limburg"][k]["annual"].values()))
        rel_mod = mod / lim_mod
        rel_pvg = pvg / res["locations"]["Limburg"]["PVGIS_E_y"]
        mon_r = [np.corrcoef([res["locations"][c][k]["monthly"][m] for m in range(1, 13)],
                             res["locations"][c]["PVGIS_E_m"])[0, 1] for c in cities]
        summ[k] = {"mean_annual": dict(zip(cities, mod.tolist())),
                   "vs_PVGISq_pct": dict(zip(cities, (100 * (mod / (pvg * q) - 1)).tolist())),
                   "MAPE_vs_PVGISq": float(np.mean(np.abs(100 * (mod / (pvg * q) - 1)))),
                   "rel_model": dict(zip(cities, rel_mod.tolist())),
                   "rel_PVGIS": dict(zip(cities, rel_pvg.tolist())),
                   "rel_err_pct": float(np.mean(np.abs(100 * (rel_mod / rel_pvg - 1)))),
                   "r_cities": float(np.corrcoef(mod, pvg)[0, 1]),
                   "r_monthly_mean": float(np.mean(mon_r)),
                   "Limburg_model_mean": float(lim_mod)}
    summ["Limburg_measured_mean"] = float(meas_mean)
    res["summary"] = summ
    with open(RES / "transfer.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    pd.concat({c: h["MLP"] for c, h in hourly.items()}, axis=1).to_csv(RES / "bih_mlp_hourly.csv")
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk in ("MAPE_vs_PVGISq", "rel_err_pct", "r_cities", "r_monthly_mean", "Limburg_model_mean")}
                      for k, v in summ.items() if isinstance(v, dict)}, ensure_ascii=False, indent=1), "q", q, "meas", meas_mean)


if __name__ == "__main__":
    main()
