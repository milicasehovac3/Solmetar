"""numbers.py - ispis svih brojeva koji se navode u tekstu rada."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from . import prep, models

RES = Path(__file__).resolve().parents[1] / "results"


def main():
    L = json.load(open(RES / "metrics.json", encoding="utf-8"))
    T = json.load(open(RES / "transfer.json", encoding="utf-8"))
    df = prep.load_limburg()
    day = df[(df["f_sun"] > 0) & df["kappa"].notna() & (df.index < prep.VAL_END)]
    print("N dnevnih sati (trening+val):", len(day), " ukupno sati:", len(df), "nedostaje kappa:", int(df["kappa"].isna().sum()))
    rows = []
    for c in ["kappa", "G", "G_b", "G_d", "G_n", "T_a", "RH", "v_w", "R", "S", "C"]:
        v = day[c] * (100 if c == "kappa" else 1)
        rows.append([c, v.min(), v.mean(), v.max(), v.std(), day[c].corr(day["kappa"])])
    print(pd.DataFrame(rows, columns=["var", "min", "mean", "max", "std", "r"]).round(3).to_string())
    yr = df["kappa"].groupby(df.index.year).sum()
    print("godisnje kWh/kWp:", yr.round(1).to_dict())
    print("snijeg sati (S>0) trening+val:", int((df.loc[df.index < prep.VAL_END, "S"] > 0).sum()))
    b = L["best"]
    m = models.MLP(len(prep.FEATURES), tuple(b["mlp"]["hidden"]), b["mlp"]["dropout"])
    l = models.LSTMReg(len(prep.SEQ_FEATURES), b["lstm"]["hidden"], b["lstm"]["layers"])
    print("parametri MLP", sum(p.numel() for p in m.parameters()), "LSTM", sum(p.numel() for p in l.parameters()))
    for k in ("MLP", "LSTM"):
        h = L["history"][k]
        print(k, "best_epoch", h["best_epoch"], "epochs", h["epochs"], "sec", round(h["seconds"], 1), "refit sec", round(h["refit_seconds"], 1), "seed", L["selected_seed"][k])
    for k, v in L["search"].items():
        vals = [r["val_RMSE"] for r in v]
        print("search", k, round(min(vals), 3), round(max(vals), 3))
    print("seeds", json.dumps({k: {m_: (round(v[m_]["mean"], 3), round(v[m_]["std"], 3)) for m_ in ("MAE", "RMSE", "MBE", "R2")} for k, v in L["test_seeds"].items()}))
    print("seed RMSE all", {k: [round(d["RMSE"], 3) for d in v["all"]] for k, v in L["test_seeds"].items()})
    print("ablation", {k: (round(v["RMSE_mean"], 3), round(v["RMSE_std"], 3), v["n_features"]) for k, v in L["ablation"].items()})
    print("importance", {k: round(v["mean"], 3) for k, v in L["importance"].items()})
    print("by_month", {k: {m_: round(x, 2) for m_, x in v.items()} for k, v in L["by_month"].items()})
    for c in T["locations"]:
        t = T["locations"][c]
        print(c, "ood_any", round(t.get("ood_any_pct", 0), 1), "Tday", round(t.get("T_day_mean", 0), 1),
              "G_y", {y: round(v) for y, v in t.get("G_annual_kWh_m2", {}).items()},
              "MLP", {y: round(v) for y, v in t["MLP"]["annual"].items()})
    print("Limburg measured", {y: round(v) for y, v in T["locations"]["Limburg"]["measured_annual"].items()})
    for k, v in T["summary"].items():
        if isinstance(v, dict):
            print(k, {kk: (round(vv, 3) if isinstance(vv, float) else {a: round(b_, 3) for a, b_ in vv.items()}) for kk, vv in v.items()})


if __name__ == "__main__":
    main()
