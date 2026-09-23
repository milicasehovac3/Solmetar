"""
train.py - kompletan eksperiment na podacima iz Limburga (Belgija):
izbor hiperparametara na validaciji, ponovno treniranje odabranih modela na treningu +
validaciji (zbog promjene ponasanja sistema kroz vrijeme), vise sjemena, evaluacija na
testnom periodu 1.7.2025. - 21.9.2026. (dnevni sati), poredjenje sa prognozama operatora
Elia, Diebold-Mariano testovi, permutaciona vaznost i ablacija osobina.

Pokretanje (iz korijena projekta):  python -m solar_nn.train
Sve greske izrazene su u procentima instalisane snage (kappa * 100).
"""
import json
import itertools
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor

from . import prep, data, models, metrics

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
RES.mkdir(exist_ok=True)
SEEDS = [0, 1, 2, 3, 4]
NN = ["Linearna regresija", "Gradijentno pojačanje", "MLP", "LSTM"]
ELIA = {"Elia, dan unaprijed": "kappa_da", "Elia, dan unaprijed (11 h)": "kappa_da11",
        "Elia, unutardnevna": "kappa_mr"}


def mk(cls, *args, seed=0):
    """Inicijalizacija tezina uz fiksno sjeme (ponovljivost)."""
    models.set_seed(seed)
    return cls(*args)


def clip(yhat):
    """Fizicko ogranicenje izlaza: 0 <= kappa_hat <= 1."""
    return np.clip(yhat, 0.0, 1.0)


def pct(v):
    return 100.0 * np.asarray(v, dtype=float)


def fit_nn(kind, cfg, D, S, seed, idx=None):
    """1) treniranje na treningu uz rano zaustavljanje (validacija), 2) ponovno treniranje
    na treningu + validaciji sa istim brojem epoha i istim rasporedom stope ucenja."""
    tr, va, tv = D["train"], D["val"], D["trval"]
    if kind == "MLP":
        Xtr, Xva, Xtv = tr["X"], va["X"], tv["X"]
        if idx is not None:
            Xtr, Xva, Xtv = Xtr[:, idx], Xva[:, idx], Xtv[:, idx]
        new = lambda: mk(models.MLP, Xtr.shape[1], tuple(cfg["hidden"]), cfg["dropout"], seed=seed)
        kw = {"lr": cfg["lr"]}
    else:
        Xtr, Xva, Xtv = S["train"], S["val"], S["trval"]
        new = lambda: mk(models.LSTMReg, Xtr.shape[2], cfg["hidden"], cfg["layers"], seed=seed)
        kw = {"lr": cfg["lr"], "max_epochs": 80, "patience": 10}
    m1, h1 = models.train_nn(new(), Xtr, tr["y"], Xva, va["y"], seed=seed, **kw)
    val = metrics.rmse(pct(va["y"]), pct(clip(models.predict(m1, Xva))))
    m2, h2 = models.train_nn(new(), Xtv, tv["y"], None, None, seed=seed,
                             lr_schedule=h1["lr"][:h1["best_epoch"]])
    h1["refit_seconds"] = h2["seconds"]
    return m1, m2, h1, val


def main():
    D = data.build()
    tr, va, te, tv = D["train"], D["val"], D["test"], D["trval"]
    df = D["df"]
    S = {k: data.sequences(D["Xseq_src"], D[k]["rows"]) for k in ("train", "val", "test", "trval")}
    y_te = pct(te["y"])
    cap_te = df["capacity_MW"].values[te["rows"]]
    log = {"meta": {
        "window_h": data.WINDOW, "seeds": SEEDS,
        "n_train": int(len(tr["y"])), "n_val": int(len(va["y"])), "n_test": int(len(te["y"])),
        "train_period": [str(df.index[tr["rows"][0]]), str(df.index[tr["rows"][-1]])],
        "val_period": [str(df.index[va["rows"][0]]), str(df.index[va["rows"][-1]])],
        "test_period": [str(df.index[te["rows"][0]]), str(df.index[te["rows"][-1]])],
        "features": prep.FEATURES, "seq_features": prep.SEQ_FEATURES,
        "capacity_test_mean_MW": float(cap_te.mean()),
        "capacity_first_MW": float(df["capacity_MW"].iloc[0]),
        "capacity_last_MW": float(df["capacity_MW"].iloc[-1]),
    }}
    val_rmse = lambda yh: metrics.rmse(pct(va["y"]), pct(clip(yh)))

    # ---------------- 1) izbor hiperparametara (samo validacioni skup) ----------------
    search = {"ridge": [], "gbm": [], "mlp": [], "lstm": []}
    for a in [0.01, 0.1, 1, 10, 100]:
        m = Ridge(alpha=a).fit(tr["X"], tr["y"])
        search["ridge"].append({"alpha": a, "val_RMSE": val_rmse(m.predict(va["X"]))})
    best_ridge = min(search["ridge"], key=lambda r: r["val_RMSE"])

    for lr, leaves, it in itertools.product([0.03, 0.1], [15, 31, 63], [300, 800]):
        m = HistGradientBoostingRegressor(learning_rate=lr, max_leaf_nodes=leaves, max_iter=it,
                                          early_stopping=False, random_state=0).fit(tr["X"], tr["y"])
        search["gbm"].append({"learning_rate": lr, "max_leaf_nodes": leaves, "max_iter": it,
                              "val_RMSE": val_rmse(m.predict(va["X"]))})
    best_gbm = min(search["gbm"], key=lambda r: r["val_RMSE"])
    print("ridge", best_ridge, "gbm", best_gbm, flush=True)

    for hidden, drop, lr in itertools.product([(64, 32), (128, 64, 32), (256, 128, 64)],
                                              [0.0, 0.1, 0.2], [1e-3, 3e-4]):
        m = mk(models.MLP, tr["X"].shape[1], hidden, drop, seed=0)
        m, h = models.train_nn(m, tr["X"], tr["y"], va["X"], va["y"], lr=lr, seed=0)
        r = {"hidden": list(hidden), "dropout": drop, "lr": lr,
             "val_RMSE": val_rmse(models.predict(m, va["X"])), "epochs": h["best_epoch"]}
        search["mlp"].append(r)
        print("MLP", r, flush=True)
    best_mlp = min(search["mlp"], key=lambda r: r["val_RMSE"])

    for hidden, layers in itertools.product([32, 64, 128], [1, 2]):
        m = mk(models.LSTMReg, S["train"].shape[2], hidden, layers, seed=0)
        m, h = models.train_nn(m, S["train"], tr["y"], S["val"], va["y"], lr=1e-3, seed=0,
                               max_epochs=80, patience=10)
        r = {"hidden": hidden, "layers": layers, "lr": 1e-3,
             "val_RMSE": val_rmse(models.predict(m, S["val"])), "epochs": h["best_epoch"]}
        search["lstm"].append(r)
        print("LSTM", r, flush=True)
    best_lstm = min(search["lstm"], key=lambda r: r["val_RMSE"])
    log["search"] = search
    log["best"] = {"ridge": best_ridge, "gbm": best_gbm, "mlp": best_mlp, "lstm": best_lstm}

    # ---------------- 2) konacni modeli (trening + validacija) i testna evaluacija ----------------
    preds = {"Persistencija (t-24)": pct(df["kappa_lag24"].values[te["rows"]])}
    ridge = Ridge(alpha=best_ridge["alpha"]).fit(tv["X"], tv["y"])
    preds["Linearna regresija"] = pct(clip(ridge.predict(te["X"])))
    gbm_kw = dict(learning_rate=best_gbm["learning_rate"], max_leaf_nodes=best_gbm["max_leaf_nodes"],
                  max_iter=best_gbm["max_iter"], early_stopping=False, random_state=0)
    gbm = HistGradientBoostingRegressor(**gbm_kw).fit(tv["X"], tv["y"])
    preds["Gradijentno pojačanje"] = pct(clip(gbm.predict(te["X"])))
    # bez ponovnog treniranja (samo trening), radi ilustracije promjene ponasanja sistema
    no_refit = {
        "Linearna regresija": pct(clip(Ridge(alpha=best_ridge["alpha"]).fit(tr["X"], tr["y"]).predict(te["X"]))),
        "Gradijentno pojačanje": pct(clip(HistGradientBoostingRegressor(**gbm_kw).fit(tr["X"], tr["y"]).predict(te["X"]))),
    }

    per_seed = {"MLP": [], "LSTM": []}
    store = {"MLP": [], "LSTM": []}
    for seed in SEEDS:
        for kind, cfg, X in (("MLP", best_mlp, te["X"]), ("LSTM", best_lstm, S["test"])):
            m1, m2, h, v = fit_nn(kind, cfg, D, S, seed)
            yh = pct(clip(models.predict(m2, X)))
            yh1 = pct(clip(models.predict(m1, X)))
            per_seed[kind].append(metrics.all_metrics(y_te, yh, 100.0))
            store[kind].append({"val": v, "model": m2, "hist": h, "yh": yh, "yh_norefit": yh1})
        print("seed", seed, per_seed["MLP"][-1]["RMSE"], per_seed["LSTM"][-1]["RMSE"], flush=True)

    sel = {}
    for k in ("MLP", "LSTM"):   # reprezentativni model = najbolji na VALIDACIJI (ne na testu)
        sel[k] = min(store[k], key=lambda t: t["val"])
        preds[k] = sel[k]["yh"]
        no_refit[k] = sel[k]["yh_norefit"]
        h = dict(sel[k]["hist"])
        log.setdefault("history", {})[k] = {kk: h[kk] for kk in ("train", "val", "lr", "best_epoch", "epochs", "seconds", "refit_seconds")}
        log.setdefault("selected_seed", {})[k] = SEEDS[store[k].index(sel[k])]

    elia_ok = np.all([np.isfinite(df[c].values[te["rows"]]) for c in ELIA.values()], axis=0)
    for nm, c in ELIA.items():
        preds[nm] = pct(df[c].values[te["rows"]])
    log["elia_missing"] = int((~elia_ok).sum())
    ref = metrics.rmse(y_te, preds["Persistencija (t-24)"])
    log["test"] = {}
    for k, v in preds.items():
        r = metrics.all_metrics(y_te, v, 100.0, rmse_ref=ref)
        r["RMSE_MW"] = float(np.sqrt(np.mean(((v - y_te) / 100 * cap_te) ** 2)))
        r["MAE_MW"] = float(np.mean(np.abs(v - y_te) / 100 * cap_te))
        log["test"][k] = r
    log["test_norefit"] = {k: metrics.all_metrics(y_te, v, 100.0) for k, v in no_refit.items()}
    log["test_seeds"] = {}
    for k, lst in per_seed.items():
        log["test_seeds"][k] = {m: {"mean": float(np.mean([d[m] for d in lst])),
                                    "std": float(np.std([d[m] for d in lst], ddof=1))} for m in lst[0]}
        log["test_seeds"][k]["all"] = lst

    # Diebold-Mariano testovi
    log["dm"] = {}
    for a, b in [("Linearna regresija", "MLP"), ("Gradijentno pojačanje", "MLP"), ("MLP", "LSTM"),
                 ("Gradijentno pojačanje", "LSTM"), ("Linearna regresija", "LSTM"),
                 ("MLP", "Elia, dan unaprijed"), ("LSTM", "Elia, dan unaprijed")]:
        s, p = metrics.diebold_mariano(y_te, preds[a], preds[b])
        log["dm"][f"{a} vs {b}"] = {"DM": s, "p": p}

    # greske po klasama oblacnosti, po mjesecima i za dane sa snijegom
    C = df["C"].values[te["rows"]]
    cls = np.where(C < 30, "vedro (C < 30 %)", np.where(C <= 70, "djelimično oblačno", "oblačno (C > 70 %)"))
    months = df.index[te["rows"]].month
    snow = df["S_72"].values[te["rows"]] > 0
    comp = NN + ["Elia, dan unaprijed"]
    log["by_cloud"], log["by_month"], log["snow"] = {}, {}, {"n": int(snow.sum())}
    for k in comp:
        log["by_cloud"][k] = {c: {"RMSE": metrics.rmse(y_te[cls == c], preds[k][cls == c]),
                                  "n": int((cls == c).sum())} for c in np.unique(cls)}
        log["by_month"][k] = {int(mo): metrics.rmse(y_te[months == mo], preds[k][months == mo])
                              for mo in np.unique(months)}
        log["snow"][k] = {"RMSE_snow": metrics.rmse(y_te[snow], preds[k][snow]),
                          "MBE_snow": metrics.mbe(y_te[snow], preds[k][snow])}

    # dnevna energija po jedinici instalisane snage (kWh/kWp po danu = suma kappa)
    dates = df.index[te["rows"]].date
    e = pd.DataFrame({"d": dates, "y": y_te / 100, **{k: preds[k] / 100 for k in comp}}).groupby("d").sum()
    log["daily_energy"] = {k: {"MAE": float(np.mean(np.abs(e[k] - e["y"]))),
                               "nMAE_pct": float(100 * np.mean(np.abs(e[k] - e["y"])) / e["y"].mean())}
                           for k in comp}
    log["daily_energy"]["n_days"] = int(len(e))
    log["daily_energy"]["mean"] = float(e["y"].mean())

    # ---------------- 3) permutaciona vaznost (MLP, test) ----------------
    mlp = sel["MLP"]["model"]
    base = metrics.rmse(y_te, pct(clip(models.predict(mlp, te["X"]))))
    groups = {"G": ["G"], "G_b, G_d, G_n": ["G_b", "G_d", "G_n"], "G(t-1), G(t+1)": ["G_lag1", "G_lead1"],
              "k_t": ["k_t"], "C": ["C"], "T_a": ["T_a"], "RH": ["RH"], "v_w": ["v_w"], "R": ["R"],
              "S, S_72": ["S", "S_72"], "geometrija": ["cos_z", "G_0", "f_sun", "w_sin"]}
    rng = np.random.default_rng(0)
    imp = {}
    for gname, cols in groups.items():
        idx = [prep.FEATURES.index(c) for c in cols]
        vals = []
        for _ in range(10):
            Xp = te["X"].copy()
            Xp[:, idx] = Xp[rng.permutation(len(Xp))][:, idx]
            vals.append(metrics.rmse(y_te, pct(clip(models.predict(mlp, Xp)))) - base)
        imp[gname] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    log["importance"] = imp
    print("importance", imp, flush=True)

    # ---------------- 4) ablacija ulaznih osobina (MLP, 3 sjemena) ----------------
    geo = ["cos_z", "G_0", "f_sun", "w_sin"]
    sets = {
        "G + geometrija": ["G"] + geo,
        "+ komponente zračenja i k_t": ["G", "G_b", "G_d", "G_n", "k_t"] + geo,
        "+ oblačnost i temperatura": ["G", "G_b", "G_d", "G_n", "k_t", "C", "T_a"] + geo,
        "sve osim susjednih sati": [c for c in prep.FEATURES if c not in ("G_lag1", "G_lead1")],
        "sve osobine (osnovni model)": prep.FEATURES,
    }
    log["ablation"] = {}
    for nm, cols in sets.items():
        idx = [prep.FEATURES.index(c) for c in cols]
        r = []
        for seed in SEEDS[:3]:
            _, m2, _, _ = fit_nn("MLP", best_mlp, D, S, seed, idx=idx)
            r.append(metrics.rmse(y_te, pct(clip(models.predict(m2, te["X"][:, idx])))))
        log["ablation"][nm] = {"RMSE_mean": float(np.mean(r)), "RMSE_std": float(np.std(r, ddof=1)),
                               "n_features": len(idx)}
        print("ablacija", nm, np.mean(r), flush=True)

    # ---------------- 5) snimanje ----------------
    out = pd.DataFrame({"time": df.index[te["rows"]], "y": y_te, "capacity_MW": cap_te, **preds,
                        "G": df["G"].values[te["rows"]], "C": C})
    out.to_csv(RES / "pred_test.csv", index=False)
    torch.save(sel["MLP"]["model"].state_dict(), RES / "mlp.pt")
    torch.save(sel["LSTM"]["model"].state_dict(), RES / "lstm.pt")
    import pickle
    with open(RES / "baselines.pkl", "wb") as f:
        pickle.dump({"ridge": ridge, "gbm": gbm, "scaler": D["scaler"]}, f)
    with open(RES / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=1, default=float)
    print(json.dumps(log["test"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
