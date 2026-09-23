"""
export_app.py - izvoz modela i podataka za pokaznu aplikaciju (app/solmetar.html).

Izvoze se: parametri standardizacije, tezine linearne regresije, MLP i LSTM mreze,
sirovi satni meteoroloski podaci (Open-Meteo) za Limburg i sedam gradova u BiH od
1.1.2021. do dana izgradnje (kvantizovani cijeli brojevi (int32), gzip + base64), izmjereni faktor
iskoristenja i prognoza operatora za Limburg, godisnji rezultati prenosa modela (PVGIS),
te referentne Python procjene za provjeru JavaScript implementacije.

Pokretanje:  python -m solar_nn.export_app
"""
import base64
import gzip
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from . import prep, data, models, transfer

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
T0, T1 = "2021-01-01 00:00", "2026-09-23 23:00"
# kvantizacija: vrijednost = cijeli broj / skala (tacno za prosjek cetiri tacke u Limburgu)
QVARS = ["G", "G_b", "G_d", "G_n", "T_a", "RH", "v_w", "R", "S", "C"]
QSCALE = [40, 40, 40, 40, 40, 4, 40, 40, 400, 4]


def b64gz(arr):
    return base64.b64encode(gzip.compress(arr.tobytes(), 9, mtime=0)).decode()


def raw_block(meteo, name):
    if name == "Limburg":
        raw = meteo[meteo["location"].str.startswith("Limburg")].groupby("time").mean(numeric_only=True)
    else:
        raw = meteo[meteo["location"] == name].set_index("time").drop(columns="location")
    idx = pd.date_range(T0, T1, freq="h")
    raw = raw.reindex(idx)
    assert raw[QVARS].notna().all().all(), name
    return raw[QVARS]


def quantize(raw):
    q = np.zeros((len(QVARS), len(raw)), dtype=np.int32)
    deq = raw.copy()
    for i, (c, s) in enumerate(zip(QVARS, QSCALE)):
        v = np.round(raw[c].values * s)
        err = np.abs(v / s - raw[c].values).max()
        assert err < 1e-6, (c, err)
        q[i] = v.astype(np.int32)
        deq[c] = v / s
    return q, deq


def main():
    L = json.load(open(RES / "metrics.json", encoding="utf-8"))
    TR = json.load(open(RES / "transfer.json", encoding="utf-8"))
    M = transfer.load_models()
    meteo = prep.load_meteo()
    b = L["best"]
    sd = lambda m: {k: v.detach().numpy().astype(float).tolist() for k, v in m.state_dict().items()}

    locs = {}
    ref = {}
    for name in ["Limburg"] + list(prep.CITIES):
        raw = raw_block(meteo, name)
        q, deq = quantize(raw)
        lat, lon = prep.LIMBURG if name == "Limburg" else prep.CITIES[name]
        locs[name] = {"label": prep.CITY_LABEL.get(name, name), "lat": lat, "lon": lon, "meteo": b64gz(q)}
        # referentne procjene iz dekvantizovanih podataka (isti ulaz kao u aplikaciji)
        df = prep.features(prep.align_meteo(deq), lat, lon)
        p = transfer.predict_frame(df, M)
        if name == "Limburg":
            sel = (df.index >= prep.VAL_END) & (df.index < prep.TEST_END)
        else:
            sel = (df.index >= "2026-06-01") & (df.index < "2026-06-08")
        ref[name] = {"t0": str(df.index[sel][0]), "n": int(sel.sum()),
                     "ridge": p["Linearna regresija"].values[sel].round(7).tolist(),
                     "mlp": p["MLP"].values[sel].round(7).tolist(),
                     "lstm": p["LSTM"].values[sel].round(7).tolist(),
                     "cos_z": df["cos_z"].values[sel].round(9).tolist(),
                     "k_t": df["k_t"].values[sel].round(9).tolist()}

    # Limburg: izmjereni kappa, prognoza operatora i instalisana snaga
    lim = prep.load_limburg(meteo).reindex(pd.date_range(T0, T1, freq="h"))
    enc = lambda s: np.where(s.notna(), np.round(s.fillna(0).values * 10000), -1).astype(np.int16)
    cap = lim["capacity_MW"].ffill()
    ch = np.r_[0, np.where(np.diff(cap.values) != 0)[0] + 1]
    locs["Limburg"].update({"kappa": b64gz(enc(lim["kappa"])), "kappa_da": b64gz(enc(lim["kappa_da"])),
                            "capacity": [[int(i), float(cap.values[i])] for i in ch]})

    te = pd.read_csv(RES / "pred_test.csv", parse_dates=["time"])
    app = {
        "t0": T0, "n_hours": int(len(pd.date_range(T0, T1, freq="h"))), "window": data.WINDOW,
        "qvars": QVARS, "qscale": QSCALE, "features": prep.FEATURES, "seq_features": prep.SEQ_FEATURES,
        "scaler_mean": M["scaler"].mean_.tolist(), "scaler_scale": M["scaler"].scale_.tolist(),
        "ridge": {"coef": M["ridge"].coef_.astype(float).tolist(), "intercept": float(M["ridge"].intercept_)},
        "mlp": {"hidden": b["mlp"]["hidden"], "state": sd(M["mlp"])},
        "lstm": {"hidden": b["lstm"]["hidden"], "layers": b["lstm"]["layers"], "state": sd(M["lstm"])},
        "periods": {"train": [prep.TRAIN_START, prep.TRAIN_END], "val": [prep.TRAIN_END, prep.VAL_END],
                    "test": [prep.VAL_END, prep.TEST_END]},
        "test_rows": [t.strftime("%Y-%m-%d %H") for t in te["time"]][:1] + [str(len(te))],
        "metrics_python": {k: L["test"][k] for k in ("Linearna regresija", "Gradijentno pojačanje", "MLP", "LSTM",
                                                     "Elia, dan unaprijed", "Persistencija (t-24)")},
        "transfer": {c: {"PVGIS_E_y": TR["locations"][c]["PVGIS_E_y"], "PVGIS_E_m": TR["locations"][c]["PVGIS_E_m"],
                         "MLP": TR["locations"][c]["MLP"]["annual"], "LSTM": TR["locations"][c]["LSTM"]["annual"]}
                     for c in TR["locations"]},
        "q_Limburg": TR["q_Limburg"],
        "locations": locs,
    }
    with open(RES / "app_model.json", "w", encoding="utf-8") as f:
        json.dump(app, f, separators=(",", ":"), ensure_ascii=False)
    with open(RES / "python_reference.json", "w") as f:
        json.dump(ref, f, separators=(",", ":"))
    print("app_model.json", round((RES / "app_model.json").stat().st_size / 1e6, 2), "MB")


if __name__ == "__main__":
    main()
