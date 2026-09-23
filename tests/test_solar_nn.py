"""
test_solar_nn.py - automatizovani testovi programskog rješenja završnog rada.

Provjeravaju formule iz rada, broj parametara mreža, pripremu podataka, metrike,
statistički test, sačuvane modele i podudarnost aplikacije Solmetar sa Pythonom.

Pokretanje (iz korijenskog foldera repozitorija):   python -m unittest -v tests.test_solar_nn
"""
import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from solar_nn import prep, metrics, models, data, transfer

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"


class TestSolarnaGeometrija(unittest.TestCase):
    def test_01_deklinacija(self):
        """Deklinacija (Cooper): oko +23,45° ljeti, oko -23,45° zimi, oko 0° u ravnodnevici."""
        self.assertAlmostEqual(np.degrees(prep.declination(172)), 23.45, delta=0.1)
        self.assertAlmostEqual(np.degrees(prep.declination(355)), -23.45, delta=0.1)
        self.assertLess(abs(np.degrees(prep.declination(81))), 1.0)

    def test_02_jednacina_vremena(self):
        """Jednačina vremena (Spencer) ostaje između -15 i +17 minuta tokom godine."""
        eot = prep.equation_of_time(np.arange(1, 366))
        self.assertGreater(eot.min(), -15.0)
        self.assertLess(eot.max(), 17.0)

    def test_03_geometrija_travnik(self):
        """Travnik 21. juna: podnevna elevacija oko 69°, noću nema Sunca, jutro/popodne po znaku sin(ω)."""
        lat, lon = prep.CITIES["Travnik"]
        idx = pd.date_range("2026-06-21 00:00", periods=24, freq="h")
        g = prep.geometry(idx, lat, lon)
        elev = 90 - np.degrees(np.arccos(g["cos_z"].max()))
        self.assertAlmostEqual(elev, 69.0, delta=1.5)
        self.assertEqual(g["f_sun"].iloc[0], 0.0)          # ponoć UTC
        self.assertLess(g["w_sin"].iloc[6], 0)             # jutro
        self.assertGreater(g["w_sin"].iloc[14], 0)         # popodne
        self.assertTrue(((g["G_0"] >= 0) & (g["G_0"] <= 1410)).all())


class TestOsobine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = prep.location_frame("Sarajevo")

    def test_04_broj_osobina(self):
        """18 osobina za statičke modele i 15 po satu za LSTM, sve prisutne u tabeli."""
        self.assertEqual(len(prep.FEATURES), 18)
        self.assertEqual(len(prep.SEQ_FEATURES), 15)
        for c in prep.FEATURES:
            self.assertIn(c, self.df.columns)

    def test_05_indeks_prozracnosti(self):
        """k_t = G / G_0 je u intervalu [0; 1,2]."""
        kt = self.df["k_t"].dropna()
        self.assertGreaterEqual(kt.min(), 0.0)
        self.assertLessEqual(kt.max(), 1.2)

    def test_06_poravnanje(self):
        """Zračenje sa oznakom t+1 pripada satu t (pomjeraj za jedan sat unazad)."""
        m = prep.load_meteo()
        raw = m[m["location"] == "Sarajevo"].set_index("time").drop(columns="location")
        al = prep.align_meteo(raw)
        t = pd.Timestamp("2025-06-15 10:00")
        self.assertAlmostEqual(al.loc[t, "G"], raw.loc[t + pd.Timedelta(hours=1), "G"])


class TestPodaci(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = prep.load_limburg()

    def test_07_faktor_iskoristenja(self):
        """κ = P / P_nom je u razumnom opsegu za dnevne sate."""
        k = self.df["kappa"].dropna()
        self.assertGreaterEqual(k.min(), 0.0)
        self.assertLess(k.max(), 1.1)

    def test_08_hronoloska_podjela(self):
        """Trening, validacija i test se ne preklapaju i idu hronološki."""
        tr, va, te = prep.split(self.df)
        self.assertFalse((tr & va).any() or (tr & te).any() or (va & te).any())
        self.assertLess(self.df.index[tr].max(), self.df.index[va].min())
        self.assertLess(self.df.index[va].max(), self.df.index[te].min())


class TestModeli(unittest.TestCase):
    def test_09_broj_parametara(self):
        """MLP 18-256-128-64-1 ima 46.081, a LSTM 2 × 32 ima 15.809 parametara (kao u radu)."""
        mlp = models.MLP(18, (256, 128, 64), 0.2)
        lstm = models.LSTMReg(15, 32, 2)
        count = lambda m: sum(p.numel() for p in m.parameters())
        self.assertEqual(count(mlp), 46081)
        self.assertEqual(count(lstm), 15809)

    def test_10_sacuvani_modeli(self):
        """Sačuvani modeli se učitavaju i daju procjene u intervalu [0, 1]."""
        M = transfer.load_models()
        df = prep.location_frame("Travnik")
        df = df[(df.index >= "2026-06-01") & (df.index < "2026-06-08")]
        p = transfer.predict_frame(df, M)
        for c in transfer.MODELS:
            self.assertTrue(((p[c] >= 0) & (p[c] <= 1)).all())
            self.assertGreater(p[c].max(), 0.3)           # ljeti ima proizvodnje


class TestMetrike(unittest.TestCase):
    def test_11_metrike(self):
        """MAE, RMSE, MBE i R² na jednostavnom primjeru."""
        y = np.array([0.0, 0.5, 1.0])
        yh = np.array([0.1, 0.5, 0.8])
        self.assertAlmostEqual(metrics.mae(y, yh), 0.1)
        self.assertAlmostEqual(metrics.rmse(y, yh), np.sqrt(0.05 / 3))
        self.assertAlmostEqual(metrics.mbe(y, yh), -0.1 / 3)
        self.assertAlmostEqual(metrics.r2(y, y), 1.0)

    def test_12_diebold_mariano(self):
        """DM test: tačniji drugi model daje pozitivnu statistiku i malu p-vrijednost."""
        rng = np.random.default_rng(0)
        y = rng.random(2000)
        s, p = metrics.diebold_mariano(y, y + rng.normal(0, 0.2, 2000), y + rng.normal(0, 0.05, 2000))
        self.assertGreater(s, 0)
        self.assertLess(p, 0.001)


class TestRezultatiIAplikacija(unittest.TestCase):
    def test_13_rezultati_iz_rada(self):
        """Sačuvane metrike odgovaraju Tabeli 5 u radu (RMSE MLP 5,69 %, LSTM 5,59 %)."""
        L = json.loads((RES / "metrics.json").read_text(encoding="utf-8"))
        self.assertAlmostEqual(L["test"]["MLP"]["RMSE"], 5.69, delta=0.01)
        self.assertAlmostEqual(L["test"]["LSTM"]["RMSE"], 5.59, delta=0.01)

    def test_14_aplikacija_jednaka_pythonu(self):
        """JavaScript implementacija (app/verify.js) odstupa od Pythona manje od 1e-6."""
        v = json.loads((RES / "js_verification.json").read_text(encoding="utf-8"))
        for loc, r in v.items():
            if loc == "seconds":
                continue
            for m in ("ridge", "mlp", "lstm"):
                self.assertLess(r[m], 1e-6, f"{loc} {m}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
