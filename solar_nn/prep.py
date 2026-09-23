"""
prep.py - ucitavanje i priprema podataka (satna rezolucija, UTC).

Izvori:
  data/v2/limburg_pv.csv  - Elia Open Data (skup ods032): izmjerena proizvodnja fotonaponskih
                            elektrana u provinciji Limburg (Belgija), pracena instalisana snaga i
                            prognoze operatora; 15-minutne vrijednosti agregirane na sat [k, k+1).
  data/v2/meteo_all.csv(.gz) - Open-Meteo Historical Forecast API: satni meteoroloski podaci za cetiri
                            tacke u Limburgu i sedam gradova u BiH (2021-01-01 do danas).
  data/v2/bih_pvgis.csv   - PVGIS 5.3 (JRC): prosjecna godisnja i mjesecna proizvodnja 1 kWp
                            sistema pod optimalnim uglom (SARAH3, 2005-2023, gubici 14 %).

Ciljna varijabla je faktor iskoristenja kappa = P / P_inst (udio instalisane snage), sto model
cini nezavisnim od velicine sistema i omogucava primjenu na drugim lokacijama.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "v2"

# lokacije: (geografska sirina, duzina) za solarnu geometriju
LIMBURG = (51.00, 5.45)
CITIES = {
    "Travnik": (44.227, 17.665), "Sarajevo": (43.856, 18.413), "Mostar": (43.343, 17.808),
    "Banja Luka": (44.772, 17.191), "Tuzla": (44.538, 18.667), "Bihac": (44.816, 15.870),
    "Trebinje": (42.712, 18.343),
}
CITY_LABEL = {"Bihac": "Bihać"}

TRAIN_START = "2021-01-01"
TRAIN_END = "2024-07-01"   # trening: 2021-01-01 .. 2024-06-30
VAL_END = "2025-07-01"     # validacija: 2024-07-01 .. 2025-06-30
TEST_END = "2026-09-22"    # test: 2025-07-01 .. 2026-09-21

# ulazne osobine (sve nezavisne od lokacije i velicine sistema)
METEO = ["G", "G_b", "G_d", "G_n", "T_a", "RH", "v_w", "R", "S", "C"]
GEOM = ["cos_z", "G_0", "k_t", "f_sun", "w_sin"]
DERIVED = ["S_72", "G_lag1", "G_lead1"]
FEATURES = METEO + GEOM + DERIVED               # 18 osobina za staticke modele
SEQ_FEATURES = METEO + GEOM                     # 15 osobina po koraku za LSTM
TARGET = "kappa"
G_SC = 1367.0                                   # solarna konstanta, W/m2

RAW_MAP = {"shortwave_radiation": "G", "direct_radiation": "G_b", "diffuse_radiation": "G_d",
           "direct_normal_irradiance": "G_n", "temperature_2m": "T_a", "relative_humidity_2m": "RH",
           "wind_speed_10m": "v_w", "precipitation": "R", "snowfall": "S", "cloud_cover": "C",
           "surface_pressure": "p"}
INTERVAL = ["G", "G_b", "G_d", "G_n", "R", "S"]   # srednja vrijednost / suma prethodnog sata
INSTANT = ["T_a", "RH", "v_w", "C"]               # trenutne vrijednosti


# ------------------------------------------------------------------ solarna geometrija
def declination(doy):
    """Deklinacija Sunca (Cooper, 1969), u radijanima."""
    return np.radians(23.45 * np.sin(np.radians(360.0 / 365.0 * (284 + doy))))


def equation_of_time(doy):
    """Jednacina vremena (Spencer, 1971), u minutama."""
    B = np.radians(360.0 / 365.0 * (doy - 1))
    return 229.18 * (0.000075 + 0.001868 * np.cos(B) - 0.032077 * np.sin(B)
                     - 0.014615 * np.cos(2 * B) - 0.040890 * np.sin(2 * B))


def cos_zenith_utc(t_utc_hours, doy, lat, lon):
    """cos(theta_z) za trenutak zadan u satima UTC (moze biti decimalan)."""
    tst = t_utc_hours + lon / 15.0 + equation_of_time(doy) / 60.0   # pravo solarno vrijeme
    omega = np.radians(15.0 * (tst - 12.0))
    phi, d = np.radians(lat), declination(doy)
    return np.sin(phi) * np.sin(d) + np.cos(phi) * np.cos(d) * np.cos(omega), omega


def geometry(index, lat, lon):
    """Geometrijske osobine za satni interval [k, k+1) UTC: srednji cos(theta_z) i G_0
    (cetiri podtacke), udio sata sa Suncem f_sun i sin(omega) sredine sata."""
    doy = index.dayofyear.values.astype(float)
    hr = index.hour.values.astype(float)
    E0 = 1 + 0.033 * np.cos(np.radians(360.0 * doy / 365.0))
    cz_sum, up = np.zeros(len(index)), np.zeros(len(index))
    for j in range(4):
        cz, _ = cos_zenith_utc(hr + (j + 0.5) / 4.0, doy, lat, lon)
        cz_sum += np.clip(cz, 0, None)
        up += (cz > 0)
    _, om = cos_zenith_utc(hr + 0.5, doy, lat, lon)
    out = pd.DataFrame(index=index)
    out["cos_z"] = cz_sum / 4.0
    out["G_0"] = G_SC * E0 * out["cos_z"]
    out["f_sun"] = up / 4.0
    out["w_sin"] = np.sin(om) * (out["f_sun"] > 0)
    return out


# ------------------------------------------------------------------ meteoroloski podaci
def load_meteo():
    f = DATA / "meteo_all.csv"
    if not f.exists():                      # u repozitoriju je datoteka komprimovana (.csv.gz)
        f = DATA / "meteo_all.csv.gz"
    m = pd.read_csv(f, comment="#", parse_dates=["time"])
    return m.rename(columns=RAW_MAP)


def align_meteo(raw):
    """Poravnanje sa satom proizvodnje [k, k+1): intervalne velicine (zracenje, padavine) iz
    oznake k+1, trenutne velicine kao srednja vrijednost oznaka k i k+1."""
    raw = raw.sort_index()
    out = pd.DataFrame(index=raw.index)
    for c in INTERVAL:
        out[c] = raw[c].shift(-1)
    for c in INSTANT:
        out[c] = 0.5 * (raw[c] + raw[c].shift(-1))
    return out


def features(met, lat, lon):
    """Kompletan skup osobina za jednu lokaciju (met: poravnani satni podaci, indeks UTC)."""
    df = met.join(geometry(met.index, lat, lon))
    g0 = df["G_0"].where(df["G_0"] > 10)
    df["k_t"] = np.clip(df["G"] / g0, 0, 1.2).fillna(0.0)
    df["S_72"] = df["S"].rolling(72, min_periods=1).sum()
    df["G_lag1"] = df["G"].shift(1)
    df["G_lead1"] = df["G"].shift(-1)
    return df


def location_frame(name, meteo=None):
    """Satne osobine za grad u BiH ili za Limburg (prosjek cetiri tacke)."""
    m = load_meteo() if meteo is None else meteo
    if name == "Limburg":
        raw = m[m["location"].str.startswith("Limburg")].groupby("time").mean(numeric_only=True)
        lat, lon = LIMBURG
    else:
        raw = m[m["location"] == name].set_index("time").drop(columns="location")
        lat, lon = CITIES[name]
    return features(align_meteo(raw), lat, lon)


def load_limburg(meteo=None):
    """Limburg: osobine + izmjerena proizvodnja, instalisana snaga i prognoze operatora (kappa)."""
    pv = pd.read_csv(DATA / "limburg_pv.csv", parse_dates=["time"]).set_index("time")
    df = location_frame("Limburg", meteo).join(pv, how="inner")
    cap = df["capacity_MW"]
    df["kappa"] = df["measured_MW"] / cap
    df["kappa_da"] = df["dayahead_MW"] / cap          # prognoza dan unaprijed (Elia)
    df["kappa_da11"] = df["dayahead11h_MW"] / cap     # prognoza dan unaprijed u 11 h
    df["kappa_mr"] = df["mostrecent_MW"] / cap        # posljednja unutardnevna prognoza
    df["kappa_lag24"] = df["kappa"].shift(24)
    df = df[(df.index >= TRAIN_START) & (df.index < TEST_END)]
    return df


def split(df):
    tr = (df.index < TRAIN_END)
    va = (df.index >= TRAIN_END) & (df.index < VAL_END)
    te = (df.index >= VAL_END) & (df.index < TEST_END)
    return tr, va, te


def daytime(df):
    return (df["f_sun"] > 0).values


def valid_rows(df, cols):
    return df[cols].notna().all(axis=1).values


def load_pvgis():
    return pd.read_csv(DATA / "bih_pvgis.csv").set_index("location")
