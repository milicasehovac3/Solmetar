"""data.py - matrice osobina, skaliranje (parametri samo iz skupa za treniranje) i sekvence za LSTM."""
import numpy as np
from sklearn.preprocessing import StandardScaler
from . import prep

WINDOW = 24  # w: duzina ulaznog prozora LSTM mreze (sati)


def seq_valid(df):
    """Uzorak t ima valjan prozor ako svih w sati [t-w+1, t] ima potpune osobine."""
    row_ok = df[prep.SEQ_FEATURES].notna().all(axis=1).values.astype(int)
    return np.convolve(row_ok, np.ones(WINDOW, dtype=int), "full")[:len(df)] == WINDOW


def transform(df, scaler):
    """Standardizovane osobine (NaN ostaju NaN) i izvor za sekvence (NaN -> 0)."""
    Xs = (df[prep.FEATURES].values - scaler.mean_) / scaler.scale_
    seq_idx = [prep.FEATURES.index(c) for c in prep.SEQ_FEATURES]
    return Xs, np.nan_to_num(Xs[:, seq_idx])


def build():
    df = prep.load_limburg()
    tr, va, te = prep.split(df)
    day = prep.daytime(df)
    common = prep.valid_rows(df, prep.FEATURES + [prep.TARGET]) & seq_valid(df) & day

    scaler = StandardScaler()
    scaler.fit(df.loc[tr & day & prep.valid_rows(df, prep.FEATURES), prep.FEATURES].values)
    Xs, Xseq_src = transform(df, scaler)

    D = {"df": df, "scaler": scaler, "Xs": Xs, "Xseq_src": Xseq_src}
    for name, m in (("train", tr), ("val", va), ("test", te), ("trval", tr | va)):
        rows = np.where(common & m)[0]
        D[name] = {"rows": rows, "X": Xs[rows].astype(np.float32),
                   "y": df[prep.TARGET].values[rows].astype(np.float32)}
    return D


def sequences(src, rows):
    idx = rows[:, None] + np.arange(-WINDOW + 1, 1)[None, :]
    return src[idx].astype(np.float32)
