"""figures.py - sve slike rada (generisu se iz podataka i results/metrics.json)."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle

from . import prep

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
RES = ROOT / "results"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Liberation Serif", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 10, "axes.titlesize": 10.5, "axes.labelsize": 10,
    "legend.fontsize": 8.5, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 220,
    "savefig.bbox": "tight", "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
})
COL = {"P": "#1f1f1f", "Linearna regresija": "#c0772b", "Gradijentno pojačanje": "#6a8e3a",
       "MLP": "#2b5c9e", "LSTM": "#9e2b4d", "Persistencija (t-1)": "#8c8c8c",
       "Persistencija (t-24)": "#b5b5b5"}
MONTHS = ["jan", "feb", "mar", "apr", "maj", "jun", "jul", "avg", "sep", "okt", "nov", "dec"]


def _comma(fig):
    fmt = matplotlib.ticker.FuncFormatter(lambda v, _: (f"{v:g}").replace(".", ","))
    for a in fig.axes:
        if a.get_xscale() == "linear" and not isinstance(a.xaxis.get_major_formatter(), matplotlib.dates.DateFormatter) \
                and a.xaxis.get_major_formatter().__class__.__name__ == "ScalarFormatter":
            a.xaxis.set_major_formatter(fmt)
        if a.get_yscale() == "linear" and a.yaxis.get_major_formatter().__class__.__name__ == "ScalarFormatter":
            a.yaxis.set_major_formatter(fmt)


def save(fig, name):
    _comma(fig)
    fig.savefig(FIG / name)
    plt.close(fig)


# ------------------------------------------------------------------ teorija
def fig_solar_geometry():
    """Elevacija Sunca i G_0 za Travnik (phi = 44,23 N) za karakteristicne dane (formule 1-4)."""
    phi = np.radians(44.23)
    H = np.linspace(4, 20, 400)
    fig, ax = plt.subplots(1, 2, figsize=(6.3, 2.6))
    for D, lab, ls in ((172, "21. jun (D = 172)", "-"), (80, "21. mart (D = 80)", "--"),
                       (355, "21. decembar (D = 355)", ":")):
        delta = np.radians(23.45 * np.sin(np.radians(360 / 365 * (284 + D))))
        omega = np.radians(15 * (H - 12))
        cz = np.sin(phi) * np.sin(delta) + np.cos(phi) * np.cos(delta) * np.cos(omega)
        alpha = np.degrees(np.arcsin(np.clip(cz, -1, 1)))
        G0 = 1367 * (1 + 0.033 * np.cos(np.radians(360 * D / 365))) * np.clip(cz, 0, None)
        ax[0].plot(H, np.clip(alpha, 0, None), "k", ls=ls, lw=1.2, label=lab)
        ax[1].plot(H, G0, "k", ls=ls, lw=1.2, label=lab)
    ax[0].set(xlabel="solarno vrijeme H (h)", ylabel=r"elevacija $\alpha_s$ (°)", title="(a) Elevacija Sunca")
    ax[1].set(xlabel="solarno vrijeme H (h)", ylabel=r"$G_0$ (W/m²)", title="(b) Vanatmosfersko zračenje")
    ax[1].legend(frameon=False, loc="upper right", fontsize=7.5)
    ax[1].set_ylim(0, 1550)
    fig.tight_layout()
    save(fig, "slika_01_solarna_geometrija.png")


def fig_pv_model():
    """Staticki model snage: P u zavisnosti od G za razlicite T_a (formule 5 i 6)."""
    G = np.linspace(0, 1000, 200)
    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    for Ta, ls in ((0, ":"), (15, "--"), (30, "-")):
        Tc = Ta + (45 - 20) / 800 * G
        Pw = 5 * G / 1000 * (1 - 0.004 * (Tc - 25)) * 0.96
        ax.plot(G, Pw, "k", ls=ls, lw=1.2, label=rf"$T_a$ = {Ta} °C")
    ax.set(xlabel="iradijansa G (W/m²)", ylabel="snaga P (kW)",
           title=r"Model (6)-(7): $P_{nom}$ = 5 kWp, $\gamma$ = -0,4 %/°C, $k_s$ = 0,96")
    ax.legend(frameon=False)
    save(fig, "slika_02_model_snage.png")


def box(ax, x, y, w, h, text, fc="#f2f2f2", fs=8.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec="#333", lw=0.9))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


def arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", lw=0.9, color="#333"))


def fig_neuron():
    fig, ax = plt.subplots(figsize=(5.4, 2.4))
    ax.set_xlim(0, 10); ax.set_ylim(0, 4.4); ax.axis("off"); ax.set_aspect("equal")
    for i, lab in enumerate([r"$x_1$", r"$x_2$", r"$\vdots$", r"$x_K$"]):
        y = 3.6 - i * 1.0
        if lab != r"$\vdots$":
            ax.add_patch(Circle((1, y), 0.28, fc="white", ec="#333"))
            arrow(ax, 1.3, y, 4.2, 2.2)
            ax.text(2.1, y + (2.2 - y) * 0.30 + 0.18, rf"$w_{{{['1','2','','K'][i]}}}$", fontsize=9)
        ax.text(1, y, lab, ha="center", va="center", fontsize=10)
    ax.add_patch(Circle((4.8, 2.2), 0.6, fc="#f2f2f2", ec="#333"))
    ax.text(4.8, 2.2, r"$\Sigma$", ha="center", va="center", fontsize=13)
    ax.text(4.8, 0.95, r"$z=\mathbf{w}^{\top}\mathbf{x}+b$", ha="center", fontsize=9)
    arrow(ax, 4.8, 3.6, 4.8, 2.85); ax.text(4.8, 3.75, "b", ha="center", fontsize=10)
    box(ax, 6.2, 1.8, 1.4, 0.8, r"$g(z)$")
    arrow(ax, 5.4, 2.2, 6.2, 2.2)
    arrow(ax, 7.6, 2.2, 8.9, 2.2)
    ax.text(9.25, 2.2, r"$a$", ha="center", va="center", fontsize=11)
    save(fig, "slika_03_neuron.png")


def fig_mlp(hidden):
    layers = [len(prep.FEATURES)] + list(hidden) + [1]
    names = [f"ulazni sloj\n$K$ = {len(prep.FEATURES)}"] + [f"skriveni sloj {i+1}\n{h} neurona" for i, h in enumerate(hidden)] + ["izlazni sloj\n$\\hat{y}_t = \\hat{\\kappa}_t$"]
    fig, ax = plt.subplots(figsize=(6.2, 3.3))
    ax.axis("off")
    xs = np.linspace(0.6, 9.4, len(layers))
    pos = []
    for n in layers:
        if n == 1:
            pos.append(np.array([2.5]))
        else:
            ys = np.linspace(0.7, 4.3, 7)
            pos.append(np.concatenate([ys[:3], ys[4:]]))
    for i in range(len(layers) - 1):
        for y1 in pos[i]:
            for y2 in pos[i + 1]:
                ax.plot([xs[i], xs[i + 1]], [y1, y2], color="#aaa", lw=0.35, zorder=1)
    for i, (x, ys) in enumerate(zip(xs, pos)):
        for y in ys:
            ax.add_patch(Circle((x, y), 0.2, fc="white" if i in (0, len(xs) - 1) else "#dde6f2", ec="#333", lw=0.8, zorder=2))
        if layers[i] > 1:
            for dy in (-0.15, 0, 0.15):
                ax.add_patch(Circle((x, 2.5 + dy), 0.035, fc="#333", ec="none", zorder=3))
        ax.text(x, 0.2, names[i], ha="center", va="top", fontsize=8)
    ax.text(5, 4.8, "ReLU + dropout u skrivenim slojevima, linearni izlaz", ha="center", fontsize=8.5)
    ax.set_xlim(0, 10); ax.set_ylim(-0.8, 5.0); ax.set_aspect("equal")
    save(fig, "slika_04_mlp.png")


def fig_lstm_cell():
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    ax.set_xlim(0, 12); ax.set_ylim(0, 6); ax.axis("off")
    ax.add_patch(FancyBboxPatch((1.5, 0.6), 8.6, 4.6, boxstyle="round,pad=0.05,rounding_size=0.3", fc="#fafafa", ec="#333"))
    ax.plot([0.3, 11.6], [4.5, 4.5], color="#333", lw=1.4)
    ax.text(0.2, 4.75, r"$\mathbf{c}_{t-1}$", fontsize=10); ax.text(10.8, 4.75, r"$\mathbf{c}_t$", fontsize=10)
    ax.plot([0.3, 2.2], [1.2, 1.2], color="#333", lw=1.0); ax.text(0.2, 0.75, r"$\mathbf{h}_{t-1}$", fontsize=10)
    ax.text(2.2, 0.1, r"$\mathbf{x}_t$", fontsize=10); ax.plot([2.4, 2.4], [0.45, 1.2], color="#333", lw=1.0)
    gates = [(2.6, r"$\sigma$", r"$\mathbf{f}_t$"), (4.3, r"$\sigma$", r"$\mathbf{i}_t$"),
             (5.9, "tanh", r"$\tilde{\mathbf{c}}_t$"), (7.6, r"$\sigma$", r"$\mathbf{o}_t$")]
    ax.plot([2.2, 8.2], [1.2, 1.2], color="#333", lw=1.0)
    for x, f, lab in gates:
        box(ax, x, 1.6, 1.1, 0.7, f, fc="#dde6f2", fs=9.5)
        ax.plot([x + 0.55, x + 0.55], [1.2, 1.6], color="#333", lw=0.9)
        ax.text(x + 0.62, 2.45, lab, fontsize=9.5)
    for x, s in ((3.15, "×"), (5.6, "+"), (9.2, "×")):
        ax.add_patch(Circle((x, 4.5), 0.28, fc="white", ec="#333", zorder=3))
        ax.text(x, 4.5, s, ha="center", va="center", fontsize=11, zorder=4)
    arrow(ax, 3.15, 2.3, 3.15, 4.2)
    ax.add_patch(Circle((5.6, 3.3), 0.26, fc="white", ec="#333", zorder=3)); ax.text(5.6, 3.3, "×", ha="center", va="center", fontsize=11, zorder=4)
    arrow(ax, 4.85, 2.3, 5.4, 3.15); arrow(ax, 6.45, 2.3, 5.8, 3.15); arrow(ax, 5.6, 3.56, 5.6, 4.22)
    box(ax, 8.6, 3.2, 1.2, 0.6, "tanh", fs=9)
    arrow(ax, 9.2, 4.22, 9.2, 3.8)
    ax.add_patch(Circle((9.2, 2.6), 0.26, fc="white", ec="#333", zorder=3)); ax.text(9.2, 2.6, "×", ha="center", va="center", fontsize=11, zorder=4)
    arrow(ax, 9.2, 3.2, 9.2, 2.86); arrow(ax, 8.7, 1.95, 8.95, 2.45)
    ax.plot([9.46, 11.6], [2.6, 2.6], color="#333", lw=1.0); ax.text(10.8, 2.15, r"$\mathbf{h}_t$", fontsize=10)
    save(fig, "slika_05_lstm_celija.png")


def fig_pipeline():
    fig, ax = plt.subplots(figsize=(6.3, 3.0))
    ax.set_xlim(0, 12.4); ax.set_ylim(0, 5.6); ax.axis("off")
    steps = ["Elia (Limburg)\n+ Open-Meteo\n2021-2026", "satna agregacija,\nporavnanje, $\\kappa = P/P_{nom}$", "solarna geometrija\ni izvedene osobine\n($\\cos\\theta_z$, $G_0$, $k_t$, $S_{72}$)",
             "hronološka podjela\n(trening / validacija /\ntest)", "izbor hiperparametara\n(validacioni skup)", "ponovno treniranje na\ntreningu + validaciji\n(5 sjemena)",
             "evaluacija na testu\n(7/2025 - 9/2026,\nDM test, Elia)", "primjena na gradove\nu BiH, poređenje\nsa PVGIS; aplikacija"]
    w, h = 2.6, 1.6
    for i, s_ in enumerate(steps):
        r, c = divmod(i, 4)
        x = 0.1 + c * 3.1
        y = 3.5 - r * 2.9
        box(ax, x, y, w, h, s_, fs=7.8)
        if c < 3:
            arrow(ax, x + w, y + h / 2, x + 3.1, y + h / 2)
    arrow(ax, 9.3 + w / 2, 3.5, 0.1 + w / 2, 0.6 + h)
    save(fig, "slika_06_tok_rada.png")


# ------------------------------------------------------------------ podaci
COL.update({"Elia, dan unaprijed": "#2e7d4f", "Elia, dan unaprijed (11 h)": "#7fb08f", "Elia, unutardnevna": "#a6d4b5",
            "y": "#1f1f1f"})


def fig_data(df):
    tr, va, te = prep.split(df)
    daily = df["kappa"].resample("D").sum(min_count=20)
    fig, ax = plt.subplots(figsize=(6.3, 2.6))
    for m, c, lab in ((tr, "#9aa9bf", "trening"), (va, "#c0772b", "validacija"), (te, "#2b5c9e", "test")):
        d = daily[daily.index.isin(df.index[m].normalize())]
        ax.plot(d.index, d.values, lw=0.5, color=c, label=lab)
    ax.set_ylabel("dnevna energija (kWh/kWp)")
    ax2 = ax.twinx(); ax2.spines["right"].set_visible(True); ax2.grid(False)
    ax2.plot(df.index, df["capacity_MW"], color="#333", lw=1.0, ls="--", label="$P_{nom}$ (MW)")
    ax2.set_ylabel("$P_{nom}$ (MW)"); ax2.set_ylim(0, 1800)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 0.98))
    save(fig, "slika_07_podjela_skupa.png")

    wk = df.loc["2023-06-12":"2023-06-18 23:00"]
    fig, ax = plt.subplots(figsize=(6.3, 2.5))
    ax.plot(wk.index, 100 * wk["kappa"], "k", lw=1.1, label=r"$\kappa$ (%)")
    ax2 = ax.twinx(); ax2.spines["right"].set_visible(True); ax2.grid(False)
    ax2.plot(wk.index, wk["G"], color="#c0772b", lw=0.9, ls="--", label="G (W/m²)")
    ax2.plot(wk.index, wk["G_d"], color="#6a8e3a", lw=0.8, ls="-.", label="$G_d$ (W/m²)")
    ax2.plot(wk.index, 5 * wk["C"], color="#2b5c9e", lw=0.7, ls=":", label="5·C (%)")
    ax.set_ylabel(r"$\kappa$ (% $P_{nom}$)"); ax2.set_ylabel("W/m²  /  5·C")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%d.%m."))
    save(fig, "slika_08_sedmica.png")

    day = df[(df["f_sun"] > 0) & df["kappa"].notna() & (df.index < prep.VAL_END)]
    fig, ax = plt.subplots(figsize=(6.3, 2.8))
    for mo, c, lab in (([12, 1, 2], "#2b5c9e", "zima (dec-feb)"), ([3, 4, 5, 9, 10, 11], "#999999", "proljeće/jesen"),
                       ([6, 7, 8], "#c0772b", "ljeto (jun-avg)")):
        s = day[day.index.month.isin(mo)].sample(frac=0.25, random_state=0)
        ax.scatter(s["G"], 100 * s["kappa"], s=2, alpha=0.35, color=c, label=lab, rasterized=True)
    ax.set(xlabel="globalno horizontalno zračenje G (W/m²)", ylabel=r"$\kappa$ (% $P_{nom}$)")
    ax.legend(frameon=False, markerscale=5)
    save(fig, "slika_09_G_vs_P.png")

    fig, ax = plt.subplots(1, 2, figsize=(6.3, 2.4))
    d2 = df[df.index < prep.VAL_END]
    prof = d2.groupby(d2.index.hour)["kappa"].mean() * 100
    ax[0].bar(prof.index, prof.values, color="#9aa9bf", ec="#333", lw=0.4)
    ax[0].set(xlabel="sat (UTC)", ylabel=r"srednji $\kappa$ (%)", title="(a) Dnevni profil")
    mon = d2["kappa"].resample("MS").sum(min_count=500)
    mon = mon.groupby(mon.index.month).mean()
    ax[1].bar(range(1, 13), mon.values, color="#9aa9bf", ec="#333", lw=0.4)
    ax[1].set_xticks(range(1, 13)); ax[1].set_xticklabels([m[0].upper() for m in MONTHS])
    ax[1].set(ylabel="kWh/kWp po mjesecu", title="(b) Srednja mjesečna energija")
    fig.tight_layout()
    save(fig, "slika_10_profili.png")

    # promjena odnosa proizvodnje i zracenja kroz vrijeme
    m = df[["kappa", "G"]].dropna()
    r = m.resample("MS").sum()
    ratio = r["kappa"] / (r["G"] / 1000)
    fig, ax = plt.subplots(figsize=(6.3, 2.5))
    for a_, b_, c in ((prep.TRAIN_START, prep.TRAIN_END, "#eef1f6"), (prep.TRAIN_END, prep.VAL_END, "#f8efe5"), (prep.VAL_END, prep.TEST_END, "#e7eef8")):
        ax.axvspan(pd.Timestamp(a_), pd.Timestamp(b_), color=c, lw=0)
    ax.plot(ratio.index, ratio.values, "k", marker="o", ms=2.5, lw=1)
    summer = ratio[ratio.index.month.isin([5, 6, 7, 8])]
    ax.plot(summer.index, summer.values, "o", ms=3.5, color="#c0772b", label="maj-avgust")
    ax.set_ylabel("kWh/kWp po kWh/m²")
    ax.text(pd.Timestamp("2022-09-01"), 1.06, "trening", ha="center", fontsize=8)
    ax.text(pd.Timestamp("2025-01-01"), 1.06, "validacija", ha="center", fontsize=8)
    ax.text(pd.Timestamp("2026-02-01"), 1.06, "test", ha="center", fontsize=8)
    ax.set_ylim(0.6, 1.1); ax.legend(frameon=False, loc="lower left")
    save(fig, "slika_11_odnos_kroz_vrijeme.png")


def fig_results(log, pred):
    h = log["history"]
    fig, ax = plt.subplots(1, 2, figsize=(6.3, 2.5), sharey=False)
    for i, k in enumerate(("MLP", "LSTM")):
        ax[i].plot(np.arange(1, len(h[k]["train"]) + 1), h[k]["train"], color="#555", lw=1.1, label="trening")
        ax[i].plot(np.arange(1, len(h[k]["val"]) + 1), h[k]["val"], color=COL[k], lw=1.1, ls="--", label="validacija")
        ax[i].axvline(h[k]["best_epoch"], color="#999", lw=0.8, ls=":")
        ax[i].set(yscale="log", xlabel="epoha", ylabel=r"MSE ($\kappa$)", title=f"({'ab'[i]}) {k}")
        ax[i].legend(frameon=False)
    fig.tight_layout()
    save(fig, "slika_12_krive_ucenja.png")

    order = ["Persistencija (t-24)", "Linearna regresija", "Gradijentno pojačanje", "MLP", "LSTM",
             "Elia, dan unaprijed", "Elia, unutardnevna"]
    labels = ["Pers.\n(t-24)", "Linearna\nregresija", "Gradij.\npojačanje", "MLP", "LSTM", "Elia, dan\nunaprijed", "Elia,\nunutardn."]
    t = log["test"]
    fig, ax = plt.subplots(figsize=(6.3, 2.7))
    x = np.arange(len(order)); w = 0.38
    ax.bar(x - w / 2, [t[k]["MAE"] for k in order], w, color="#bbbbbb", ec="#333", lw=0.4, label="MAE")
    ax.bar(x + w / 2, [t[k]["RMSE"] for k in order], w, color=["#2b5c9e" if "Elia" not in k else "#2e7d4f" for k in order], ec="#333", lw=0.4, label="RMSE")
    for i, k in enumerate(order):
        ax.text(i + w / 2, t[k]["RMSE"] + 0.2, f"{t[k]['RMSE']:.2f}".replace(".", ","), ha="center", fontsize=7.5)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7.8)
    ax.set_ylabel(r"greška (% $P_{nom}$)"); ax.legend(frameon=False)
    save(fig, "slika_13_poredjenje.png")

    fig, axs = plt.subplots(1, 3, figsize=(6.3, 2.35), sharex=True, sharey=True)
    for a, k in zip(axs, ["Linearna regresija", "MLP", "LSTM"]):
        a.scatter(pred["y"], pred[k], s=1.5, alpha=0.3, color=COL[k], rasterized=True)
        a.plot([0, 80], [0, 80], "k", lw=0.8)
        a.set(title=f"{k}\n$R^2$ = {t[k]['R2']:.3f}".replace(".", ","), xlabel=r"izmjereni $\kappa$ (%)")
        a.set_aspect("equal")
    axs[0].set_ylabel(r"procjena $\hat{\kappa}$ (%)")
    fig.tight_layout()
    save(fig, "slika_14_scatter.png")

    p = pred.set_index("time")
    full = pd.DataFrame(index=pd.date_range("2026-06-08", "2026-06-14 23:00", freq="h")).join(p)
    fig, ax = plt.subplots(figsize=(6.3, 2.6))
    ax.plot(full.index, full["y"].fillna(0), "k", lw=1.3, label="izmjereno")
    for k, ls in (("Linearna regresija", ":"), ("MLP", "--"), ("LSTM", "-."), ("Elia, dan unaprijed", "-")):
        ax.plot(full.index, full[k].fillna(0), color=COL[k], lw=0.9, ls=ls, label=k)
    ax.set_ylabel(r"$\kappa$ (% $P_{nom}$)"); ax.legend(frameon=False, ncol=5, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=7.3)
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%d.%m."))
    save(fig, "slika_15_sedmica_test.png")

    bc = log["by_cloud"]
    classes = ["vedro (C < 30 %)", "djelimično oblačno", "oblačno (C > 70 %)"]
    models_ = ["Linearna regresija", "Gradijentno pojačanje", "MLP", "LSTM", "Elia, dan unaprijed"]
    fig, ax = plt.subplots(1, 2, figsize=(6.3, 2.6))
    x = np.arange(3); w = 0.16
    for i, k in enumerate(models_):
        ax[0].bar(x + (i - 2) * w, [bc[k][c]["RMSE"] for c in classes], w, color=COL[k], label=k, ec="#333", lw=0.3)
    ax[0].set_xticks(x); ax[0].set_xticklabels(["vedro", "djelimično\noblačno", "oblačno"])
    ax[0].set(ylabel=r"RMSE (% $P_{nom}$)", title="(a) Po klasama oblačnosti")
    bm = log["by_month"]
    mo = [7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6]
    for k in models_:
        ax[1].plot(range(12), [bm[k][str(m)] for m in mo], marker="o", ms=2.5, lw=1, color=COL[k], label=k)
    ax[1].set_xticks(range(12)); ax[1].set_xticklabels([MONTHS[m - 1][0].upper() for m in mo])
    ax[1].set(ylabel=r"RMSE (% $P_{nom}$)", title="(b) Po mjesecima")
    fig.legend(*ax[0].get_legend_handles_labels(), frameon=False, fontsize=7.3, ncol=5, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    save(fig, "slika_16_greske.png")

    imp = log["importance"]
    lab = {"G": "$G$", "G_b, G_d, G_n": "$G_b$, $G_d$, $G_n$", "G(t-1), G(t+1)": "$G_{t-1}$, $G_{t+1}$", "k_t": "$k_t$",
           "C": "$C$", "T_a": "$T_a$", "RH": "$RH$", "v_w": "$v_w$", "R": "$R$", "S, S_72": "$S$, $S_{72}$",
           "geometrija": r"geometrija ($\cos\theta_z$, $G_0$, $f_{sun}$, $\sin\omega$)"}
    names = sorted(imp, key=lambda k: imp[k]["mean"])
    fig, ax = plt.subplots(figsize=(5.8, 2.9))
    ax.barh([lab[n] for n in names], [imp[n]["mean"] for n in names], xerr=[imp[n]["std"] for n in names],
            color="#2b5c9e", ec="#333", lw=0.4, error_kw=dict(lw=0.7))
    ax.set_xlabel(r"porast RMSE nakon permutacije, $\Delta$RMSE (% $P_{nom}$)")
    save(fig, "slika_17_vaznost.png")


def fig_transfer(T):
    cities = list(prep.CITIES)
    lab = [prep.CITY_LABEL.get(c, c) for c in cities]
    q = T["q_Limburg"]
    avg = lambda d: np.mean(list(d.values()))
    fig, ax = plt.subplots(figsize=(6.3, 2.7))
    x = np.arange(len(cities)); w = 0.26
    ax.bar(x - w, [avg(T["locations"][c]["MLP"]["annual"]) for c in cities], w, color=COL["MLP"], ec="#333", lw=0.3, label="MLP")
    ax.bar(x, [avg(T["locations"][c]["LSTM"]["annual"]) for c in cities], w, color=COL["LSTM"], ec="#333", lw=0.3, label="LSTM")
    ax.bar(x + w, [T["locations"][c]["PVGIS_E_y"] * q for c in cities], w, color="#d99a0b", ec="#333", lw=0.3,
           label=f"PVGIS × $\\zeta$ ($\\zeta$ = {q:.3f})".replace(".", ","))
    ax.set_xticks(x); ax.set_xticklabels(lab, fontsize=8)
    ax.set_ylabel("kWh/kWp godišnje"); ax.set_ylim(800, 1450)
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    save(fig, "slika_18_bih_godisnje.png")

    fig, axs = plt.subplots(1, 2, figsize=(6.3, 2.5), sharey=True)
    for a, c, t in zip(axs, ["Travnik", "Mostar"], ["(a) Travnik", "(b) Mostar"]):
        L = T["locations"][c]
        a.plot(range(1, 13), [L["MLP"]["monthly"][str(m)] for m in range(1, 13)], "o-", ms=3, lw=1.1, color=COL["MLP"], label="MLP")
        a.plot(range(1, 13), [L["LSTM"]["monthly"][str(m)] for m in range(1, 13)], "s--", ms=2.5, lw=1, color=COL["LSTM"], label="LSTM")
        a.plot(range(1, 13), [v * q for v in L["PVGIS_E_m"]], "^:", ms=3, lw=1, color="#d99a0b", label=r"PVGIS × $\zeta$")
        a.set_xticks(range(1, 13)); a.set_xticklabels([m[0].upper() for m in MONTHS]); a.set_title(t)
    axs[0].set_ylabel("kWh/kWp po mjesecu"); axs[0].legend(frameon=False, fontsize=7.5)
    fig.tight_layout()
    save(fig, "slika_19_bih_mjesecno.png")


def main():
    df = prep.load_limburg()
    with open(RES / "metrics.json", encoding="utf-8") as f:
        log = json.load(f)
    with open(RES / "transfer.json", encoding="utf-8") as f:
        T = json.load(f)
    pred = pd.read_csv(RES / "pred_test.csv", parse_dates=["time"])
    fig_solar_geometry(); fig_pv_model(); fig_neuron(); fig_mlp(log["best"]["mlp"]["hidden"])
    fig_lstm_cell(); fig_pipeline(); fig_data(df); fig_results(log, pred); fig_transfer(T)
    print("slike:", sorted(p.name for p in FIG.glob("*.png")))


if __name__ == "__main__":
    main()
