import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

DATA_PATH = "src/ACES-2026/Data/selected_267_profiles_2019_wide.csv"
OUT_PATH  = "src/ACES-2026/Data/profile_analyse.png"


def load(path=DATA_PATH):
    df = pd.read_csv(path)
    df["Datum"] = pd.to_datetime(df["Datum"])
    return df


def plot_profile_analysis(df, out=OUT_PATH):
    load_cols = [c for c in df.columns if c != "Datum"]
    vals = df[load_cols].values           # (8760, 267)
    total = vals.sum(axis=1)              # Gesamtlast je Stunde [kW]
    datum = df["Datum"]

    plt.rcParams.update({
        "font.family":       "sans-serif",
        "font.size":         12,
        "axes.titlesize":    13,
        "axes.labelsize":    12,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "grid.color":        "#DDDDDD",
        "grid.linewidth":    0.7,
        "legend.frameon":    False,
    })

    fig = plt.figure(figsize=(18, 14))
    gs  = fig.add_gridspec(3, 2, hspace=0.42, wspace=0.32)

    # ── 1. Summendauerlinie ────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    sorted_total = np.sort(total)[::-1]
    ax1.fill_between(range(len(sorted_total)), sorted_total / 1000,
                     alpha=0.55, color="#2E86AB")
    ax1.plot(sorted_total / 1000, color="#2E86AB", linewidth=1.2)
    ax1.axhline(sorted_total.mean() / 1000, color="gray", linestyle="--",
                linewidth=1, label=f"Ø {sorted_total.mean()/1000:.1f} MW")
    ax1.set_xlabel("Stunden (sortiert)")
    ax1.set_ylabel("Gesamtlast [MW]")
    ax1.set_title(f"Summendauerlinie – alle {len(load_cols)} Gebäude")
    ax1.legend()
    ax1.grid(True)

    # ── 2. Einzelne Dauerkurven (alle 267 als dünne Linien) ───────────────
    ax2 = fig.add_subplot(gs[1, 0])
    for col in load_cols:
        sorted_ind = np.sort(vals[:, load_cols.index(col)])[::-1]
        ax2.plot(sorted_ind, color="#2E86AB", alpha=0.08, linewidth=0.6)
    # Median und 90. Perzentil hervorheben
    p50 = np.sort(np.median(vals, axis=1))[::-1]
    p90 = np.sort(np.percentile(vals, 90, axis=1))[::-1]
    ax2.plot(p50, color="black",   linewidth=1.4, label="Median")
    ax2.plot(p90, color="#C73E1D", linewidth=1.4, label="90. Pz.")
    ax2.set_xlabel("Stunden (sortiert)")
    ax2.set_ylabel("Last [kW]")
    ax2.set_title("Einzelne Dauerkurven")
    ax2.legend()
    ax2.grid(True)

    # ── 3. Gesamtlastzeitreihe ────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(datum, total / 1000, color="#2E86AB", linewidth=0.5, alpha=0.8)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax3.xaxis.set_major_locator(mdates.MonthLocator())
    ax3.set_ylabel("Gesamtlast [MW]")
    ax3.set_title("Gesamtlast Zeitreihe 2019")
    ax3.grid(True)

    # ── 4. Nullwerte je Zeitstempel ───────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 0])
    zeros_per_ts = (vals == 0).sum(axis=1)
    ax4.plot(datum, zeros_per_ts, color="#A23B72", linewidth=0.5, alpha=0.7)
    ax4.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax4.xaxis.set_major_locator(mdates.MonthLocator())
    ax4.set_ylabel("Anzahl Gebäude mit Last = 0")
    ax4.set_title("Nullwerte je Zeitstempel")
    ax4.grid(True)

    # ── 5. Nullwerte je Gebäude (Histogramm) ─────────────────────────────
    ax5 = fig.add_subplot(gs[2, 1])
    zeros_per_bld = (vals == 0).sum(axis=0)
    ax5.hist(zeros_per_bld, bins=40, color="#F18F01", edgecolor="white")
    ax5.axvline(zeros_per_bld.mean(), color="black", linestyle="--",
                linewidth=1.2, label=f"Ø {zeros_per_bld.mean():.0f} h/a")
    ax5.set_xlabel("Stunden mit Last = 0 [h/a]")
    ax5.set_ylabel("Anzahl Gebäude")
    ax5.set_title("Nullstunden je Gebäude")
    ax5.legend()
    ax5.grid(True)

    fig.suptitle("Lastprofilanalyse – 267 Gebäude, Jerrishoe 2019",
                 fontsize=15, fontweight="bold", y=1.01)

    plt.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"Plot gespeichert: {out}")

    # ── Konsolenauswertung ────────────────────────────────────────────────
    print(f"\n--- Nullwert-Auswertung ---")
    print(f"Zeitstempel mit mind. 1 Null:  {(zeros_per_ts > 0).sum()} / 8760 h")
    print(f"Max. Nullgebäude gleichzeitig: {zeros_per_ts.max()} / {len(load_cols)}")
    print(f"Gebäude mit   0 Nullstunden:   {(zeros_per_bld == 0).sum()}")
    print(f"Gebäude mit >100 Nullstunden:  {(zeros_per_bld > 100).sum()}")
    print(f"Gebäude mit >500 Nullstunden:  {(zeros_per_bld > 500).sum()}")
    print(f"Gebäude mit >1000 Nullstunden: {(zeros_per_bld > 1000).sum()}")

    plt.show()


if __name__ == "__main__":
    df = load()
    plot_profile_analysis(df)
