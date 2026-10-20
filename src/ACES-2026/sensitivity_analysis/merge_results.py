"""Korrigierte Einzellaeufe in die Gesamtdatei zurueckschreiben.

Hintergrund: sensitivity.py ueberschreibt sensitivity_results.csv bei jedem
Aufruf. Laeufe, die wegen eines Wechsels der Netzentgeltstufe einzeln
wiederholt wurden, muessen daher nachtraeglich in die Gesamttabelle
uebernommen werden.

Aufruf aus dem Repo-Wurzelverzeichnis:

    python src/ACES-2026/merge_results.py sensitivity_results_lower.csv korrektur1.csv korrektur2.csv

Das erste Argument ist die vollstaendige Tabelle, alle weiteren enthalten die
korrigierten Zeilen. Ergebnis: Data/sensitivity_results.csv
"""

import sys
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parent / "Data"


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)

    paths = [DATA / a if not Path(a).is_absolute() else Path(a) for a in sys.argv[1:]]
    for p in paths:
        if not p.exists():
            sys.exit(f"FEHLER: {p} nicht gefunden.")

    base = pd.read_csv(paths[0]).set_index("scenario")
    for p in paths[1:]:
        patch = pd.read_csv(p).set_index("scenario")
        for name, row in patch.iterrows():
            if name in base.index:
                print(f"ersetzt : {name}  "
                      f"{base.loc[name, 'lcoh_eur_per_mwh']:.2f} -> "
                      f"{row['lcoh_eur_per_mwh']:.2f} EUR/MWh")
            else:
                print(f"ergaenzt: {name}  {row['lcoh_eur_per_mwh']:.2f} EUR/MWh")
            base.loc[name] = row

    out = DATA / "sensitivity_results.csv"
    base.to_csv(out)
    print(f"\nGeschrieben: {out}  ({len(base)} Zeilen)")

    if "vbh_consistent" in base.columns:
        bad = base[~base["vbh_consistent"].astype(bool)]
        if len(bad):
            print("\nWARNUNG: weiterhin inkonsistente Netzentgeltstufe in:")
            for name in bad.index:
                print(f"  {name}")
        else:
            print("Alle Laeufe konsistent.")


if __name__ == "__main__":
    main()
