import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------
# Daten laden (ein Jahr als Beispiel)
# --------------------------------------------------

load_file = r"src/Testprojects/Data/2017-2019 Stadtwerke Flensburg Heat Network Data Hourly.xlsx"
df = pd.read_excel(load_file, skiprows=1, header=0, usecols=[0, 1])
df.columns = ['Datum', 'Wärmeleistung in MW']
df['Datum'] = pd.to_datetime(df['Datum'])
df['Wärmeleistung in MW'] = pd.to_numeric(df['Wärmeleistung in MW'], errors='coerce')

# Nur 2017
df = df[df['Datum'].dt.year == 2017].dropna()
Q = df['Wärmeleistung in MW'].values
n = len(Q)

# --------------------------------------------------
# FFT
# --------------------------------------------------

# Fourier-Transformation
fft_vals = np.fft.rfft(Q)           # komplexe Koeffizienten
fft_freq = np.fft.rfftfreq(n, d=1)  # Frequenz in 1/Stunden
fft_amp  = np.abs(fft_vals)         # Amplitude = "Stärke" jeder Frequenz

# Perioden in Stunden (anschaulicher als Frequenz)
perioden = 1 / fft_freq[1:]         # erstes Element (DC) überspringen
amplituden = fft_amp[1:]

# --------------------------------------------------
# Plot 1: Amplitudenspektrum
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(perioden, amplituden, linewidth=0.8)
ax.set_xscale('log')
ax.set_xlim([1, n])
ax.axvline(24,   color='red',    linestyle='--', label='24h  (Tagesgang)')
ax.axvline(168,  color='orange', linestyle='--', label='168h (Wochengang)')
ax.axvline(8760, color='green',  linestyle='--', label='8760h (Jahresgang)')
ax.set_xlabel("Periode [Stunden] (log)")
ax.set_ylabel("Amplitude")
ax.set_title("Fourier-Amplitudenspektrum – Wärmeleistung 2017")
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.show()

# --------------------------------------------------
# Rekonstruktion: nur bestimmte Frequenzen behalten
# --------------------------------------------------

def rekonstruiere(fft_vals, n, perioden_min, perioden_max):
    """Behält nur Frequenzanteile in [perioden_min, perioden_max] Stunden."""
    fft_gefiltert = fft_vals.copy()
    freqs = np.fft.rfftfreq(n, d=1)
    for i, f in enumerate(freqs):
        if f == 0:
            continue
        p = 1 / f
        if not (perioden_min <= p <= perioden_max):
            fft_gefiltert[i] = 0
    return np.fft.irfft(fft_gefiltert, n=n)

# Jahresgang: Perioden > 500h
Q_jahresgang = rekonstruiere(fft_vals, n, 500, n)

# Tagesgang: Perioden 20–30h
Q_tagesgang = rekonstruiere(fft_vals, n, 20, 30)

# Wochengang: Perioden 150–200h
Q_wochengang = rekonstruiere(fft_vals, n, 150, 200)

# Residuum: alles was übrig bleibt
Q_residuum = Q - Q_jahresgang - Q_tagesgang - Q_wochengang

# --------------------------------------------------
# Plot 2: Zerlegung
# --------------------------------------------------

t = np.arange(n)
fig, axes = plt.subplots(5, 1, figsize=(14, 16), sharex=True)

axes[0].plot(t, Q,             color='steelblue', linewidth=0.6)
axes[0].set_ylabel("MW")
axes[0].set_title("Original")
axes[0].grid(True)

axes[1].plot(t, Q_jahresgang,  color='green',  linewidth=1.2)
axes[1].set_ylabel("MW")
axes[1].set_title("Jahresgang (deterministisch, >500h)")
axes[1].grid(True)

axes[2].plot(t, Q_tagesgang,   color='orange', linewidth=0.8)
axes[2].set_ylabel("MW")
axes[2].set_title("Tagesgang (20–30h)")
axes[2].grid(True)

axes[3].plot(t, Q_wochengang,  color='purple', linewidth=0.8)
axes[3].set_ylabel("MW")
axes[3].set_title("Wochengang (150–200h)")
axes[3].grid(True)

axes[4].plot(t, Q_residuum,    color='red',    linewidth=0.5, alpha=0.7)
axes[4].set_ylabel("MW")
axes[4].set_title(f"Residuum (stochastisch) – Std={Q_residuum.std():.1f} MW")
axes[4].set_xlabel("Stunde im Jahr")
axes[4].grid(True)

plt.tight_layout()
plt.show()

# --------------------------------------------------
# Zusammenfassung: Anteil jeder Komponente
# --------------------------------------------------

print(f"Varianz Original:    {Q.var():.1f}")
print(f"Varianz Jahresgang:  {Q_jahresgang.var():.1f}  ({Q_jahresgang.var()/Q.var()*100:.1f}%)")
print(f"Varianz Tagesgang:   {Q_tagesgang.var():.1f}   ({Q_tagesgang.var()/Q.var()*100:.1f}%)")
print(f"Varianz Wochengang:  {Q_wochengang.var():.1f}   ({Q_wochengang.var()/Q.var()*100:.1f}%)")
print(f"Varianz Residuum:    {Q_residuum.var():.1f}   ({Q_residuum.var()/Q.var()*100:.1f}%)")
