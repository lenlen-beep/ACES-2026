import pandas as pd

# Import Meteostat library and dependencies
from datetime import datetime
import meteostat as ms
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from statsmodels.tsa.ar_model import AutoReg


def load_temperature_data(
    lat,
    lon,
    year,
    reload_data=False,
    cache_dir="Testprojects/weather_cache"
):

    # --------------------------------------------------
    # Cache-Ordner
    # --------------------------------------------------

    cache_path = Path(cache_dir)
    cache_path.mkdir(exist_ok=True)

    # --------------------------------------------------
    # Dateiname
    # --------------------------------------------------

    filename = f"weather_{lat:.3f}_{lon:.3f}_{year}.csv"

    filepath = cache_path / filename

    # --------------------------------------------------
    # Cache prüfen
    # --------------------------------------------------

    if filepath.exists() and not reload_data:

        print("Lade Wetterdaten aus Cache ...")

        df = pd.read_csv(
            filepath,
            index_col=0,
            parse_dates=True
        )

    else:

        print("Lade Wetterdaten von Meteostat ...")

        # --------------------------------------------------
        # Nächste Station finden
        # --------------------------------------------------

        nearby = ms.stations.nearby(ms.Point(lat, lon), limit=1)

        station_id = nearby.index[0]

        print(f"\nVerwendete Station: {station_id}")
        
        # --------------------------------------------------
        # Zeitraum
        # --------------------------------------------------

        start = datetime(year, 1, 1)
        end = datetime(year, 12, 31, 23, 59)

        # --------------------------------------------------
        # Wetterdaten laden
        # --------------------------------------------------

        ts = ms.hourly(
            station_id,
            start,
            end
        )

        df = ts.fetch()

        # --------------------------------------------------
        # Lokal speichern
        # --------------------------------------------------

        df.to_csv(filepath)

        print(f"\nGespeichert unter:\n{filepath}")

    # --------------------------------------------------
    # Temperatur extrahieren
    # --------------------------------------------------

    temperature = df["temp"]

    return temperature, df


# ======================================================
# Jahre 2014–2024 laden und cachen
# ======================================================

LAT       = 54.78
LON       = 9.43
CACHE_DIR = "src/Testprojects/stochastic_test_scripts/weather_cache_10y"

temp_series = {}

for year in range(2014, 2025):
    temp, _ = load_temperature_data(
        lat=LAT,
        lon=LON,
        year=year,
        reload_data=False,
        cache_dir=CACHE_DIR
    )
    temp_series[year] = temp
    print(f"  {year}: {len(temp.dropna())} Stunden geladen")

print(f"\nFertig. {len(temp_series)} Jahre in Cache: {CACHE_DIR}")


# ======================================================
# FFT – Temperaturanalyse 2014–2024
# ======================================================

# Alle Jahre zu einer langen Zeitreihe zusammenführen
# Auf sauberen stündlichen Index interpolieren → keine Lücken → saubere FFT
T_raw   = pd.concat(temp_series.values())
T_raw   = T_raw[~T_raw.index.duplicated()]          # doppelte Zeitstempel (DST) entfernen

idx_voll = pd.date_range(start='2014-01-01', end='2024-12-31 23:00', freq='1h')
T_interp = T_raw.reindex(idx_voll).interpolate(method='time')

n_luecken = idx_voll.difference(T_raw.index).size
print(f"  Lücken gefüllt: {n_luecken} Stunden interpoliert")

T_gesamt = T_interp.values
n        = len(T_gesamt)
print(f"\nGesamtzeitreihe: {n} Stunden ({n/8760:.1f} Jahre)")

# --------------------------------------------------
# Linearer Klimatrend (Rolling-Jahresmittel → Linearfit)
# --------------------------------------------------

def trend_func(x, a, b):
    return a * x + b

T_s      = pd.Series(T_gesamt)
T_roll_Y = T_s.rolling(window=8760, center=True).mean()
valid    = T_roll_Y.dropna()
x_valid  = valid.index.to_numpy().astype(float)

popt_t, _ = curve_fit(trend_func, x_valid, valid.values)
a_t, b_t  = popt_t

x_all    = np.arange(n, dtype=float)
T_trend  = trend_func(x_all, a_t, b_t)
T_dtrend = T_gesamt - T_trend

print(f"\nKlimatrend: {a_t * 8760:.4f} °C/Jahr")

# --------------------------------------------------
# FFT-Amplitudenspektrum (nur für Visualisierung)
# --------------------------------------------------

fft_vals = np.fft.rfft(T_dtrend)
fft_freq = np.fft.rfftfreq(n, d=1)
fft_amp  = np.abs(fft_vals)
perioden = 1 / fft_freq[1:]

# --------------------------------------------------
# Parametrische Zerlegung: Jahres- und Tagesgang
#
# Jahresgang: α_y · sin(ω_y · t + θ_y)  mit ω_y = 2π / (365.25 · 24)
# Tagesgang:  α_d · sin(ω_d · t + θ_d)  mit ω_d = 2π / 24
# --------------------------------------------------

x_arr   = np.arange(n, dtype=float)
omega_y = 2 * np.pi / (365.25 * 24)
omega_d = 2 * np.pi / 24

def seasonality_y(x, alpha, theta):
    return alpha * np.sin(omega_y * x + theta)

def seasonality_d(x, alpha, theta):
    return alpha * np.sin(omega_d * x + theta)

popt_y, _ = curve_fit(seasonality_y, x_arr, T_dtrend - T_dtrend.mean(),
                       p0=[7.0, -1.0], maxfev=10000)
alpha_y, theta_y = popt_y
T_jahresgang = seasonality_y(x_arr, alpha_y, theta_y) + T_dtrend.mean()

print(f"\nJahresgang-Fit:  α={alpha_y:.3f}°C,  θ={theta_y:.3f} rad")

popt_d, _ = curve_fit(seasonality_d, x_arr, T_dtrend - T_jahresgang,
                       p0=[2.0, 0.0], maxfev=10000)
alpha_d, theta_d = popt_d
T_tagesgang = seasonality_d(x_arr, alpha_d, theta_d)

print(f"Tagesgang-Fit:   α={alpha_d:.3f}°C,  θ={theta_d:.3f} rad")

T_residuum = T_dtrend - T_jahresgang - T_tagesgang

varianz = T_gesamt.var()
print("\nVarianzzerlegung Temperatur (2014–2024):")
for name, komp in [('Trend', T_trend), ('Jahresgang', T_jahresgang),
                   ('Tagesgang', T_tagesgang), ('Residuum', T_residuum)]:
    print(f"  {name:<12}: {komp.var()/varianz*100:.1f}%  (Std={komp.std():.2f}°C)")

# --------------------------------------------------
# AR(1) / OU-Prozess auf Residuen
#
# AR(1):  X_t = γ·X_{t-1} + ε_t
# OU:     dX = -κ·X dt + σ dW
# Zusammenhang (dt=1h): γ = 1−κ,  σ = std(ε)
# --------------------------------------------------

T_residuum_s = pd.Series(T_residuum)

ar_model = AutoReg(T_residuum_s, lags=1, old_names=False, trend='n')
ar_fit   = ar_model.fit()
gamma    = ar_fit.params.iloc[0]
kappa    = 1 - gamma                   # mean-reversion speed [1/h]
sigma_ou = ar_fit.resid.std()          # Volatilität [°C/√h]
std_kappa = ar_fit.bse.iloc[0]

korrelationszeit = 1 / kappa

print(ar_fit.summary())
print(f"\nOU-Parameter aus AR(1)-Fit:")
print(f"  γ (AR-Koeffizient)  = {gamma:.6f}")
print(f"  κ = 1−γ             = {kappa:.6f} ± {std_kappa:.6f}  [1/h]")
print(f"  σ (Residuen-Std)    = {sigma_ou:.4f}  [°C/√h]")
print(f"  Korrelationszeit    = 1/κ = {korrelationszeit:.1f} h  ({korrelationszeit/24:.1f} Tage)")

# Autokorrelation empirisch + exp(-β·lag)-Fit zur Kreuzvalidierung
lags_h   = np.arange(0, 201)
autocorr = [T_residuum_s.corr(T_residuum_s.shift(int(l))) for l in lags_h]

def model_acf(x, beta):
    return np.exp(-beta * x)

popt_acf, _ = curve_fit(model_acf, lags_h, autocorr)
beta_acf    = popt_acf[0]

# sigma aus verschobenen Residuen (Kreuzvalidierung)
gamma_acf     = 1.0 - beta_acf
resid_shift   = T_residuum_s.shift(1).iloc[1:]
resid_manual  = T_residuum_s.iloc[1:] - gamma_acf * resid_shift
sigma_manual  = resid_manual.std()

print(f"\nKreuzvalidierung über ACF-Fit:")
print(f"  β (exp-Fit)          = {beta_acf:.6f}  [1/h]  (≈ κ)")
print(f"  σ (manuelle Methode) = {sigma_manual:.4f}  [°C/√h]")

# --------------------------------------------------
# Saisonale Volatilität σ(t)
# Fit auf AR(1)-Innovationen: ε_t = X_t − γ·X_{t−1}
# σ(t) = σ_mean + σ_amp · sin(ω_y · t + θ_σ)
# Erwartung: σ kleiner im Winter, größer im Sommer (maritime Dämpfung)
# --------------------------------------------------

ar_innov   = pd.Series(T_residuum[1:] - gamma * T_residuum[:-1])
sigma_roll = ar_innov.rolling(window=720, center=True).std()
valid_s    = sigma_roll.dropna()
x_sigma    = (valid_s.index + 1).to_numpy().astype(float)

def sigma_seasonal(x, sigma_mean, sigma_amp, theta_s):
    return sigma_mean + sigma_amp * np.sin(omega_y * x + theta_s)

popt_s, _ = curve_fit(sigma_seasonal, x_sigma, valid_s.values,
                       p0=[sigma_ou, sigma_ou * 0.2, 0.0], maxfev=10000)
sigma_mean_s, sigma_amp_s, theta_s_fit = popt_s

sigma_t = sigma_seasonal(x_all, sigma_mean_s, sigma_amp_s, theta_s_fit)

print(f"\nSaisonale Volatilität σ(t):")
print(f"  σ_mean = {sigma_mean_s:.4f} °C/√h")
print(f"  σ_amp  = {sigma_amp_s:.4f} °C/√h  ({abs(sigma_amp_s)/sigma_mean_s*100:.1f}% Modulation)")
print(f"  σ_min  = {sigma_t.min():.4f} °C/√h,  σ_max = {sigma_t.max():.4f} °C/√h")


# --------------------------------------------------
# Plots
# --------------------------------------------------

fig, axes = plt.subplots(2, 3, figsize=(20, 10))

# Amplitudenspektrum
axes[0, 0].plot(perioden, fft_amp[1:], linewidth=0.6, color='steelblue')
axes[0, 0].set_xscale('log')
#axes[0, 0].axvline(24,   color='red',    linestyle='--', label='24h  Tagesgang')
#axes[0, 0].axvline(168,  color='orange', linestyle='--', label='168h Wochengang')
#axes[0, 0].axvline(8760, color='green',  linestyle='--', label='8760h Jahresgang')
axes[0, 0].set_xlabel("Periode [h] (log)")
axes[0, 0].set_ylabel("Amplitude")
axes[0, 0].set_title("FFT Amplitudenspektrum\nTemperatur 2014–2024")
axes[0, 0].legend(fontsize=8)
axes[0, 0].grid(True, alpha=0.3)

# Zeitreihe + Jahresgang + Trend
t = np.arange(n)
axes[0, 1].plot(t, T_gesamt,              color='steelblue', linewidth=0.4, alpha=0.5, label='Original')
axes[0, 1].plot(t, T_jahresgang + T_trend, color='green',     linewidth=1.5,            label='Jahresgang + Trend')
axes[0, 1].plot(t, T_trend,               color='orange',    linewidth=1.5, linestyle='--', label='Trend')
axes[0, 1].set_xlabel("Stunde")
axes[0, 1].set_ylabel("Temperatur [°C]")
axes[0, 1].set_title("Original + Jahresgang + Trend (2014–2024)")
axes[0, 1].legend(fontsize=8)
axes[0, 1].grid(True, alpha=0.3)

# Varianzzerlegung
namen_v  = ['Trend', 'Jahresgang', 'Tagesgang', 'Residuum']
vars_v   = [T_trend.var(), T_jahresgang.var(), T_tagesgang.var(), T_residuum.var()]
farben_v = ['orange', 'green', 'red', 'gray']
bars = axes[0, 2].bar(namen_v, [v/varianz*100 for v in vars_v], color=farben_v, alpha=0.8)
for bar, pct in zip(bars, [v/varianz*100 for v in vars_v]):
    axes[0, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f'{pct:.1f}%', ha='center', fontsize=9)
axes[0, 2].set_ylabel("Anteil Gesamtvarianz [%]")
axes[0, 2].set_title("Varianzzerlegung Temperatur")
axes[0, 2].grid(True, axis='y', alpha=0.3)

# Residuum Zeitreihe + saisonales σ(t)-Band
sigma_t_proc = sigma_t / np.sqrt(1 - gamma**2)  # Prozess-Std für Visualisierung
axes[1, 0].plot(t, T_residuum, color='gray', linewidth=0.4, alpha=0.8, label='Residuum')
axes[1, 0].fill_between(t, -sigma_t_proc, sigma_t_proc, alpha=0.25, color='steelblue',
                         label='±σ(t) saisonal')
axes[1, 0].axhline(0, color='black', linewidth=0.8)
axes[1, 0].set_xlabel("Stunde")
axes[1, 0].set_ylabel("Residuum [°C]")
axes[1, 0].set_title(f"Residuum + saisonale Volatilität\n"
                     f"σ_mean={sigma_mean_s:.3f}, σ_amp={sigma_amp_s:.3f} °C/√h")
axes[1, 0].legend(fontsize=8)
axes[1, 0].grid(True, alpha=0.3)

# Autokorrelation empirisch + exp-Fit
acf_fit = model_acf(lags_h, beta_acf)
axes[1, 1].plot(lags_h, autocorr, color='gray', linewidth=0.8, label='Empirische ACF')
axes[1, 1].plot(lags_h, acf_fit,  color='red',  linewidth=2.0,
                label=f'exp(−β·h),  β={beta_acf:.4f}/h')
axes[1, 1].axhline(0, color='black', linewidth=0.6)
axes[1, 1].axvline(1/beta_acf, color='orange', linestyle='--',
                   label=f'1/β = {1/beta_acf:.1f}h = {1/beta_acf/24:.1f}d')
axes[1, 1].set_xlabel("Lag [h]")
axes[1, 1].set_ylabel("Autokorrelation")
axes[1, 1].set_title(f"Autokorrelation T_residuum + exp-Fit\n"
                     f"Korrelationszeit = {korrelationszeit:.1f}h = {korrelationszeit/24:.1f} Tage")
axes[1, 1].legend(fontsize=8)
axes[1, 1].grid(True, alpha=0.3)

# AR(1)-Simulation mit saisonalem σ(t): X_t = γ·X_{t-1} + σ(t)·ε_t
sigma_stationaer = sigma_mean_s / np.sqrt(1 - gamma**2)
T_ou_sim = np.zeros(n)
T_ou_sim[0] = np.random.normal(0, sigma_stationaer)
for i in range(1, n):
    T_ou_sim[i] = gamma * T_ou_sim[i-1] + sigma_t[i] * np.random.normal()

t_kurz = np.arange(720)
axes[1, 2].plot(t_kurz, T_residuum[:720], color='gray', linewidth=0.8,
                label=f'T_residuum (Std={T_residuum.std():.2f}°C)')
axes[1, 2].plot(t_kurz, T_ou_sim[:720],   color='red',  linewidth=0.8,
                alpha=0.8, label=f'OU-Simulation (Std={sigma_stationaer:.2f}°C)')
axes[1, 2].axhline(0, color='black', linewidth=0.6)
axes[1, 2].set_xlabel("Stunde")
axes[1, 2].set_ylabel("Temperatur [°C]")
axes[1, 2].set_title("Residuum vs. OU-Simulation (erste 30 Tage)\n"
                     "Gleiche Skalierung = Modell passt")
axes[1, 2].legend(fontsize=8)
axes[1, 2].grid(True, alpha=0.3)

plt.suptitle("FFT + OU-Prozess – Temperaturanalyse Flensburg 2014–2024",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()


# ======================================================
# Monte-Carlo-Simulation
# T_sim(t) = T_jahresgang(t) + T_tagesgang(t) + OU(t)
# ======================================================

def simulate_temperature_mc(T_jahresgang_year, T_tagesgang_year,
                             gamma, sigma_t_year, n_sim=100, seed=None,
                             T_trend_year=None):
    """
    Simuliert n_sim Realisierungen eines Temperaturjahres (Stundenwerte).
    T_sim = T_trend + T_jahresgang + T_tagesgang + AR(1) mit saisonalem σ(t)
    Rückgabe: Array (n_sim, n_stunden).
    """
    n_h = len(T_jahresgang_year)
    rng = np.random.default_rng(seed)
    sigma_stat = sigma_t_year.mean() / np.sqrt(1 - gamma**2)
    if T_trend_year is None:
        T_trend_year = np.zeros(n_h)

    T_mc = np.zeros((n_sim, n_h))
    for s in range(n_sim):
        ou = np.zeros(n_h)
        ou[0] = rng.normal(0, sigma_stat)
        noise = rng.normal(0, 1, n_h)
        for i in range(1, n_h):
            ou[i] = gamma * ou[i-1] + sigma_t_year[i] * noise[i]
        T_mc[s] = T_trend_year + T_jahresgang_year + T_tagesgang_year + ou

    return T_mc


# Parametrisches Referenzjahr für MC (8760 Stunden ab Jahresbeginn)
n_h_jahr     = 8760
x_jahr       = np.arange(n_h_jahr, dtype=float)
T_jg_param   = seasonality_y(x_jahr, alpha_y, theta_y) + T_dtrend.mean()
T_tg_param   = seasonality_d(x_jahr, alpha_d, theta_d)
T_trend_rep  = np.full(n_h_jahr, T_trend.mean())
sigma_t_year = sigma_seasonal(x_jahr, sigma_mean_s, sigma_amp_s, theta_s_fit)

T_mc = simulate_temperature_mc(T_jg_param, T_tg_param, gamma, sigma_t_year,
                                n_sim=200, seed=42, T_trend_year=T_trend_rep)

print(f"\nMonte-Carlo: {T_mc.shape[0]} Simulationen × {T_mc.shape[1]} Stunden")
print(f"  Mittlere Std der Simulationen: {T_mc.std(axis=1).mean():.2f}°C")
print(f"  Std Referenzjahr (2014):       {T_gesamt[:n_h_jahr].std():.2f}°C")

# --------------------------------------------------
# MC-Plot: Perzentilband + Extremwerte + Referenzjahr
# --------------------------------------------------

t_h = np.arange(n_h_jahr)
p1, p5, p25, p50, p75, p95, p99 = np.percentile(T_mc, [1, 5, 25, 50, 75, 95, 99], axis=0)
T_mc_min = T_mc.min(axis=0)
T_mc_max = T_mc.max(axis=0)

n_complete  = (n // n_h_jahr) * n_h_jahr
T_jahre     = T_gesamt[:n_complete].reshape(-1, n_h_jahr)  # (n_jahre, 8760)
n_jahre     = T_jahre.shape[0]
jahre       = list(range(2014, 2014 + n_jahre))

fig2, ax = plt.subplots(figsize=(16, 5))
ax.fill_between(t_h, T_mc_min, T_mc_max, alpha=0.08, color='steelblue', label='Min–Max')
ax.fill_between(t_h, p1,  p99, alpha=0.12, color='steelblue', label='1–99 Perzentil')
ax.fill_between(t_h, p5,  p95, alpha=0.20, color='steelblue', label='5–95 Perzentil')
ax.fill_between(t_h, p25, p75, alpha=0.35, color='steelblue', label='25–75 Perzentil')
ax.plot(t_h, p50, color='steelblue', linewidth=1.2, label='Median')

colors_hist = plt.cm.tab10(np.linspace(0, 1, n_jahre))
for i, (jahr, farbe) in enumerate(zip(jahre, colors_hist)):
    ax.plot(t_h, T_jahre[i], color=farbe, linewidth=0.5, alpha=0.7, label=str(jahr))

ax.set_xlabel("Stunde")
ax.set_ylabel("Temperatur [°C]")
ax.set_title(f"MC-Simulation Temperatur (n={T_mc.shape[0]}) + historische Jahre  [saisonales σ(t)]\n"
             f"γ={gamma:.6f}, σ_mean={sigma_mean_s:.4f}, σ_amp={sigma_amp_s:.4f}°C/√h, "
             f"Korrelationszeit={korrelationszeit:.1f}h")
ax.legend(fontsize=7, loc='upper right', ncol=2)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

