#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Thermal waves analysis (Tasks 1.5 and 1.6), merged version.

For each drive period, this script:
  1. Fits every thermistor in every repeat to a sinusoid plus linear drift,
     and propagates the fit uncertainties to amplitude and phase.
  2. Computes gamma_i (amplitude ratio) and dphi_i (phase lag) relative to
     thermistor 0 for all thermistors, and combines the repeats (repeat-to-repeat
     scatter and individual fit uncertainties).
  3. Task 1.5a: D from the single pair of thermistors 0 and 3 (compare Task 1.4).
  4. Tasks 1.5b-d: fits gamma = exp(-x sqrt(w/2D)) and dphi = x sqrt(w/2D)
     against distance using all thermistors, giving D from each method.
  5. Task 1.6b: plots gamma and dphi against distance for every period, D versus
     omega for each method, and prints an overall weighted-mean D.

Add repeats or periods by editing DATASETS.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


DATASETS = {
    5: ["data/brass_T5_rep1.csv", "data/brass_T5_rep2.csv", "data/brass_T5_rep3.csv"],
    15: ["data/brass_T15_rep1.csv", "data/brass_T15_rep2.csv", "data/brass_T15_rep3.csv"],
    25: ["data/brass_T25_rep1.csv", "data/brass_T25_rep2.csv", "data/brass_T25_rep3.csv"],
    35: ["data/brass_T35_rep1.csv", "data/brass_T35_rep2.csv", "data/brass_T35_rep3.csv"],
}

THERMISTOR_0 = 0
THERMISTOR_3 = 3
DD_MM = 5.0  # thermistor spacing in mm, so D comes out in mm^2/s


# -----------------------------------------------------------------------------
# Data loading and sine fitting
# -----------------------------------------------------------------------------

def load_dataset(path):
    """Load a thermal-waves CSV exported by the laboratory software."""
    import pandas as pd

    data = pd.read_csv(path, header=3)
    timestamp = data.iloc[:, 0].to_numpy(dtype=float)
    output_voltage = data.iloc[:, 1].to_numpy(dtype=float)
    output_current = data.iloc[:, 2].to_numpy(dtype=float)
    temperatures = data.iloc[:, 3:].to_numpy(dtype=float)
    return timestamp, output_voltage, output_current, temperatures


def fit_sine_with_drift(t, temperature, period, return_fit=False):
    """Fit a cos(wt) + b sin(wt) + c + d t over a whole number of periods.

    Returns amplitude A = sqrt(a^2 + b^2) and phase = arctan2(b, a), so that the
    oscillation is A cos(wt - phase): a larger phase means a later peak, and the
    phase lag up the cylinder comes out positive (Task 1.3). The covariance
    matrix from curve_fit is propagated to the amplitude and phase. The linear
    term absorbs slow residual temperature drift.
    """
    t = np.asarray(t, dtype=float)
    temperature = np.asarray(temperature, dtype=float)
    t = t - t[0]

    n_periods = int(t[-1] // period)
    if n_periods < 1:
        raise ValueError(f"Dataset is shorter than one period ({period} s).")

    keep = t < n_periods * period
    t_fit = t[keep]
    y_fit = temperature[keep]
    omega = 2 * np.pi / period

    def model(t, a, b, c, d):
        return a * np.cos(omega * t) + b * np.sin(omega * t) + c + d * t

    popt, pcov = curve_fit(model, t_fit, y_fit)
    a, b, c, d = popt

    amplitude = np.hypot(a, b)
    phase = np.arctan2(b, a)

    # Propagate cov(a, b) to amplitude and phase.
    grad_amplitude = np.array([a / amplitude, b / amplitude, 0.0, 0.0])
    grad_phase = np.array([-b / amplitude**2, a / amplitude**2, 0.0, 0.0])
    amplitude_unc = np.sqrt(max(grad_amplitude @ pcov @ grad_amplitude, 0.0))
    phase_unc = np.sqrt(max(grad_phase @ pcov @ grad_phase, 0.0))

    if return_fit:
        fit = model(t_fit, *popt)
        return {"t": t_fit, "data": y_fit, "fit": fit, "residual": y_fit - fit}
    return amplitude, amplitude_unc, phase, phase_unc


# -----------------------------------------------------------------------------
# One repeat: gamma and phase lag for every thermistor
# -----------------------------------------------------------------------------

def analyse_repeat(path, period):
    t, _, _, temperatures = load_dataset(path)
    fits = np.array([fit_sine_with_drift(t, temperatures[:, i], period)
                     for i in range(temperatures.shape[1])])
    amp, amp_unc, phase, phase_unc = fits.T
    phase = np.unwrap(phase)  # lags past pi would otherwise wrap round

    gamma = amp / amp[THERMISTOR_0]
    gamma_unc = gamma * np.sqrt((amp_unc / amp) ** 2
                                + (amp_unc[THERMISTOR_0] / amp[THERMISTOR_0]) ** 2)
    phase_lag = phase - phase[THERMISTOR_0]
    phase_lag_unc = np.sqrt(phase_unc**2 + phase_unc[THERMISTOR_0] ** 2)
    # Thermistor 0 is the reference: gamma = 1 and lag = 0 exactly.
    gamma_unc[THERMISTOR_0] = 0.0
    phase_lag_unc[THERMISTOR_0] = 0.0

    return {"gamma": gamma, "gamma_unc": gamma_unc,
            "phase_lag": phase_lag, "phase_lag_unc": phase_lag_unc}


# -----------------------------------------------------------------------------
# Combining repeats
# -----------------------------------------------------------------------------

def mean_and_uncertainty(values, fit_uncertainties):
    """Mean over repeats, with uncertainty including repeat scatter and fit error.

    values and fit_uncertainties have shape (n_repeats, n_thermistors).
    The first term is the standard error of the repeats. The second is the
    propagated uncertainty of the mean from the individual fits. They are added
    in quadrature, which is conservative (the repeat scatter already contains
    some of the same noise).
    """
    values = np.asarray(values, dtype=float)
    fit_uncertainties = np.asarray(fit_uncertainties, dtype=float)
    n = len(values)
    mean = values.mean(axis=0)
    repeat_term = values.std(axis=0, ddof=1) / np.sqrt(n) if n > 1 else 0.0 * mean
    fit_term = np.sqrt(np.sum(fit_uncertainties**2, axis=0)) / n
    return mean, np.hypot(repeat_term, fit_term), repeat_term, fit_term


def combine_repeats(repeat_results):
    gamma, ug, ug_rep, ug_fit = mean_and_uncertainty(
        [r["gamma"] for r in repeat_results], [r["gamma_unc"] for r in repeat_results])
    lag, up, up_rep, up_fit = mean_and_uncertainty(
        [r["phase_lag"] for r in repeat_results], [r["phase_lag_unc"] for r in repeat_results])
    return {"gamma": gamma, "gamma_unc": ug,
            "gamma_repeat_unc": ug_rep, "gamma_fit_unc": ug_fit,
            "phase_lag": lag, "phase_lag_unc": up,
            "phase_repeat_unc": up_rep, "phase_fit_unc": up_fit}


# -----------------------------------------------------------------------------
# Diffusivity
# -----------------------------------------------------------------------------

def diffusivity_from_attenuation(gamma, gamma_unc, omega, x):
    """Task 1.5a: D from a single thermistor at distance x (mm)."""
    log_gamma = np.log(gamma)
    D = omega * x**2 / (2 * log_gamma**2)
    return D, abs(2 * D / (log_gamma * gamma)) * gamma_unc


def diffusivity_from_phase(phase_lag, phase_lag_unc, omega, x):
    """Task 1.5a: D from a single thermistor at distance x (mm)."""
    D = omega * x**2 / (2 * phase_lag**2)
    return D, abs(2 * D / phase_lag) * phase_lag_unc


def fit_D_vs_distance(x, gamma, gamma_unc, lag, lag_unc, omega):
    """Tasks 1.5b-d: fit the plane-slab model to all thermistors (skipping 0)."""
    def attenuation_model(x, D):
        return np.exp(-x * np.sqrt(omega / (2 * D)))

    def lag_model(x, D):
        return x * np.sqrt(omega / (2 * D))

    s = slice(1, None)  # thermistor 0 is the reference point
    (D_att,), c_att = curve_fit(attenuation_model, x[s], gamma[s], p0=[30],
                                sigma=gamma_unc[s], absolute_sigma=True)
    (D_ph,), c_ph = curve_fit(lag_model, x[s], lag[s], p0=[30],
                              sigma=lag_unc[s], absolute_sigma=True)
    return (D_att, np.sqrt(c_att[0, 0]), D_ph, np.sqrt(c_ph[0, 0]),
            attenuation_model, lag_model)


def weighted_mean(values, errors):
    w = 1 / np.asarray(errors) ** 2
    return np.sum(w * values) / np.sum(w), 1 / np.sqrt(np.sum(w))


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def plot_example_fit(period, path):
    """Raw data, fit and residuals for every thermistor in one file.
    Residuals should look like random noise around zero. A slow trend means the
    temperatures had not settled; a repeating wiggle means the wave is not a
    pure sine."""
    t, _, _, temperatures = load_dataset(path)
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(11.5, 7), sharex=True,
                                      gridspec_kw={"height_ratios": [3, 1]})
    for i in range(temperatures.shape[1]):
        r = fit_sine_with_drift(t, temperatures[:, i], period, return_fit=True)
        dots, = top.plot(r["t"], r["data"], ".", ms=2, alpha=0.4)
        top.plot(r["t"], r["fit"], "-", lw=1.2, color=dots.get_color(),
                 label=f"Thermistor {i}")
        bottom.plot(r["t"], r["residual"], lw=0.6, color=dots.get_color())
    top.set(ylabel="Temperature (°C)",
            title=f"{path}: data (dots) and sine + drift fit (lines), period = {period} s")
    top.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5))
    bottom.axhline(0, color="k", lw=0.6)
    bottom.set(xlabel="Time (s)", ylabel="Residual (°C)")
    fig.tight_layout()
    fig.savefig(f"example_fit_T{period}.png", dpi=180)


def plot_vs_distance(results):
    """Task 1.5b-d / 1.6b: gamma and phase lag against distance, all periods."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    for r in results:
        x, xx = r["x"], np.linspace(0, r["x"][-1], 100)
        line = ax1.errorbar(x, r["gamma"], r["gamma_unc"], fmt="o", capsize=2,
                            label=f"period = {r['period']} s")
        colour = line[0].get_color()
        ax1.plot(xx, r["attenuation_model"](xx, r["D_attenuation"]), color=colour)
        ax2.errorbar(x, r["phase_lag"], r["phase_lag_unc"], fmt="o", capsize=2, color=colour)
        ax2.plot(xx, r["lag_model"](xx, r["D_phase"]), color=colour)
    ax1.set(xlabel="Distance from thermistor 0 (mm)", ylabel=r"Amplitude ratio $\gamma$",
            title="Attenuation", yscale="log")
    ax2.set(xlabel="Distance from thermistor 0 (mm)", ylabel=r"Phase lag $\Delta\phi$ (rad)",
            title="Phase lag")
    ax1.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("gamma_and_phase_vs_distance.png", dpi=180)


def plot_diffusivity(results, method, filename):
    """D against omega with a weighted straight-line fit D = intercept + slope*omega.
    The plane-slab model predicts slope = 0. A clearly non-zero slope means the
    model is breaking down at some frequencies. Note the intercept is D
    extrapolated to omega = 0, where the model is least reliable, so use the
    weighted mean as the best estimate instead."""
    omega = np.array([r["omega"] for r in results])
    D = np.array([r[f"D_{method}"] for r in results])
    uD = np.array([r[f"uD_{method}"] for r in results])

    def line(w, intercept, slope):
        return intercept + slope * w

    popt, pcov = curve_fit(line, omega, D, sigma=uD, absolute_sigma=True,
                           p0=[np.mean(D), 0.0])
    (intercept, slope), (u_intercept, u_slope) = popt, np.sqrt(np.diag(pcov))

    xx = np.linspace(0.9 * omega.min(), 1.1 * omega.max(), 200)
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    ax.errorbar(omega, D, yerr=uD, fmt="o", capsize=4, label=f"{method} estimate")
    ax.plot(xx, line(xx, *popt), "--", label="weighted linear fit")
    ax.set(xlabel=r"Angular frequency $\omega$ (rad s$^{-1}$)",
           ylabel=r"Thermal diffusivity $D$ (mm$^2$ s$^{-1}$)",
           title=f"Diffusivity from {method} (all-thermistor fit)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(filename, dpi=180)

    print(f"  Linear fit: D = ({intercept:.4g} ± {u_intercept:.2g})"
          f" + ({slope:.4g} ± {u_slope:.2g}) omega")
    if len(results) == 2:
        print("  (only 2 periods: the line passes through both points and tests nothing)")
    print(f"  Slope is {abs(slope) / u_slope:.1f} sigma from zero")


# -----------------------------------------------------------------------------
# Main analysis
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    all_results = []

    for period in sorted(DATASETS):
        paths = DATASETS[period]
        plot_example_fit(period, paths[0])  # check the fit on the first repeat

        combined = combine_repeats([analyse_repeat(p, period) for p in paths])
        omega = 2 * np.pi / period
        x = np.arange(len(combined["gamma"])) * DD_MM

        # Task 1.5a: single pair, thermistors 0 and 3
        x3 = (THERMISTOR_3 - THERMISTOR_0) * DD_MM
        g3, ug3 = combined["gamma"][THERMISTOR_3], combined["gamma_unc"][THERMISTOR_3]
        p3, up3 = combined["phase_lag"][THERMISTOR_3], combined["phase_lag_unc"][THERMISTOR_3]
        D3_att, uD3_att = diffusivity_from_attenuation(g3, ug3, omega, x3)
        D3_ph, uD3_ph = diffusivity_from_phase(p3, up3, omega, x3)

        # Tasks 1.5b-d: all thermistors
        D_att, uD_att, D_ph, uD_ph, att_model, lag_model = fit_D_vs_distance(
            x, combined["gamma"], combined["gamma_unc"],
            combined["phase_lag"], combined["phase_lag_unc"], omega)

        all_results.append({"period": period, "omega": omega, "x": x,
                            "D_attenuation": D_att, "uD_attenuation": uD_att,
                            "D_phase": D_ph, "uD_phase": uD_ph,
                            "attenuation_model": att_model,
                            "lag_model": lag_model, **combined})

        print(f"\nPeriod {period} s, omega = {omega:.5g} rad/s, {len(paths)} repeat(s)")
        print(f"  Thermistors 0 and 3 only (Task 1.5a, compare with Task 1.4):")
        print(f"    gamma_3 = {g3:.4g} ± {ug3:.2g}   "
              f"(repeats ±{combined['gamma_repeat_unc'][THERMISTOR_3]:.2g}, "
              f"fit ±{combined['gamma_fit_unc'][THERMISTOR_3]:.2g})")
        print(f"    lag_3   = {p3:.4g} ± {up3:.2g} rad   "
              f"(repeats ±{combined['phase_repeat_unc'][THERMISTOR_3]:.2g}, "
              f"fit ±{combined['phase_fit_unc'][THERMISTOR_3]:.2g})")
        print(f"    D from attenuation = {D3_att:.4g} ± {uD3_att:.2g} mm^2/s")
        print(f"    D from phase lag   = {D3_ph:.4g} ± {uD3_ph:.2g} mm^2/s")
        print(f"  All thermistors, fit against distance (Tasks 1.5b-d):")
        print(f"    D from attenuation = {D_att:.4g} ± {uD_att:.2g} mm^2/s")
        print(f"    D from phase lag   = {D_ph:.4g} ± {uD_ph:.2g} mm^2/s")

    plot_vs_distance(all_results)

    D_a = np.array([r["D_attenuation"] for r in all_results])
    uD_a = np.array([r["uD_attenuation"] for r in all_results])
    D_p = np.array([r["D_phase"] for r in all_results])
    uD_p = np.array([r["uD_phase"] for r in all_results])
    Da_mean, uDa_mean = weighted_mean(D_a, uD_a)
    Dp_mean, uDp_mean = weighted_mean(D_p, uD_p)
    print("\nOverall D, weighted mean over all periods (check D vs omega plots first!)")
    print(f"  attenuation: D = {Da_mean:.4g} ± {uDa_mean:.2g} mm^2/s")
    print(f"  phase lag:   D = {Dp_mean:.4g} ± {uDp_mean:.2g} mm^2/s")

    if len(all_results) >= 2:
        print("\nAttenuation method")
        plot_diffusivity(all_results, "attenuation", "D_vs_omega_attenuation.png")
        print("\nPhase-lag method")
        plot_diffusivity(all_results, "phase", "D_vs_omega_phase.png")
    else:
        print("\nAt least two periods are needed for a straight-line fit of D vs omega.")

    plt.show()