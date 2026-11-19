"""
Task 2.4: characteristic impedance Z0 from pulse reflections.

Probe 1 (CH1) at node 0, probe 2 (CH2) at node 20, 50 us pulses, end resistor R.
At node 20 the incident pulse and its echo from the end are separate in time
(the echo returns 2 x 20 sections x 2.2 us = 88 us later), so each can be
measured on its own.

For each file:  Gamma_measured = (echo height) / (incident height), both measured
as the average voltage over the flat middle of each pulse.

Theory (Eq. 2.7):  Gamma = (R - Z0) / (R + Z0).
The echo also loses a little height travelling the extra 40 sections to the end
and back, so we fit  Gamma_measured = A (R - Z0) / (R + Z0),  with A (loss factor)
and Z0 free. The fitted Z0 is the resistance at which the echo would vanish.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

FILES = {           # end resistance (ohm, from multimeter) -> scope file
    0.35: "0.35.CSV", 517: "517.CSV", 1055: "1055.CSV", 1531: "1531.CSV",
    2030: "2.03k.CSV", 2500: "2.5k.CSV", 3020: "3.02k.CSV", 3510: "3.51k.CSV",
    4020: "4.02k.CSV", 4500: "4.5k.CSV", 5010: "5.01k.CSV",
}
EXCLUDE = {
    2030: "echo larger than at 5 kOhm: resistance probably misread (20.3 k?)",
    0.35: "echo ~6% smaller than fit predicts: real end resistance likely ~5 ohm "
          "(switch, leads, potentiometer minimum)",
}
# Task 2.4c, direct method: fill in after the lab. Turn the potentiometer until the
# echo at node 20 vanishes and measure it (switch down): Z0_DIRECT. Then find the
# lowest and highest settings where the echo is still invisible: NULL_RANGE.
Z0_DIRECT = 147.0         # measured 19 Nov: echo vanished at 147 ohm (switch down, multimeter)
NULL_RANGE = None         # not measured (echo grows continuously); if None, uses 1% meter accuracy

NODE = 20
T_SECTION = 2.2          # us per section (from Task 2.3)
L, C = 330e-6, 15e-9


def cross(t, v, level, start=0):
    i = np.where((v[start:-1] < level) & (v[start + 1:] >= level))[0][0] + start
    return t[i] + (level - v[i]) * (t[i + 1] - t[i]) / (v[i + 1] - v[i])


def measure(path):
    """Works with files containing CH1 and CH2, or CH2 only (time, CH2)."""
    d = np.genfromtxt(path, delimiter=",", skip_header=1)
    d = d[~np.isnan(d).any(axis=1)]
    t = (d[:, 0] - d[0, 0]) * 1e6
    c2 = d[:, -1]                                                  # CH2 is always the last column
    b2 = np.median(c2[: len(c2) // 10])                            # baseline before the pulse
    if d.shape[1] >= 3:                                            # CH1 present: start search after
        c1 = d[:, 1]                                               # the pulse leaves node 0
        b1 = np.median(c1[: len(c1) // 10])
        start = np.searchsorted(t, cross(t, c1, b1 + 1.25))
    else:
        start = 0
    t2 = cross(t, c2, b2 + 1.25, start)                            # pulse reaches node 20
    t_echo = t2 + 2 * (40 - NODE) * T_SECTION                      # echo back at node 20
    flat = lambda a: np.mean(c2[(t >= a + 15) & (t < a + 40)]) - b2  # middle of the pulse
    return t, c2, t2, t_echo, flat(t2), flat(t_echo)


R_all, G_all, traces = [], [], {}
print("   R (ohm)   incident (V)   echo (V)   Gamma")
for R, f in sorted(FILES.items()):
    t, c2, t2, te, inc, ref = measure(f)
    R_all.append(R); G_all.append(ref / inc); traces[R] = (t, c2, t2, te)
    note = f"   <- excluded: {EXCLUDE[R]}" if R in EXCLUDE else ""
    print(f"  {R:8.2f}   {inc:10.3f}   {ref:8.3f}   {ref / inc:+.3f}{note}")
R_all, G_all = np.array(R_all), np.array(G_all)
use = np.array([R not in EXCLUDE for R in R_all])

model = lambda R, A, Z0: A * (R - Z0) / (R + Z0)
p, _ = curve_fit(model, R_all[use], G_all[use], p0=[0.9, 150])
scatter = np.std(G_all[use] - model(R_all[use], *p), ddof=2)       # Gamma noise from fit
p, cov = curve_fit(model, R_all[use], G_all[use], p0=p,
                   sigma=np.full(use.sum(), scatter), absolute_sigma=True)
A, Z0 = p
sA, sZ0 = np.sqrt(np.diag(cov))
s_meter = 0.01 * Z0                   # assumed 1% multimeter accuracy on R -> ~1% on Z0
sZ0_tot = np.hypot(sZ0, s_meter)
Z0_th = np.sqrt(L / C)
sZ0_th = Z0_th * 0.5 * np.hypot(0.20, 0.10)

print(f"\nFit: Z0 = {Z0:.1f} ± {sZ0:.1f} (fit) ± {s_meter:.1f} (meter) = {Z0:.0f} ± {sZ0_tot:.0f} ohm,"
      f"  loss factor A = {A:.3f} ± {sA:.3f}")
print(f"Theory: Z0 = sqrt(L/C) = {Z0_th:.0f} ± {sZ0_th:.0f} ohm")
print(f"Difference = {abs(Z0 - Z0_th):.1f} ohm = "
      f"{abs(Z0 - Z0_th) / np.hypot(sZ0_tot, sZ0_th):.2f} combined standard deviations")

if Z0_DIRECT is not None:
    sZ0_dir = 0.5 * (NULL_RANGE[1] - NULL_RANGE[0]) if NULL_RANGE else 0.01 * Z0_DIRECT
    how = ("half the range where the echo is invisible" if NULL_RANGE
           else "multimeter accuracy only, 1%")
    print(f"\nDirect null measurement: Z0 = {Z0_DIRECT:.0f} ± {sZ0_dir:.1f} ohm ({how})")
    for name, val, err in [("fit", Z0, sZ0_tot), ("theory", Z0_th, sZ0_th)]:
        print(f"  vs {name}: difference {abs(Z0_DIRECT - val):.1f} ohm = "
              f"{abs(Z0_DIRECT - val) / np.hypot(sZ0_dir, err):.2f} combined standard deviations")

# --- Plot 1: Gamma vs R --------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4.8))
RR = np.logspace(0, 4, 400)
ax.plot(RR, model(RR, A, Z0), "-", color="C0", label=f"fit: Z0 = {Z0:.0f} Ω, A = {A:.2f}")
ax.plot(RR, (RR - Z0_th) / (RR + Z0_th), "--", color="0.5", lw=1,
        label=f"ideal (no loss), Z0 = {Z0_th:.0f} Ω")
ax.plot(R_all[use], G_all[use], "o", color="C0", label="measured")
ax.plot(np.maximum(R_all[~use], 1), G_all[~use], "x", color="C3", ms=9, label="excluded")
ax.axhline(0, color="k", lw=0.6)
ax.axvline(Z0, color="C0", ls=":", lw=1)
if Z0_DIRECT is not None:
    ax.axvline(Z0_DIRECT, color="C2", lw=1.5, label=f"direct null: {Z0_DIRECT:.0f} Ω")
    if NULL_RANGE:
        ax.axvspan(*NULL_RANGE, color="C2", alpha=0.15)
ax.set(xscale="log", xlabel="End resistance R (Ω)", ylabel="Reflection coefficient (echo / incident)",
       title="Reflection at the end of the line vs end resistance")
ax.legend(fontsize=8, loc="lower right")
fig.tight_layout()
fig.savefig("gamma_vs_R.png", dpi=160)

# --- Plot 2: traces at node 20 for a few resistances -----------------------------
fig, ax = plt.subplots(figsize=(9, 4.5))
for R, col in [(0.35, "C3"), (517, "C1"), (1531, "C2"), (5010, "C0")]:
    t, c2, t2, te = traces[R]
    ax.plot(t - t2, c2, color=col, lw=0.9, label=f"R = {R:g} Ω")
ax.axvline(2 * (40 - NODE) * T_SECTION, color="k", ls="--", lw=0.8)
ax.text(2 * (40 - NODE) * T_SECTION + 2, 2.6, "echo from the end", fontsize=8)
ax.set(xlim=(-20, 160), xlabel="Time after incident pulse reaches node 20 (µs)", ylabel="CH2 (V)",
       title="Node 20: incident pulse, then the echo, for different end resistances")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("echo_traces_node20.png", dpi=160)
plt.show()