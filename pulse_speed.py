"""
Task 2.3b: square pulses on the open-ended transmission line.

Probe 1 (CH1) stays at node 0; probe 2 (CH2) is moved along the line. Save one
scope trace (CSV) per CH2 position and list them in TRACES below.

For each trace the script finds:
  - the delay between the pulse's rising edge at node 0 and at the probe
    (time where each trace crosses 50% of the pulse height)
  - the 10-90% rise time at the probe (how much the sharp edge has rounded)
  - the peak height at the probe
It then plots all traces, marks where the echo from the open end should arrive,
and fits delay against node number to get the delay per section and the speed.

Set SIMULATE = True to generate traces from a model of the line instead
(useful to test the script, and to compare with your real traces).
"""
import numpy as np
import matplotlib.pyplot as plt

TRACES = {            # CH2 node -> saved scope file
    5: "data/open_node05.csv",
    10: "data/open_node10.csv",
    20: "data/open_node20.csv",
    30: "data/open_node30.csv",
    40: "data/open_node40.csv",
}
SIMULATE = False
N_SECTIONS = 40
L, C = 330e-6, 15e-9                    # per section (H, F)
T_SECTION = np.sqrt(L * C)              # predicted delay per section, about 2.2 us


def load_trace(path):
    """Read a scope CSV: first column time (s), then CH1 and CH2 (V).
    Header and text lines are skipped, so most export formats work."""
    rows = []
    for line in open(path, errors="ignore"):
        parts = line.replace(";", ",").split(",")
        try:
            rows.append([float(p) for p in parts[:3]])
        except ValueError:
            continue
    t, ch1, ch2 = np.array(rows).T
    return t, ch1, ch2


def crossing_time(t, v, level, start=0):
    """First time after index `start` where v rises through `level` (interpolated)."""
    i = np.where((v[start:-1] < level) & (v[start + 1:] >= level))[0]
    if len(i) == 0:
        return np.nan
    i = i[0] + start
    return t[i] + (level - v[i]) * (t[i + 1] - t[i]) / (v[i + 1] - v[i])


def simulate(node, r_loss=2.0, t_end=300e-6):
    """Lumped LC ladder driven by a 0-5 V, 50 us pulse through the generator's
    50 ohm output; open far end. r_loss = series resistance per inductor."""
    from scipy.integrate import solve_ivp
    n = N_SECTIONS + 1                                   # nodes 0..40

    def vs(t):
        return 5.0 if 1e-6 < t < 51e-6 else 0.0

    def rhs(t, y):
        v, i = y[:n], y[n:]                              # node voltages, inductor currents
        dv = np.zeros(n)
        dv[0] = ((vs(t) - v[0]) / 50.0 - i[0]) / C
        dv[1:-1] = (i[:-1] - i[1:]) / C
        dv[-1] = i[-1] / C                               # open end: no current out
        di = (v[:-1] - v[1:] - r_loss * i) / L
        return np.concatenate([dv, di])

    t = np.linspace(0, t_end, 6000)
    sol = solve_ivp(rhs, (0, t_end), np.zeros(2 * n - 1), t_eval=t, max_step=0.2e-6)
    return t, sol.y[0], sol.y[node]


# ---------------------------------------------------------------------------
results = []
fig, ax = plt.subplots(len(TRACES), 1, figsize=(9, 1.9 * len(TRACES)), sharex=True)
for k, (node, path) in enumerate(sorted(TRACES.items())):
    t, ch1, ch2 = simulate(node) if SIMULATE else load_trace(path)
    t = t * 1e6                                          # microseconds
    height = ch1.max() - ch1.min()
    half = ch1.min() + 0.5 * height
    t1 = crossing_time(t, ch1, half)
    # CH2 levels measured from its flat baseline before the pulse arrives
    # (not its minimum, which ringing and reflections can push below zero)
    base2 = np.median(ch2[t < t1]) if np.any(t < t1) else ch2[0]
    top2 = ch2.max()
    level = lambda f: base2 + f * (top2 - base2)
    t2 = crossing_time(t, ch2, level(0.5))
    i0 = np.searchsorted(t, t1)                          # search after the pulse starts
    rise = crossing_time(t, ch2, level(0.9), i0) - crossing_time(t, ch2, level(0.1), i0)
    results.append((node, t2 - t1, rise, ch2.max()))

    ax[k].plot(t, ch1, color="0.6", lw=1, label="CH1 (node 0)")
    ax[k].plot(t, ch2, color="C0", lw=1.2, label=f"CH2 (node {node})")
    echo = t1 + (2 * N_SECTIONS - node) * T_SECTION * 1e6   # predicted echo from open end
    ax[k].axvline(echo, color="C3", ls="--", lw=1, label="predicted echo arrival")
    ax[k].set_ylabel("V")
    ax[k].legend(fontsize=7, loc="upper right")
ax[-1].set_xlabel("Time (µs)")
fig.suptitle("Square pulse along the open-ended line")
fig.tight_layout()
fig.savefig("pulse_traces.png", dpi=160)

nodes, delays, rises, peaks = map(np.array, zip(*results))
print(" node   delay (µs)   delay/section (µs)   rise time (µs)   peak (V)")
for n_, d, r, p in results:
    print(f"  {n_:3d}   {d:9.2f}   {d / n_:14.2f}   {r:16.2f}   {p:8.2f}")

(slope, offset), cov = np.polyfit(nodes, delays, 1, cov=True)
err = np.sqrt(cov[0, 0])
print(f"\nFit: delay per section = {slope:.3f} ± {err:.3f} µs "
      f"(theory sqrt(LC) = {T_SECTION * 1e6:.2f} µs)")
print(f"Speed = {1e6 / slope:.3g} ± {1e6 * err / slope**2:.2g} sections/s "
      f"(theory 1/sqrt(LC) = {1 / T_SECTION:.3g})")

fig2, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
a1.plot(nodes, delays, "o")
xx = np.linspace(0, N_SECTIONS, 50)
a1.plot(xx, slope * xx + offset, "-", label=f"fit: {slope:.2f} µs/section")
a1.plot(xx, xx * T_SECTION * 1e6, "k--", lw=1, label="theory √(LC)")
a1.set(xlabel="Probe position (section)", ylabel="Delay after node 0 (µs)", title="Arrival time")
a1.legend(fontsize=8)
a2.plot(nodes, rises, "s")
a2.set(xlabel="Probe position (section)", ylabel="10–90% rise time (µs)",
       title="Edge gets less sharp along the line")
fig2.tight_layout()
fig2.savefig("pulse_delay_fit.png", dpi=160)
plt.show()