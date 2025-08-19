
# pattern_analytics.py — classifier, ensemble, sensitivity, phase scan
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def classify_pattern(field):
    F = np.fft.fftshift(np.abs(np.fft.fft2(field)))
    F /= (F.max() + 1e-8)
    N = F.shape[0]; cy = cx = N//2
    y, x = np.indices(F.shape)
    r = np.sqrt((x-cx)**2 + (y-cy)**2).astype(int)
    radial = np.bincount(r.ravel(), F.ravel())
    counts = np.bincount(r.ravel()) + 1e-8
    radial = radial / counts
    ring_peak = radial[3:].max() if radial.size > 3 else 0.0

    thresh = 0.5
    yy, xx = np.where(F > thresh)
    if yy.size:
        ang = np.arctan2(yy - cy, xx - cx)
        bins = np.histogram(ang, bins=30, range=(-np.pi, np.pi))[0]
        orient_peak = bins.max() / (bins.sum() + 1e-8)
    else:
        orient_peak = 0.0

    if orient_peak > 0.2 and ring_peak < 0.15:
        return "stripes"
    if ring_peak > 0.22:
        return "spots"
    return "labyrinth"

def ensemble_label(sim_runner, params, steps=400, seeds=5):
    labels = []
    for s in range(seeds):
        U, V, field_name = sim_runner(params, steps=steps, rng_seed=s)
        field = V if field_name == "V" else U
        labels.append(classify_pattern(field))
    c = Counter(labels)
    label, n = c.most_common(1)[0]
    confidence = n / seeds
    return label, confidence, dict(c)

def local_sensitivity_gray_scott(sim_runner, params, steps=300, delta=0.002):
    U0, V0, fn0 = sim_runner(params, steps=steps, rng_seed=0); base_field = V0 if fn0=="V" else U0
    base_label = classify_pattern(base_field)
    out = []
    for key in ("F","k"):
        p_lo = dict(params); p_hi = dict(params)
        p_lo[key] = p_lo[key] - delta
        p_hi[key] = p_hi[key] + delta
        U1, V1, fn1 = sim_runner(p_lo, steps=steps, rng_seed=1); field_lo = V1 if fn1=="V" else U1
        U2, V2, fn2 = sim_runner(p_hi, steps=steps, rng_seed=2); field_hi = V2 if fn2=="V" else U2
        out.append({"param": key, "base": base_label, "minus_delta": classify_pattern(field_lo), "plus_delta": classify_pattern(field_hi)})
    return out

def phase_scan_gray_scott(sim_runner, base_params, F_list, k_list, steps=250):
    labels = []
    for F in F_list:
        row = []
        for k in k_list:
            p = dict(base_params); p.update({"F": float(F), "k": float(k)})
            U, V, fn = sim_runner(p, steps=steps, rng_seed=0)
            field = V if fn=="V" else U
            row.append(classify_pattern(field))
        labels.append(row)
    return labels

def render_phase_heatmap(F_list, k_list, labels):
    cat = {"stripes":0, "spots":1, "labyrinth":2}
    import numpy as np
    Z = np.array([[cat.get(lbl,2) for lbl in row] for row in labels], dtype=float)
    cmap = plt.get_cmap("viridis", 3)
    fig, ax = plt.subplots(figsize=(5,4))
    im = ax.imshow(Z, origin="lower", aspect="auto", cmap=cmap, vmin=-0.5, vmax=2.5,
                   extent=[min(k_list), max(k_list), min(F_list), max(F_list)])
    ax.set_xlabel("k"); ax.set_ylabel("F"); ax.set_title("Gray–Scott phase scan")
    cbar = fig.colorbar(im, ticks=[0,1,2])
    cbar.ax.set_yticklabels(["stripes","spots","labyrinth"])
    return fig
