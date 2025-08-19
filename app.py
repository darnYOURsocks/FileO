# app.py — Streamlit app for Gray–Scott "duality" patterns (mobile-friendly submit)
import streamlit as st
import numpy as np
import re
import matplotlib.pyplot as plt
import io
import json

# ========== 1) Pattern detection ==========
def detect_duality(text: str):
    """Find 'between X and Y' (case-insensitive)."""
    match = re.search(r"\bbetween\s+(.+?)\s+and\s+(.+?)([\.!\?]|$)", text, re.IGNORECASE)
    if match:
        return (match.group(1).strip(), match.group(2).strip())
    return None

# ========== 2) Sim parameters (presets) ==========
def stripe_params(size=128, dt=1.0):
    # Classic Gray–Scott "stripe-ish" regime
    return dict(Du=0.16, Dv=0.08, F=0.035, k=0.060, dt=dt, size=size)

def spot_params(size=128, dt=1.0):
    # "Spots" regime
    p = stripe_params(size=size, dt=dt)
    p["F"] = 0.022
    p["k"] = 0.051
    return p

# ========== 3) Simulation core (NumPy) ==========
def laplacian(Z):
    """5‑point Laplacian on a torus via roll (periodic BC)."""
    return (
        np.roll(Z, 1, axis=0) + np.roll(Z, -1, axis=0) +
        np.roll(Z, 1, axis=1) + np.roll(Z, -1, axis=1) - 4.0 * Z
    )

def initialize_grids(N, rng=None):
    """Start U≈1, V≈0 with a noisy disk seed at center."""
    rng = rng or np.random.default_rng()
    U = np.ones((N, N), dtype=np.float32)
    V = np.zeros((N, N), dtype=np.float32)

    r = max(3, N // 6)
    cx = N // 2
    cy = N // 2
    y, x = np.ogrid[-r:r+1, -r:r+1]
    mask = x * x + y * y <= r * r
    U[cx - r:cx + r + 1, cy - r:cy + r + 1][mask] = 0.5 + 0.1 * rng.random(mask.sum())
    V[cx - r:cx + r + 1, cy - r:cy + r + 1][mask] = 0.25 + 0.1 * rng.random(mask.sum())
    U += 0.02 * (rng.random((N, N)) - 0.5)
    V += 0.02 * (rng.random((N, N)) - 0.5)
    np.clip(U, 0.0, 1.0, out=U)
    np.clip(V, 0.0, 1.0, out=V)
    return U, V

def step_gray_scott(U, V, Du, Dv, F, k, dt):
    Lu = laplacian(U)
    Lv = laplacian(V)
    uvv = U * V * V
    U += (Du * Lu - uvv + F * (1.0 - U)) * dt
    V += (Dv * Lv + uvv - (F + k) * V) * dt
    np.clip(U, 0.0, 1.0, out=U)
    np.clip(V, 0.0, 1.0, out=V)

def simulate(params, steps, progress_cb=None, render_every=100, throttle_ms=0):
    N = params["size"]
    U, V = initialize_grids(N)
    Du, Dv, F, k, dt = params["Du"], params["Dv"], params["F"], params["k"], params["dt"]
    for i in range(steps):
        step_gray_scott(U, V, Du, Dv, F, k, dt)
        if progress_cb and (i % render_every == 0 or i == steps - 1):
            progress_cb(i + 1, steps, V)
        if throttle_ms:
            import time
            time.sleep(throttle_ms / 1000.0)
    return U, V

# ========== 4) Tiny FFT pattern classifier (optional label) ==========
def classify_pattern(V):
    Fmag = np.fft.fftshift(np.abs(np.fft.fft2(V)))
    Fmag = Fmag / (Fmag.max() + 1e-8)
    N = Fmag.shape[0]
    cy = cx = N // 2
    y, x = np.indices(Fmag.shape)
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(np.int32)
    radial = np.bincount(r.ravel(), Fmag.ravel(), minlength=r.max() + 1)
    counts = np.bincount(r.ravel(), minlength=r.max() + 1) + 1e-8
    radial /= counts
    ring_peak = radial[3:].max() if radial.size > 3 else 0.0

    thresh = 0.5
    yy, xx = np.where(Fmag > thresh)
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

# ========== 5) TREE response ==========
def tree_response(duality, params, label):
    a, b = duality
    return {
        "translation": f"The tension between '{a}' and '{b}' can stabilise into structure — here it looks like **{label}**.",
        "recommendation": f"Alternate deep blocks for '{a}' and '{b}' to discover a rhythm rather than a tug-of-war.",
        "explanation": f"Simulated with Gray–Scott. Feed (F={params['F']}) and kill (k={params['k']}) steer the regime.",
        "experiment": f"For one week, log when you lean into '{a}' vs '{b}'. Adjust block length to stabilise energy.",
    }

# ========== 6) Streamlit UI (mobile‑friendly) ==========
st.set_page_config(page_title="Duality Pattern Simulator", page_icon="🌊", layout="wide")

st.title("🌊 Duality Pattern Simulator (Gray–Scott)")
st.caption("Type a sentence like “I feel torn between ambition and rest.” — then tap **Run Simulation**.")

# Inputs (left) · Results (right)
colL, colR = st.columns([1, 1])

with colL:
    st.subheader("Input")
    examples = [
        "I feel torn between ambition and rest.",
        "Between being social and needing solitude.",
        "Caught between tradition and innovation.",
        "Between perfectionism and acceptance.",
        "Struggling between independence and connection.",
    ]
    ex_choice = st.selectbox("Try an example (optional)", ["—"] + examples, index=0)

    user_text = st.text_input(
        "Enter your duality (use 'between X and Y')",
        value=examples[0] if ex_choice == "—" else ex_choice,
        placeholder="I feel torn between A and B",
        help="On mobile, type here, then tap the button below."
    )

    st.markdown("**Preset**")
    preset = st.radio(
        "Choose a pattern tendency",
        ["Stripes (classic)", "Spots (islands)"],
        horizontal=True,
        label_visibility="collapsed",
    )

    st.markdown("**Parameters**")
    N = st.slider("Grid size (N×N)", 64, 192, 128, step=32)
    steps = st.slider("Simulation steps", 100, 1500, 600, step=50)
    dt = st.select_slider("Time step (dt)", [0.5, 1.0, 1.5], value=1.0)
    throttle = st.checkbox("Slow preview (throttle)", value=False, help="Adds tiny delays so you can watch it form.")
    throttle_ms = 10 if throttle else 0

    # 🌟 Mobile‑friendly explicit submit:
    run_clicked = st.button("Run Simulation", use_container_width=True, type="primary")

with colR:
    st.subheader("Pattern")
    canvas = st.empty()
    prog = st.progress(0, text="Waiting…")

# Handle click
if run_clicked:
    dual = detect_duality(user_text or "")
    if not dual:
        st.error("No duality detected. Try a phrase like “between focus and distraction”.")
    else:
        # choose preset
        params = (stripe_params(size=N, dt=dt) if preset.startswith("Stripes")
                  else spot_params(size=N, dt=dt))
        # live progress callback
        def on_progress(done, total, Vfield):
            prog.progress(done / total, text=f"Simulating… {done}/{total}")
            fig, ax = plt.subplots(figsize=(5.0, 5.0))
            ax.imshow(Vfield, cmap="viridis")
            ax.set_title(f"{dual[0]} vs {dual[1]} — step {done}/{total}")
            ax.axis("off")
            canvas.pyplot(fig, clear_figure=True)
            plt.close(fig)

        # run sim
        U, V = simulate(params, steps=steps, progress_cb=on_progress, render_every=max(1, steps // 50), throttle_ms=throttle_ms)
        prog.progress(1.0, text="Done ✓")
        label = classify_pattern(V)

        # Show downloads (PNG + JSON)
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        ax.imshow(V, cmap="viridis")
        ax.set_title(f"Pattern: {dual[0]} vs {dual[1]}  ·  {label}")
        ax.axis("off")
        png_buf = io.BytesIO()
        fig.savefig(png_buf, format="png", dpi=180, bbox_inches="tight", pad_inches=0)
        png_buf.seek(0)
        st.download_button("Download image (PNG)", data=png_buf, file_name="duality_pattern.png", mime="image/png")
        plt.close(fig)

        artifact = {
            "input": user_text,
            "duality": {"a": dual[0], "b": dual[1]},
            "params": params,
            "label": label,
            "metrics": {"mean_abs_grad": float(np.mean(np.abs(np.gradient(V))))}
        }
        st.download_button("Download run as JSON", data=json.dumps(artifact, indent=2), file_name="run.json", mime="application/json")

        # TREE block
        st.subheader("🌳 TREE Insights")
        tree = tree_response(dual, params, label)
        st.markdown(f"**Translation** — {tree['translation']}")
        st.markdown(f"**Recommendation** — {tree['recommendation']}")
        st.markdown(f"**Explanation** — {tree['explanation']}")
        st.markdown(f"**Experiment** — {tree['experiment']}")

        st.divider()
        st.markdown("### 🧬 How it works")
        st.markdown(
            "We simulate a **Gray–Scott reaction–diffusion** system on an N×N grid. "
            "With the chosen feed (F) and kill (k) rates, the two fields interact and diffuse to form "
            "**stripes, spots, or labyrinths**. A tiny FFT heuristic labels the pattern."
        )
else:
    st.info("Type your sentence, adjust sliders if you like, then tap **Run Simulation**.")
