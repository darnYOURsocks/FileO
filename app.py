# app.py — Streamlit app for Gray–Scott "duality" patterns
import streamlit as st
import numpy as np
import re
import matplotlib.pyplot as plt

# ========== Core from the user's simple_pattern.py (adapted) ==========

def detect_duality(text: str):
    """Simple regex to find 'between X and Y'"""
    match = re.search(r"\bbetween\s+(.+?)\s+and\s+(.+?)([\.!\?]|$)", text, re.IGNORECASE)
    if match:
        return (match.group(1).strip(), match.group(2).strip())
    return None

def laplacian(Z):
    """5-point Laplacian for diffusion."""
    return (
        np.roll(Z, 1, axis=0) + np.roll(Z, -1, axis=0) +
        np.roll(Z, 1, axis=1) + np.roll(Z, -1, axis=1) - 4 * Z
    )

def run_gray_scott(Du, Dv, F, k, dt, steps, N, seed_radius=20, rng_seed=0, progress_cb=None, render_every=100):
    """Run Gray–Scott RD model and return (U, V)."""
    rng = np.random.default_rng(rng_seed)
    U = np.ones((N, N), dtype=np.float64)
    V = np.zeros((N, N), dtype=np.float64)

    r = min(seed_radius, max(3, N//6))
    cx = N // 2
    cy = N // 2
    U[cx-r:cx+r, cy-r:cy+r] = 0.50
    V[cx-r:cx+r, cy-r:cy+r] = 0.25

    U += 0.05 * rng.random((N, N))
    V += 0.05 * rng.random((N, N))

    for i in range(steps):
        Lu = laplacian(U)
        Lv = laplacian(V)
        uvv = U * V * V
        U = U + (Du * Lu - uvv + F * (1 - U)) * dt
        V = V + (Dv * Lv + uvv - (F + k) * V) * dt
        np.clip(U, 0, 1, out=U)
        np.clip(V, 0, 1, out=V)

        if progress_cb and (i % render_every == 0 or i == steps - 1):
            progress_cb(i + 1, steps, V)

    return U, V

def generate_tree_response(duality, params):
    a, b = duality
    metaphor = f"Alternating '{a}' and '{b}' may form a stable rhythm, similar to biological patterns like stripes or waves."
    recommendation = "Consider leaning into this natural rhythm instead of fighting it. Schedule dedicated time for both."
    explanation = (f"This uses a reaction–diffusion (Gray–Scott) model. Feed (F={params['F']}) and kill (k={params['k']}) control pattern type.")
    experiment = f"Observe your own cycles for a week. Identify natural periods of '{a}' and '{b}'."
    return {
        "translation": metaphor,
        "recommendation": recommendation,
        "explanation": explanation,
        "experiment": experiment,
    }

# ========== Streamlit UI ==========

st.set_page_config(page_title="Duality Pattern Simulator", page_icon="🌊", layout="wide")

st.title("🌊 Duality Pattern Simulator (Gray–Scott)")
st.write(
    """Transform life’s tensions into beautiful patterns using reaction–diffusion mathematics.
Type a sentence like **“I feel torn between ambition and rest.”** and watch a pattern form."""
)

col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("Input")
    default_text = "I feel torn between ambition and rest."
    text = st.text_area("Enter your duality (use 'between X and Y'):", value=default_text, height=90)
    detected = detect_duality(text)
    if detected:
        st.success(f"Detected duality: **{detected[0]}** vs **{detected[1]}**")
    else:
        st.warning("No duality detected. Try the format: 'between X and Y'.")

    st.subheader("Parameters")
    N = st.slider("Grid size (N×N)", 64, 256, 128, step=16)
    steps = st.slider("Simulation steps", 100, 3000, 1000, step=100)
    Du = st.slider("Du (diffusion of U)", 0.01, 0.5, 0.16, step=0.01)
    Dv = st.slider("Dv (diffusion of V)", 0.01, 0.5, 0.08, step=0.01)
    F = st.slider("Feed F", 0.005, 0.08, 0.035, step=0.001)
    k = st.slider("Kill k", 0.02, 0.08, 0.060, step=0.001)
    dt = st.select_slider("dt (time step)", [0.5, 1.0, 1.5, 2.0], value=1.0)
    seed_radius = st.slider("Seed radius", 3, max(6, N//4), max(10, N//6))

    run = st.button("Generate Pattern", type="primary", use_container_width=True)

with col_right:
    st.subheader("Pattern")
    canvas = st.empty()  # placeholder for image
    prog = st.progress(0, text="Waiting…")

    if run and detected:
        params = {"Du": Du, "Dv": Dv, "F": F, "k": k, "dt": float(dt), "steps": steps}
        def on_progress(done, total, Vfield):
            prog.progress(done / total, text=f"Simulating… {done}/{total}")
            fig, ax = plt.subplots(figsize=(5, 5))
            im = ax.imshow(Vfield, cmap="viridis")
            ax.set_title(f"{detected[0]} vs {detected[1]} — step {done}/{total}")
            ax.axis("off")
            canvas.pyplot(fig, clear_figure=True)
            plt.close(fig)

        U, V = run_gray_scott(Du, Dv, F, k, float(dt), steps, N, seed_radius=seed_radius, rng_seed=0, progress_cb=on_progress, render_every=max(1, steps//50))
        prog.progress(1.0, text="Done ✓")

        st.caption("Tip: try F=0.022, k=0.051 for spots; F≈0.035, k≈0.060 for stripes.")

        st.subheader("🌳 TREE Insights")
        tree = generate_tree_response(detected, {"F": F, "k": k})
        st.markdown(f"**Translation**: {tree['translation']}")
        st.markdown(f"**Recommendation**: {tree['recommendation']}")
        st.markdown(f"**Explanation**: {tree['explanation']}")
        st.markdown(f"**Experiment**: {tree['experiment']}")
    else:
        st.info("Choose parameters and click **Generate Pattern**.")

st.markdown("---")
st.subheader("🧠 The Science")
st.write(
    """We simulate the **Gray–Scott reaction–diffusion** system. With specific feed (F) and kill (k) values,
two interacting fields (U as activator, V as inhibitor) can self‑organise into **stripes, spots, or labyrinths**.
The metaphor: tensions like *“A vs B”* may stabilise into workable **rhythms** rather than deadlocks."""
)
