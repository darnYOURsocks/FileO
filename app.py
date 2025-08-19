
# app.py — Streamlit UI for deeper, more "real" functions
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import io, json, re

from sim_models import run_model, gs_preset, gm_preset, fhn_preset
from pattern_analytics import classify_pattern, ensemble_label, local_sensitivity_gray_scott, phase_scan_gray_scott, render_phase_heatmap

def detect_duality(text: str):
    m = re.search(r"\bbetween\s+(.+?)\s+and\s+(.+?)([\.!\?]|$)", text, re.IGNORECASE)
    return (m.group(1).strip(), m.group(2).strip()) if m else None

def tree_response(duality, model, params, label, confidence=None):
    a, b = duality
    conf_txt = f" (confidence {confidence:.0%})" if confidence is not None else ""
    translation = f"The tension between '{a}' and '{b}' produced **{label}**{conf_txt} using **{model.replace('_',' ')}** dynamics."
    recommendation = f"Alternate focused blocks for '{a}' and '{b}' until your energy stabilises; tweak parameters if outcomes feel off."
    explanation = f"We simulated {model.replace('_',' ')}; key knobs: " + ", ".join([k for k in params.keys() if k in ('F','k','a','b','Du','Dv')])
    experiment = f"Log a week of cycles. If small changes flip patterns, you're near a bifurcation — adjust cadence gently."
    return dict(translation=translation, recommendation=recommendation, explanation=explanation, experiment=experiment)

st.set_page_config(page_title="Duality Patterns — Deeper", page_icon="🧪", layout="wide")
st.title("🧪 Duality Pattern Lab (Deeper, No Cheating)")
st.caption("Real PDEs, measured patterns, ensembles & sensitivity.")

left, right = st.columns([1,1])

with left:
    st.subheader("Input")
    default_text = "I feel torn between ambition and rest."
    text = st.text_area("Use 'between X and Y'", value=default_text, height=90)
    dual = detect_duality(text)
    if dual:
        st.success(f"Detected: **{dual[0]}** vs **{dual[1]}**")
    else:
        st.warning("No duality detected yet.")

    st.subheader("Model & Preset")
    model = st.selectbox("Model", ["gray_scott", "gierer_meinhardt", "fhn"], index=0, format_func=lambda s: s.replace("_"," ").title())
    preset = st.selectbox("Preset", ["auto", "stripes/spots", "oscillation/waves"], index=0)

    st.subheader("Controls")
    size = st.slider("Grid size (N×N)", 64, 192, 128, step=32)
    steps = st.slider("Simulation steps", 100, 1500, 600, step=50)

    with st.expander("Advanced"):
        ensemble_on = st.checkbox("Ensemble (multi-seed) label", value=True)
        seeds = st.slider("Ensemble seeds", 3, 9, 5, step=2, disabled=not ensemble_on)
        sensitivity_on = st.checkbox("Sensitivity (Gray–Scott F,k)", value=False)
        phase_on = st.checkbox("Phase scan (Gray–Scott F,k)", value=False)
        throttle = st.checkbox("Slow preview (throttle)")

    run = st.button("Run", type="primary", use_container_width=True)

with right:
    st.subheader("Pattern")
    canvas = st.empty()
    prog = st.progress(0, text="Idle")

def choose_params(model, preset, size):
    if model == "gray_scott":
        goal = "spots" if "spot" in preset else "stripes"
        return gs_preset(goal=goal, size=size)
    if model == "gierer_meinhardt":
        return gm_preset(size=size)
    if model == "fhn":
        return fhn_preset(size=size)
    raise ValueError("unknown model")

if run and dual:
    params = choose_params(model, preset, size)
    def on_progress(done, total, field):
        if throttle:
            fig, ax = plt.subplots(figsize=(5,5))
            ax.imshow(field, cmap="viridis"); ax.axis("off")
            ax.set_title(f"{dual[0]} vs {dual[1]} — {done}/{total}")
            canvas.pyplot(fig, clear_figure=True); plt.close(fig)
        prog.progress(done/total, text=f"Simulating… {done}/{total}")

    U, V, field_name = run_model(model, params, steps=steps, rng_seed=0, progress=on_progress, render_every=max(1, steps//50))
    field = V if field_name == "V" else U
    label = classify_pattern(field)

    conf = None; counts = None
    if ensemble_on:
        def sim_runner(p, steps=steps, rng_seed=0):
            return run_model(model, p, steps=steps, rng_seed=rng_seed)
        label, conf, counts = ensemble_label(sim_runner, params, steps=min(steps, 500), seeds=seeds)

    fig, ax = plt.subplots(figsize=(5.5,5.5))
    ax.imshow(field, cmap="viridis"); ax.axis("off")
    title_lab = f"{dual[0]} vs {dual[1]} · {label}"
    if conf is not None: title_lab += f" ({conf:.0%})"
    ax.set_title(title_lab)
    canvas.pyplot(fig, clear_figure=True); plt.close(fig)
    prog.progress(1.0, text="Done ✓")

    buf = io.BytesIO()
    fig2, ax2 = plt.subplots(figsize=(5.5,5.5)); ax2.imshow(field, cmap="viridis"); ax2.axis("off")
    fig2.savefig(buf, format="png", dpi=180, bbox_inches="tight", pad_inches=0)
    st.download_button("Download image (PNG)", data=buf.getvalue(), file_name="pattern.png", mime="image/png")
    plt.close(fig2)

    artifact = {"input": text, "duality": {"a": dual[0], "b": dual[1]}, "model": model, "params": params,
                "label": label, "confidence": conf, "counts": counts,
                "metrics": {"mean_abs_grad": float(np.mean(np.abs(np.gradient(field))))}}
    st.download_button("Download run (JSON)", data=json.dumps(artifact, indent=2), file_name="run.json", mime="application/json")

    st.subheader("🌳 TREE Insights")
    tree = tree_response(dual, model, params, label, confidence=conf)
    st.markdown(f"**Translation** — {tree['translation']}")
    st.markdown(f"**Recommendation** — {tree['recommendation']}")
    st.markdown(f"**Explanation** — {tree['explanation']}")
    st.markdown(f"**Experiment** — {tree['experiment']}")

    if sensitivity_on and model == "gray_scott":
        st.divider(); st.markdown("### 🔎 Local sensitivity (F,k)")
        def sim_runner(p, steps=steps, rng_seed=0):
            return run_model(model, p, steps=steps, rng_seed=rng_seed)
        out = local_sensitivity_gray_scott(sim_runner, params, steps=min(steps, 400))
        st.table(out)

    if phase_on and model == "gray_scott":
        st.divider(); st.markdown("### 🗺️ Phase scan (F,k)")
        F_list = np.linspace(max(0.01, params['F']-0.01), params['F']+0.01, 6)
        k_list = np.linspace(max(0.03, params['k']-0.01), params['k']+0.01, 6)
        def gs_runner(p, steps=250, rng_seed=0):
            return run_model("gray_scott", p, steps=steps, rng_seed=rng_seed)
        labels = phase_scan_gray_scott(gs_runner, params, F_list, k_list, steps=250)
        fig3 = render_phase_heatmap(F_list, k_list, labels)
        st.pyplot(fig3, use_container_width=True)

else:
    st.info("Enter a duality, pick a model, and press **Run**.")

st.markdown("---")
st.markdown("This lab uses real PDEs and shows measured outcomes (labels, ensembles, sensitivity). No magic, just maths.")
