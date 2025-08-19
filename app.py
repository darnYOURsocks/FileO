# streamlit_pattern_app.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import re

# ===== 1. PATTERN ENGINE =====
def detect_duality(text: str):
    """Simple regex to find 'between X and Y'"""
    match = re.search(r"\bbetween\s+(.+?)\s+and\s+(.+?)([\.!\?]|$)", text, re.IGNORECASE)
    if match:
        return (match.group(1).strip(), match.group(2).strip())
    return None

# ===== 2. DOMAIN MAPPER =====
def get_params():
    """Parameters for Gray-Scott that tend to form stripes"""
    return {
        "Du": 0.16, "Dv": 0.08, "F": 0.035, "k": 0.060,
        "dt": 1.0, "steps": 500, "size": 128
    }

# ===== 3. SIMULATION LAYER =====
def laplacian(Z):
    return (np.roll(Z, 1, axis=0) + np.roll(Z, -1, axis=0) +
            np.roll(Z, 1, axis=1) + np.roll(Z, -1, axis=1) - 4 * Z)

def run_gray_scott(params):
    N = params["size"]
    Du, Dv, F, k, dt, steps = params["Du"], params["Dv"], params["F"], params["k"], params["dt"], params["steps"]

    U = np.ones((N, N))
    V = np.zeros((N, N))

    r = 20
    U[N//2-r:N//2+r, N//2-r:N//2+r] = 0.50
    V[N//2-r:N//2+r, N//2-r:N//2+r] = 0.25

    U += 0.05 * np.random.random((N, N))
    V += 0.05 * np.random.random((N, N))

    for _ in range(steps):
        Lu, Lv = laplacian(U), laplacian(V)
        uvv = U * V * V
        U = U + (Du * Lu - uvv + F * (1 - U)) * dt
        V = V + (Dv * Lv + uvv - (F + k) * V) * dt

    return U, V

# ===== 4. TRANSLATION =====
def generate_tree_response(duality, params):
    a, b = duality
    metaphor = f"Alternating '{a}' and '{b}' may form a stable rhythm, like biological stripes."
    recommendation = f"Lean into this rhythm instead of fighting it. Schedule time for both."
    explanation = f"Feed (F={params['F']}), kill (k={params['k']}) shape whether patterns stabilize."
    experiment = f"Track your natural shifts between '{a}' and '{b}' for a week."
    return f"""
**TREE Response**
- **Translation:** {metaphor}  
- **Recommendation:** {recommendation}  
- **Explanation:** {explanation}  
- **Experiment:** {experiment}  
"""

# ===== 5. STREAMLIT APP =====
st.title("🌱 Pattern Engine Demo")
st.write("Type something like: *I feel torn between ambition and rest.*")

# ✅ FIX: Use text_input (Enter submits immediately)
user_input = st.text_input("Enter your thought:", "")

if user_input:
    duality = detect_duality(user_input)
    if not duality:
        st.warning("No duality found. Try a phrase with 'between X and Y'.")
    else:
        st.success(f"Detected duality: {duality[0]} vs. {duality[1]}")
        params = get_params()
        U, V = run_gray_scott(params)

        response = generate_tree_response(duality, params)
        st.markdown(response)

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(V, cmap="viridis")
        ax.set_title(f"{duality[0]} vs {duality[1]}")
        ax.axis("off")
        st.pyplot(fig)
