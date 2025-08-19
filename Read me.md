Duality Pattern Simulator — Streamlit

A Streamlit app that turns a sentence like **“I feel torn between ambition and rest.”** into
a living **reaction–diffusion (Gray–Scott)** pattern and a **TREE** insight block.

## 🚀 Quickstart (local)

```bash
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows:
# .venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## 🌐 Deploy to Streamlit Community Cloud
1. Push this folder to a **GitHub repo** (root should contain `app.py` and `requirements.txt`).
2. On https://share.streamlit.io, select your repo/branch, set **Main file path** to `app.py`.
3. Click **Deploy**.

## 🎛️ Tips
- **Stripes**: F≈0.035, k≈0.060  
- **Spots**: F≈0.022, k≈0.051  
- Larger **N** gives more detail (but is slower).  
- Use smaller **N** and fewer **steps** on laptops/phones.
