
# sim_models.py — reaction–diffusion models: Gray–Scott, Gierer–Meinhardt, FitzHugh–Nagumo
import numpy as np

def laplacian(Z):
    return (np.roll(Z, 1, 0) + np.roll(Z, -1, 0) + np.roll(Z, 1, 1) + np.roll(Z, -1, 1) - 4.0*Z)

def seed_disk(U, V, rng, r):
    N = U.shape[0]
    cx = N//2; cy = N//2
    y, x = np.ogrid[-r:r+1, -r:r+1]
    mask = x*x + y*y <= r*r
    U[cx-r:cx+r+1, cy-r:cy+r+1][mask] = 0.5 + 0.1 * rng.random(mask.sum())
    V[cx-r:cx+r+1, cy-r:cy+r+1][mask] = 0.25 + 0.1 * rng.random(mask.sum())

def init_fields(N, rng_seed=0, noise=0.02, seed_radius=None, ones_u=True):
    rng = np.random.default_rng(rng_seed)
    U = np.ones((N,N), dtype=np.float32) if ones_u else np.zeros((N,N), dtype=np.float32)
    V = np.zeros((N,N), dtype=np.float32)
    r = max(3, (N//6 if seed_radius is None else seed_radius))
    seed_disk(U, V, rng, r)
    U += noise * (rng.random((N,N)) - 0.5)
    V += noise * (rng.random((N,N)) - 0.5)
    return U, V

# ---------------- Gray–Scott ----------------
def run_gray_scott(params, steps=600, rng_seed=0, progress=None, render_every=0):
    N = int(params.get("size", 128))
    Du = float(params.get("Du", 0.16))
    Dv = float(params.get("Dv", 0.08))
    F  = float(params.get("F", 0.035))
    k  = float(params.get("k", 0.060))
    dt = float(params.get("dt", 1.0))
    U, V = init_fields(N, rng_seed=rng_seed, noise=0.02)
    for i in range(steps):
        Lu = laplacian(U); Lv = laplacian(V)
        uvv = U*V*V
        U += (Du*Lu - uvv + F*(1.0 - U)) * dt
        V += (Dv*Lv + uvv - (F + k)*V) * dt
        np.clip(U, 0.0, 1.0, out=U)
        np.clip(V, 0.0, 1.0, out=V)
        if progress and (render_every and (i % render_every==0 or i==steps-1)):
            progress(i+1, steps, V)
    return U, V, "V"

def gs_preset(goal="stripes", size=128, dt=1.0):
    if goal == "spots":
        F, k = 0.022, 0.051
    else:
        F, k = 0.035, 0.060
    return {"Du":0.16, "Dv":0.08, "F":F, "k":k, "dt":dt, "size":size}

# ---------------- Gierer–Meinhardt (scaled) ----------------
def run_gierer_meinhardt(params, steps=800, rng_seed=0, progress=None, render_every=0):
    N = int(params.get("size", 128))
    Du = float(params.get("Du", 0.2))
    Dv = float(params.get("Dv", 0.1))
    a  = float(params.get("a", 0.02))
    b  = float(params.get("b", 0.0))
    dt = float(params.get("dt", 0.2))
    eps = 1e-6
    U, V = init_fields(N, rng_seed=rng_seed, noise=0.01)
    for i in range(steps):
        Lu = laplacian(U); Lv = laplacian(V)
        uv = (U*U) / (V + eps)
        U += (Du*Lu + uv - U + a) * dt
        V += (Dv*Lv + U*U - V + b) * dt
        U = np.maximum(U, 0.0); V = np.maximum(V, 0.0)
        if progress and (render_every and (i % render_every==0 or i==steps-1)):
            progress(i+1, steps, U)
    return U, V, "U"

def gm_preset(goal="spots", size=128, dt=0.2):
    return {"Du":0.2, "Dv":0.1, "a":0.02, "b":0.0, "dt":dt, "size":size}

# ---------------- FitzHugh–Nagumo ----------------
def run_fitzhugh_nagumo(params, steps=1200, rng_seed=0, progress=None, render_every=0):
    N = int(params.get("size", 128))
    Du = float(params.get("Du", 0.1))
    Dv = float(params.get("Dv", 0.05))
    a  = float(params.get("a", 0.7))
    b  = float(params.get("b", 0.8))
    eps= float(params.get("eps", 0.08))
    I  = float(params.get("I", 0.0))
    dt = float(params.get("dt", 0.2))
    rng = np.random.default_rng(rng_seed)
    U = 0.1 * (rng.random((N,N)) - 0.5)
    V = 0.1 * (rng.random((N,N)) - 0.5)
    for i in range(steps):
        Lu = laplacian(U); Lv = laplacian(V)
        U += (Du*Lu + U - (U**3)/3.0 - V + I) * dt
        V += (Dv*Lv + eps*(U + a - b*V)) * dt
        if progress and (render_every and (i % render_every==0 or i==steps-1)):
            progress(i+1, steps, U)
    return U, V, "U"

def fhn_preset(mode="oscillation", size=128, dt=0.2):
    return {"Du":0.1, "Dv":0.05, "a":0.7, "b":0.8, "eps":0.08, "I":0.0, "dt":dt, "size":size}

def run_model(model, params, steps, rng_seed=0, progress=None, render_every=0):
    if model == "gray_scott":
        return run_gray_scott(params, steps=steps, rng_seed=rng_seed, progress=progress, render_every=render_every)
    elif model == "gierer_meinhardt":
        return run_gierer_meinhardt(params, steps=steps, rng_seed=rng_seed, progress=progress, render_every=render_every)
    elif model == "fhn":
        return run_fitzhugh_nagumo(params, steps=steps, rng_seed=rng_seed, progress=progress, render_every=render_every)
    else:
        raise ValueError(f"Unknown model: {model}")
