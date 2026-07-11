#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyse post‑simulation des données DOP – Version unifiée
Utilise par défaut dt = 1.0 s, t_transient = 5000 s
(si ces valeurs sont présentes dans le fichier .npz, elles sont lues automatiquement)
Calcule : λ₁ (Rosenstein), d₂ (Grassberger-Procaccia), return map, Poincaré, spectre.
"""

import sys
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, welch, argrelextrema
from scipy.spatial import KDTree
from scipy.spatial.distance import pdist
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PARAMÈTRES PAR DÉFAUT (si non trouvés dans le fichier .npz)
# ============================================================================
DEFAULT_DT = 1.0            # pas d'échantillonnage (s)
DEFAULT_T_TRANSIENT = 5000  # durée du transitoire (s)

EMBED_DIM_LYAP = 5
EMBED_DELAY_LYAP = 15
MIN_NEIGHBORS = 20
MAX_POINTS_LYAP = 40000

EMBED_DIM_CORR = [4, 5, 6, 7, 8, 9, 10]
EMBED_DELAY_CORR = 10
MAX_POINTS_CORR = 20000

POINCARE_ORDER = 15
CHAOS_THRESHOLD = 0.005

# Paramètres pour le spectre
USE_WELCH = True
NPERSEG = 2048
FFT_WINDOW = 'hann'

# ============================================================================
# 1. Exposant de Lyapunov maximal (Rosenstein)
# ============================================================================
def lyapunov_max(signal, dt, embed_dim=EMBED_DIM_LYAP, embed_delay=EMBED_DELAY_LYAP,
                 min_neighbors=MIN_NEIGHBORS, max_points=MAX_POINTS_LYAP):
    N = len(signal) - (embed_dim - 1) * embed_delay
    if N < 200:
        return 0.0
    if N > max_points:
        step = N // max_points
        signal = signal[::step]
        dt = dt * step
        N = len(signal) - (embed_dim - 1) * embed_delay
        print(f"   Lyapunov: sous‑échantillonnage facteur {step}, nouveau dt={dt:.3f} s")
    X = np.array([signal[i:i + embed_dim * embed_delay:embed_delay] for i in range(N)])
    X = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-10)
    tree = KDTree(X)
    distances, indices = tree.query(X, k=min_neighbors+1)
    distances = distances[:, 1:]
    indices = indices[:, 1:]
    max_t = min(60, N // 2)
    div = np.zeros(max_t)
    cnt = np.zeros(max_t)
    for i in range(N - max_t):
        for j_idx in indices[i]:
            if j_idx + max_t >= N:
                continue
            d0 = distances[i, np.where(indices[i] == j_idx)[0][0]]
            if d0 < 1e-10:
                continue
            for k in range(1, max_t):
                if i + k >= N or j_idx + k >= N:
                    break
                dk = np.linalg.norm(X[i+k] - X[j_idx+k])
                if dk > 1e-10:
                    div[k] += np.log(dk / d0)
                    cnt[k] += 1
    mask = cnt > 0
    if np.sum(mask) < 10:
        return 0.0
    div = np.divide(div, cnt, where=mask)
    t_range = np.arange(1, max_t)[mask[1:]]
    d_range = div[1:][mask[1:]]
    n_points = min(20, len(t_range))
    if n_points < 5:
        return 0.0
    coeffs = np.polyfit(t_range[:n_points], d_range[:n_points], 1)
    return coeffs[0] / dt

# ============================================================================
# 2. Dimension de corrélation (Grassberger-Procaccia)
# ============================================================================
def correlation_dimension(signal, dt, embed_dims=EMBED_DIM_CORR, delay=EMBED_DELAY_CORR,
                          max_points=MAX_POINTS_CORR):
    N = len(signal)
    if N < 500:
        return 0.0, False
    if N > max_points:
        step = N // max_points
        signal = signal[::step]
        dt = dt * step
        N = len(signal)
        print(f"   CorrDim: sous‑échantillonnage facteur {step}, nouveau dt={dt:.3f} s")
    pentes = []
    for m in embed_dims:
        L = N - (m - 1) * delay
        if L < 100:
            continue
        X = np.array([signal[i:i + m * delay:delay] for i in range(L)])
        dist = pdist(X)
        if len(dist) == 0:
            continue
        r_min = np.min(dist[dist > 0]) * 0.9
        r_max = np.max(dist) * 0.3
        if r_min >= r_max:
            continue
        r = np.logspace(np.log10(r_min), np.log10(r_max), 30)
        C = np.array([np.mean(dist < ri) for ri in r])
        mask = (C > 0.01) & (C < 0.99)
        if np.sum(mask) < 5:
            continue
        coeffs = np.polyfit(np.log10(r[mask]), np.log10(C[mask]), 1)
        pentes.append(coeffs[0])
    if not pentes:
        return 0.0, False
    d2 = pentes[-1]
    converged = len(pentes) >= 3 and abs(pentes[-1] - pentes[-2]) < 0.1
    return d2, converged

# ============================================================================
# 3. Return map et section de Poincaré
# ============================================================================
def poincare_return(A, t, dt, order=POINCARE_ORDER):
    maxima_idx = argrelextrema(A, np.greater, order=order)[0]
    maxima_idx = maxima_idx[maxima_idx > 10]
    maxima_idx = maxima_idx[maxima_idx < len(A)-10]
    A_max = A[maxima_idx]
    t_max = t[maxima_idx]
    return_map = np.vstack([A_max[:-1], A_max[1:]]).T
    return t_max, A_max, return_map

# ============================================================================
# 4. Tracé des figures
# ============================================================================
def plot_all(A, B, t, dt, beta, D, lyap, d2, regime, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    # Série temporelle (zoom 500 s)
    window = 500
    n_win = int(window / dt)
    if len(t) > n_win:
        t_plot = t[:n_win] - t[0]
        A_plot = A[:n_win]
    else:
        t_plot = t - t[0]
        A_plot = A
    plt.figure(figsize=(12,4))
    plt.plot(t_plot, A_plot, 'r-', linewidth=0.8)
    plt.xlabel('Time (s)'); plt.ylabel('[O₂] (µM)')
    plt.title(f'Time series - D={D}s, β={beta} [{regime}]')
    plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'timeseries_D{D}_beta{beta}.pdf'), dpi=150)
    plt.close()

    # Spectre de puissance
    sig = A - np.mean(A)
    if len(sig) > 100:
        if USE_WELCH:
            nperseg = min(NPERSEG, len(sig)//2)
            noverlap = nperseg // 2
            f, Pxx = welch(sig, fs=1/dt, nperseg=nperseg, noverlap=noverlap, window=FFT_WINDOW)
            amp = np.sqrt(Pxx)
        else:
            n = len(sig)
            window = np.hanning(n) if FFT_WINDOW == 'hann' else np.ones(n)
            sigw = sig * window
            f = np.fft.rfftfreq(n, dt)
            amp = np.abs(np.fft.rfft(sigw)) / n
        mask = f <= 0.2
        plt.figure(figsize=(10,6))
        plt.plot(f[mask], amp[mask], 'k-', linewidth=0.8)
        plt.xlabel('Frequency (Hz)'); plt.ylabel('Amplitude (µM)')
        plt.title(f'Power spectrum - D={D}s, β={beta} [{regime}]')
        plt.grid(True); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'spectrum_D{D}_beta{beta}.pdf'), dpi=150)
        plt.close()

    # Attracteur
    step = max(1, len(A)//3000)
    plt.figure(figsize=(8,8))
    plt.plot(B[::step], A[::step], 'b-', linewidth=0.5, alpha=0.7)
    plt.xlabel('[NADH] (µM)'); plt.ylabel('[O₂] (µM)')
    plt.title(f'Phase portrait - D={D}s, β={beta} [{regime}]')
    plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'attractor_D{D}_beta{beta}.pdf'), dpi=150)
    plt.close()

    # Section de Poincaré
    t_max, A_max, _ = poincare_return(A, t, dt)
    plt.figure(figsize=(12,6))
    plt.plot(t_max, A_max, 'r.', markersize=2, alpha=0.5)
    plt.xlabel('Time (s)'); plt.ylabel('[O₂] at maxima (µM)')
    plt.title(f'Poincaré section - D={D}s, β={beta} [{regime}]')
    plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'poincare_section_D{D}_beta{beta}.pdf'), dpi=150)
    plt.close()

    # Return map
    _, _, ret_map = poincare_return(A, t, dt)
    plt.figure(figsize=(8,8))
    plt.plot(ret_map[:,0], ret_map[:,1], 'b.', markersize=2, alpha=0.5)
    plt.xlabel('A_n (µM)'); plt.ylabel('A_{n+1} (µM)')
    plt.title(f'Return map - D={D}s, β={beta} [{regime}]')
    plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'return_map_D{D}_beta{beta}.pdf'), dpi=150)
    plt.close()

# ============================================================================
# MAIN
# ============================================================================
def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    npz_file = sys.argv[1]
    # Lecture dt et t_transient depuis le fichier ou par défaut
    data = np.load(npz_file)
    dt = float(data['dt']) if 'dt' in data else DEFAULT_DT
    t_transient = float(data['t_transient']) if 't_transient' in data else DEFAULT_T_TRANSIENT
    # Si des arguments supplémentaires sont fournis, ils priment
    if len(sys.argv) > 2:
        dt = float(sys.argv[2])
    if len(sys.argv) > 3:
        t_transient = float(sys.argv[3])

    # Extraction des séries
    if 'A' in data and 'B' in data and 't' in data:
        A = data['A']
        B = data['B']
        t = data['t']
        beta_val = float(data['beta']) if 'beta' in data else 0.0
        D_val = float(data['D']) if 'D' in data else 0.0
    elif 'sol' in data:
        sol = data['sol']
        A = sol[:, 0]
        B = sol[:, 1]
        t = np.linspace(0, len(A)*dt, len(A))
        beta_val = 0.0
        D_val = 0.0
    else:
        sol = data[list(data.keys())[0]]
        A = sol[:, 0]
        B = sol[:, 1]
        t = np.linspace(0, len(A)*dt, len(A))
        beta_val = 0.0
        D_val = 0.0

    # Extraction de beta et D depuis le nom si nécessaire
    if beta_val == 0.0 and D_val == 0.0:
        match = re.search(r'data_beta([0-9.]+)_D([0-9.]+)\.npz', npz_file)
        if match:
            beta_val = float(match.group(1))
            D_val = float(match.group(2))
            print(f"   (beta, D) extraits du nom du fichier : β={beta_val}, D={D_val}")

    # Élimination du transitoire
    idx = int(t_transient / dt)
    if idx >= len(A):
        idx = len(A) // 2
    A = A[idx:]
    B = B[idx:]
    t = t[idx:]

    # Calculs
    print("\n   Calcul de λ₁ (Rosenstein)...")
    lyap = lyapunov_max(A, dt)
    print("   Calcul de d₂ (Grassberger-Procaccia)...")
    d2, conv = correlation_dimension(A, dt)

    if lyap > CHAOS_THRESHOLD:
        regime = "CHAOS"
    elif lyap > 0.001:
        regime = "WEAK_CHAOS"
    else:
        regime = "PERIODIC"

    # Affichage
    print("\n" + "="*60)
    print(f"Fichier : {npz_file}")
    print(f"dt = {dt} s, transitoire = {t_transient} s")
    print(f"Points analysés : {len(A)}")
    print(f"β = {beta_val}, D = {D_val} s")
    print(f"λ₁ = {lyap:.6f}")
    print(f"d₂ = {d2:.3f} (convergence={conv})")
    print(f"Régime : {regime}")
    print("="*60)

    out_dir = f"analyse_post_{npz_file.replace('.npz','')}"
    plot_all(A, B, t, dt, beta_val, D_val, lyap, d2, regime, out_dir)
    print(f"✅ Figures sauvegardées dans : {out_dir}/")

if __name__ == "__main__":
    main()
