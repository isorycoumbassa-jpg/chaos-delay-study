#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modèle DOP – Version interactive
Saisie de β et D, exécution d'une seule simulation.
Utilisable sur Mac et VM (adapter t_total si nécessaire).
"""

import numpy as np
import matplotlib.pyplot as plt
from jitcdde import jitcdde, y, t
import os
import json
from scipy.signal import find_peaks, welch
from scipy.spatial.distance import cdist, pdist
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PARAMÈTRES FIXES (modifiables selon vos besoins)
# ============================================================================
k2, k3, k4, k5, k6 = 800.0, 0.05, 20.0, 1.5, 0.001
a0, k7, k8 = 8.0, 0.1, 0.61
k1_fixed = 0.08
CI_FIXES = [6.0, 150.0, 0.1, 0.1]

# Paramètres de simulation (à ajuster)
dt = 1.0
t_total = 50000          # Pour des calculs rapides – passer à 100000 sur VM
t_transient = 5000

# Options
SAVE_FIGURES = True
USE_SVD = False          # Activer sur VM si besoin
USE_RQA = False

# Paramètres SVD/RQA (si activés)
CORR_EMBED_DIMS = [4, 5, 6, 7, 8, 9, 10]
CORR_EMBED_DELAY = 10
RQA_EMBED_DIM = 3
RQA_EMBED_DELAY = 10
RQA_THRESHOLD = 0.15

# Seuils
CHAOS_LYAP_THRESHOLD = 0.005
WEAK_CHAOS_D2_THRESHOLD = 2.0
QUASIPERIODIC_D2_THRESHOLD = 1.5

# ============================================================================
# FONCTIONS DE CALCUL
# ============================================================================
def calculer_lyapunov(signal, dt, embed_dim=5, embed_delay=15, min_neighbors=20):
    N = len(signal) - (embed_dim - 1) * embed_delay
    if N < 200:
        return 0.0
    # Sous‑échantillonnage pour mémoire
    if N > 30000:
        step = N // 30000
        signal = signal[::step]
        dt = dt * step
        N = len(signal) - (embed_dim - 1) * embed_delay
    X = np.array([signal[i:i + embed_dim * embed_delay:embed_delay] for i in range(N)])
    X = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-10)
    from scipy.spatial import KDTree
    tree = KDTree(X)
    distances, indices = tree.query(X, k=min_neighbors+1)
    distances = distances[:, 1:]
    indices = indices[:, 1:]
    max_t = min(60, N // 2)
    divergence = np.zeros(max_t)
    counts = np.zeros(max_t)
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
                    divergence[k] += np.log(dk / d0)
                    counts[k] += 1
    mask = counts > 0
    if np.sum(mask) < 10:
        return 0.0
    divergence = np.divide(divergence, counts, where=mask)
    t_range = np.arange(1, max_t)[mask[1:]]
    d_range = divergence[1:][mask[1:]]
    n_points = min(20, len(t_range))
    if n_points < 5:
        return 0.0
    coeffs = np.polyfit(t_range[:n_points], d_range[:n_points], 1)
    return coeffs[0] / dt

def calculer_dimension_correlation(signal, dt, embed_dims=CORR_EMBED_DIMS,
                                   delay=CORR_EMBED_DELAY, n_r=30):
    N = len(signal)
    if N < 500:
        return 0.0, [], False
    if N > 15000:
        step = N // 15000
        signal = signal[::step]
        dt = dt * step
    pentes = []
    for m in embed_dims:
        L = len(signal) - (m - 1) * delay
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
        r = np.logspace(np.log10(r_min), np.log10(r_max), n_r)
        C = np.array([np.mean(dist < ri) for ri in r])
        mask = (C > 0.01) & (C < 0.99)
        if np.sum(mask) < 5:
            continue
        coeffs = np.polyfit(np.log10(r[mask]), np.log10(C[mask]), 1)
        pentes.append(coeffs[0])
    if not pentes:
        return 0.0, [], False
    d2 = pentes[-1]
    convergence = len(pentes) >= 3 and abs(pentes[-1] - pentes[-2]) < 0.1
    return d2, pentes, convergence

def recurrence_quantification(signal, embed_dim=RQA_EMBED_DIM,
                              embed_delay=RQA_EMBED_DELAY,
                              threshold=RQA_THRESHOLD):
    N = len(signal) - (embed_dim - 1) * embed_delay
    if N < 100:
        return 0.0, 0.0, 0.0
    if N > 800:
        step = N // 800
        signal = signal[::step]
        N = len(signal) - (embed_dim - 1) * embed_delay
    X = np.array([signal[i:i + embed_dim * embed_delay:embed_delay] for i in range(N)])
    dist = cdist(X, X)
    eps = threshold * np.std(dist)
    R = (dist < eps).astype(int)
    RR = np.sum(R) / (N * N)
    diag_lengths = []
    for diag in range(-N+1, N):
        diagonal = np.diag(R, k=diag)
        diff_diag = np.diff(np.concatenate(([0], diagonal, [0])))
        starts = np.where(diff_diag == 1)[0]
        ends = np.where(diff_diag == -1)[0]
        lengths = ends - starts
        diag_lengths.extend(lengths[lengths >= 2])
    DET = np.sum(diag_lengths) / np.sum(R) if np.sum(R) > 0 else 0.0
    Lmax = max(diag_lengths) if diag_lengths else 0.0
    return RR, DET, Lmax

def cycle_libre(beta):
    """Simule le cycle libre (sans feedback) à partir de CI fixes."""
    print("  [Cycle libre] ...")
    equations = [
        -k1_fixed*y(0)*y(1)*y(2) - k3*y(0)*y(1)*y(3) + k7*(a0 - y(0)),
        -k1_fixed*y(0)*y(1)*y(2) - k3*y(0)*y(1)*y(3) + k8,
        +k1_fixed*y(0)*y(1)*y(2) - 2*k2*y(2)**2 + 2*k3*y(0)*y(1)*y(3) - k4*y(2) + k6,
        +2*k2*y(2)**2 - k3*y(0)*y(1)*y(3) - k5*y(3)
    ]
    DDE = jitcdde(equations)
    DDE.constant_past(CI_FIXES)
    DDE.generate_lambdas()
    DDE.step_on_discontinuities()
    n_points = int(t_total / dt)
    t_array = np.linspace(0, t_total, n_points)
    sol = [DDE.integrate(time) for time in t_array]
    sol = np.array(sol)
    A = sol[:, 0]
    idx = int(t_transient / dt)
    A_perm = A[idx:]
    A_av = np.mean(A_perm)
    return A_av

def simulate_with_feedback(D, beta, A_av):
    def feedback():
        alpha = (y(0, t-D) - A_av) / A_av
        return beta * alpha * y(0)
    equations = [
        -k1_fixed*y(0)*y(1)*y(2) - k3*y(0)*y(1)*y(3) + k7*(a0 - y(0)) + feedback(),
        -k1_fixed*y(0)*y(1)*y(2) - k3*y(0)*y(1)*y(3) + k8,
        +k1_fixed*y(0)*y(1)*y(2) - 2*k2*y(2)**2 + 2*k3*y(0)*y(1)*y(3) - k4*y(2) + k6,
        +2*k2*y(2)**2 - k3*y(0)*y(1)*y(3) - k5*y(3)
    ]
    DDE = jitcdde(equations)
    DDE.constant_past(CI_FIXES)
    DDE.generate_lambdas()
    DDE.step_on_discontinuities()
    n_points = int(t_total / dt)
    t_array = np.linspace(0, t_total, n_points)
    sol = []
    for time in t_array:
        try:
            sol.append(DDE.integrate(time))
        except Exception as e:
            print(f"    Erreur d'intégration à t={time}: {e}")
            break
    if len(sol) < 10:
        raise RuntimeError("Intégration trop courte")
    return np.array(t_array[:len(sol)]), np.array(sol)

def analyser_resultats(t_array, sol, D, beta, A_av, out_dir):
    A = sol[:, 0]
    B = sol[:, 1]
    idx = int(t_transient / dt)
    if idx >= len(A):
        idx = len(A) // 2
    t_perm = t_array[idx:]
    A_perm = A[idx:]
    B_perm = B[idx:]

    # Période
    #peaks, _ = find_peaks(A_perm, distance=50)
    #peaks, _ = find_peaks(A_perm, distance=distance_pas, height=np.percentile(A_perm, 50))
    # distance en pas : on veut environ la moitié de la période attendue (~6 s)
    distance_pas = int(6.0 / dt)   # avec dt=0.02 → 300 pas
    peaks, _ = find_peaks(A_perm, distance=distance_pas, height=np.percentile(A_perm, 50))
    if len(peaks) > 5:
        t_peaks = t_perm[peaks]
        periodes = np.diff(t_peaks)
        periode = np.mean(periodes)
    else:
        periode = 0

    # Lyapunov
    lyap = calculer_lyapunov(A_perm, dt)

    # SVD (optionnel)
    if USE_SVD:
        d2, _, _ = calculer_dimension_correlation(A_perm, dt)
    else:
        d2 = 0.0

    # RQA (optionnel)
    if USE_RQA:
        RR, DET, _ = recurrence_quantification(A_perm)
    else:
        RR, DET = 0.0, 0.0

    # Classification
    if lyap > CHAOS_LYAP_THRESHOLD:
        regime = "CHAOS"
    elif d2 > WEAK_CHAOS_D2_THRESHOLD and DET < 0.6:
        regime = "WEAK_CHAOS"
    elif d2 > QUASIPERIODIC_D2_THRESHOLD:
        regime = "QUASIPERIODIC"
    else:
        regime = "PERIODIC"

    # Figures
    if SAVE_FIGURES and out_dir:
        os.makedirs(out_dir, exist_ok=True)
        # Timeserie
        window_s = 500
        n_window = int(window_s / dt)
        if len(t_perm) > n_window:
            t_plot = t_perm[:n_window] - t_perm[0]
            A_plot = A_perm[:n_window]
        else:
            t_plot = t_perm - t_perm[0]
            A_plot = A_perm
        plt.figure(figsize=(12,4))
        plt.plot(t_plot, A_plot, 'r-', linewidth=0.8)
        plt.xlabel('Time (s)')
        plt.ylabel('[O₂] (µM)')
        plt.title(f'Timeseries - D={D}s, β={beta} [{regime}]')
        plt.grid(True)
        plt.savefig(os.path.join(out_dir, f'timeseries_D{D}_beta{beta}.pdf'), dpi=150, bbox_inches='tight')
        plt.close()
        # Spectre
        signal_center = A_perm - np.mean(A_perm)
        if len(signal_center) > 100:
            nperseg = min(1024, len(signal_center)//4)
            f, Pxx = welch(signal_center, fs=1/dt, nperseg=nperseg)
            amplitude = np.sqrt(Pxx)
            plt.figure(figsize=(10,6))
            mask = f <= 0.2
            plt.plot(f[mask], amplitude[mask], 'k-')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Amplitude (µM)')
            plt.title(f'Spectrum - D={D}s, β={beta} [{regime}]')
            plt.grid(True)
            plt.savefig(os.path.join(out_dir, f'spectrum_D{D}_beta{beta}.pdf'), dpi=150, bbox_inches='tight')
            plt.close()
        # Attracteur
        plt.figure(figsize=(8,8))
        step = max(1, len(A_perm)//3000)
        plt.plot(B_perm[::step], A_perm[::step], 'b-', linewidth=0.5, alpha=0.7)
        plt.xlabel('[NADH] (µM)')
        plt.ylabel('[O₂] (µM)')
        plt.title(f'Attractor - D={D}s, β={beta} [{regime}]')
        plt.grid(True)
        plt.savefig(os.path.join(out_dir, f'attractor_D{D}_beta{beta}.pdf'), dpi=150, bbox_inches='tight')
        plt.close()
        # Recurrence plot (si RQA activé)
        if USE_RQA and regime in ["CHAOS", "WEAK_CHAOS"]:
            N_rqa = min(500, len(A_perm))
            if N_rqa > (RQA_EMBED_DIM - 1) * RQA_EMBED_DELAY:
                X_rqa = np.array([A_perm[i:i + RQA_EMBED_DIM * RQA_EMBED_DELAY:RQA_EMBED_DELAY]
                                  for i in range(N_rqa - (RQA_EMBED_DIM - 1) * RQA_EMBED_DELAY)])
                dist_rqa = cdist(X_rqa, X_rqa)
                eps = RQA_THRESHOLD * np.std(dist_rqa)
                R = (dist_rqa < eps).astype(int)
                plt.figure(figsize=(8,8))
                plt.imshow(R, cmap='binary', interpolation='nearest')
                plt.xlabel('Time index')
                plt.ylabel('Time index')
                plt.title(f'Recurrence plot - D={D}s, β={beta} (RR={RR:.4f}, DET={DET:.3f})')
                plt.colorbar(label='Recurrence')
                plt.savefig(os.path.join(out_dir, f'recurrence_D{D}_beta{beta}.pdf'), dpi=150, bbox_inches='tight')
                plt.close()
        # SVD spectrum (si SVD activé)
        if USE_SVD and regime in ["CHAOS", "WEAK_CHAOS"]:
            N_svd = len(A_perm)
            embed_dim_svd = 25
            L = N_svd - embed_dim_svd + 1
            H = np.zeros((embed_dim_svd, L))
            for i in range(embed_dim_svd):
                H[i, :] = A_perm[i:i+L]
            U, S, Vt = np.linalg.svd(H, full_matrices=False)
            sing_vals = S / S[0]
            eff_dim = np.sum(sing_vals > 0.01)
            plt.figure(figsize=(8,5))
            plt.semilogy(range(1, len(sing_vals)+1), sing_vals, 'bo-', linewidth=1.5, markersize=4)
            plt.axhline(y=0.01, color='r', linestyle='--', alpha=0.5, label='Noise level')
            plt.xlabel('Singular value index')
            plt.ylabel('Normalized singular values')
            plt.title(f'SVD spectrum - D={D}s, β={beta} (dim={eff_dim})')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.savefig(os.path.join(out_dir, f'svd_spectrum_D{D}_beta{beta}.pdf'), dpi=150, bbox_inches='tight')
            plt.close()
    # Sauvegarde des séries temporelles pour post-traitement
    np.savez(f'data_beta{beta}_D{D}.npz',
             A=A_perm,          # concentration O₂ (µM)
             B=B_perm,          # concentration NADH (µM)
             t=t_perm,          # temps (s)
	     beta=beta,         
             D=D,
             periode=periode,
             lyap=lyap,
             d2=d2,
             regime=regime,
             dt=dt)
    print(f"    Données sauvegardées dans data_beta{beta}_D{D}.npz")
    return periode, lyap, d2, RR, DET, regime

# ============================================================================
# MAIN INTERACTIF
# ============================================================================
def main():
    print("\n" + "="*80)
    print("MODÈLE DOP - SIMULATION INTERACTIVE")
    print("="*80)
    print("Paramètres courants :")
    print(f"  dt = {dt} s, t_total = {t_total} s, transitoire = {t_transient} s")
    print(f"  SVD = {USE_SVD}, RQA = {USE_RQA}, sauvegarde figures = {SAVE_FIGURES}")
    print("-"*50)

    # Saisie de β
    while True:
        try:
            beta = float(input("Entrez β (ex: 0.0304, 0.04, 0.08, etc.) : "))
            break
        except ValueError:
            print("❌ Valeur invalide. Entrez un nombre.")

    # Saisie de D
    while True:
        try:
            D = float(input("Entrez D (s) (ex: 5, 10, 20, 50, 100) : "))
            break
        except ValueError:
            print("❌ Valeur invalide. Entrez un nombre.")

    # Dossier de sortie
    out_dir = f"figures_DOP_beta{beta}_interactif"
    print(f"\n  --- Simulation : β={beta}, D={D} s ---")
    print(f"    Dossier : {out_dir}")

    try:
        # 1. Cycle libre
        A_av = cycle_libre(beta)

        # 2. Simulation avec feedback
        t_arr, sol = simulate_with_feedback(D, beta, A_av)

        # 3. Analyse
        periode, lyap, d2, RR, DET, regime = analyser_resultats(t_arr, sol, D, beta, A_av, out_dir)

        # Sauvegarde des résultats (un seul point)
        result = [{
            'beta': beta,
            'D': D,
            'periode': periode,
            'lyap': lyap,
            'd2': d2,
            'RR': RR,
            'DET': DET,
            'regime': regime
        }]
        with open(f'resultats_beta{beta}_D{D}.json', 'w') as f:
            json.dump(result, f, indent=2)

        print("\n  ✅ RÉSULTATS")
        print(f"    Période = {periode:.1f} s")
        print(f"    λ = {lyap:.6f}")
        print(f"    d₂ = {d2:.3f}")
        if USE_RQA:
            print(f"    RR = {RR:.5f}, DET = {DET:.4f}")
        print(f"    → {regime}")

    except Exception as e:
        print(f"\n  ❌ ERREUR : {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
