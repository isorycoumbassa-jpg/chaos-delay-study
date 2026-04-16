#!/usr/bin/env python3
"""
Modèle DOP (Degn-Olsen-Perram) - Réaction Peroxydase-Oxydase
AVEC FEEDBACK DE ROESKY (1993) - Version Cloud/Local

COMPATIBLE :
- Cloud (GCP, AWS, Azure) avec RAM ≥ 32 GB
- Local (Mac, Linux) avec RAM ≥ 8 GB
- Détection automatique des ressources
- Adaptation dynamique des paramètres

PROTOCOLE ROESKY :
- β = 0.0304 (coupling strength)
- k₁ = 0.08 (paramètre de bifurcation)
- dt = 1.0 s (haute résolution)

Variables DOP :
- y(0) = A = [O₂]   (oxygène)
- y(1) = B = [NADH] (nicotinamide adénine dinucléotide)
- y(2) = X (intermédiaire)
- y(3) = Y (intermédiaire)

AUTEUR : Ibrahima Sory Coumbassa
Université de Conakry, Guinée
"""

import numpy as np
import matplotlib.pyplot as plt
from jitcdde import jitcdde, y, t
import os
import sys
import json
import time
from datetime import datetime
from scipy.signal import find_peaks, welch
from scipy.spatial.distance import cdist, pdist
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# TENTATIVE D'IMPORT DE PSUTIL (OPTIONNEL POUR DÉTECTION)
# ============================================================================
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("⚠ psutil non installé. Installation recommandée: pip install psutil")
    print("  Utilisation des valeurs par défaut pour la détection.")

# ============================================================================
# DÉTECTION AUTOMATIQUE DE L'ENVIRONNEMENT
# ============================================================================
def detect_environment():
    """Détecte si on est sur cloud ou local et ajuste les paramètres."""
    
    if PSUTIL_AVAILABLE:
        try:
            mem = psutil.virtual_memory()
            mem_total_gb = mem.total / (1024**3)
        except:
            mem_total_gb = 8.0
    else:
        mem_total_gb = 8.0  # Valeur par défaut
    
    # Détection cloud vs local
    is_cloud = mem_total_gb >= 16
    
    # Ajustement des paramètres
    if is_cloud:
        # Configuration cloud (haute performance)
        config = {
            'DT': 1.0,
            'T_TOTAL': 100000,
            'T_TRANSIENT': 5000,
            'USE_SVD': True,
            'USE_RQA': True,
            'SAVE_FIGURES': True,
            'ENV_NAME': 'CLOUD'
        }
    else:
        # Configuration locale (économie mémoire)
        config = {
            'DT': 1.0,
            'T_TOTAL': 50000,
            'T_TRANSIENT': 5000,
            'USE_SVD': False,
            'USE_RQA': False,
            'SAVE_FIGURES': True,
            'ENV_NAME': 'LOCAL'
        }
    
    return config, mem_total_gb

# Détection
CONFIG, MEM_TOTAL = detect_environment()

# ============================================================================
# PARAMÈTRES DE SIMULATION (adaptés automatiquement)
# ============================================================================
dt = CONFIG['DT']
t_total = CONFIG['T_TOTAL']
t_transient = CONFIG['T_TRANSIENT']
USE_SVD = CONFIG['USE_SVD']
USE_RQA = CONFIG['USE_RQA']
SAVE_FIGURES = CONFIG['SAVE_FIGURES']
ENV_NAME = CONFIG['ENV_NAME']

# ============================================================================
# PARAMÈTRES FIXES DU MODÈLE DOP (Olsen 2024)
# ============================================================================
k2, k3, k4, k5, k6 = 800.0, 0.05, 20.0, 1.5, 0.001
a0, k7, k8 = 8.0, 0.1, 0.61

# Paramètres de l'étude
k1_fixed = 0.08
beta_fixed = 0.0304

# Conditions initiales FIXES
CI_FIXES = [6.0, 150.0, 0.1, 0.1]  # [A, B, X, Y]

# Paramètres pour le calcul de Lyapunov (ajustés pour dt)
LYAP_EMBED_DIM = 5
LYAP_EMBED_DELAY = int(15.0 / dt)   # 15 secondes en pas
LYAP_MIN_NEIGHBORS = 20

# Paramètres pour SVD et RQA
CORR_EMBED_DIMS = [4, 5, 6, 7, 8, 9, 10]
CORR_EMBED_DELAY = int(10.0 / dt)   # 10 secondes en pas
RQA_EMBED_DIM = 3
RQA_EMBED_DELAY = int(10.0 / dt)   # 10 secondes en pas
RQA_THRESHOLD = 0.15

# Seuils pour la classification
CHAOS_LYAP_THRESHOLD = 0.005
WEAK_CHAOS_D2_THRESHOLD = 2.0
QUASIPERIODIC_D2_THRESHOLD = 1.5

print("="*80)
print("MODÈLE DOP (Degn-Olsen-Perram) - VERSION CLOUD/LOCAL")
print(f"Environnement : {ENV_NAME} (RAM totale: {MEM_TOTAL:.1f} GB)")
print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)
print(f"Paramètres : k₁={k1_fixed}, β={beta_fixed}")
print(f"CI : [O₂]₀={CI_FIXES[0]}, [NADH]₀={CI_FIXES[1]}")
print(f"dt={dt} s, t_total={t_total} s, transitoire={t_transient} s")
print(f"USE_SVD={USE_SVD}, USE_RQA={USE_RQA}, SAVE_FIGURES={SAVE_FIGURES}")
print("="*80)


# ============================================================================
# 1. EXPOSANT DE LYAPUNOV (Rosenstein)
# ============================================================================
def calculer_lyapunov(signal, dt, embed_dim=LYAP_EMBED_DIM,
                      embed_delay=LYAP_EMBED_DELAY,
                      min_neighbors=LYAP_MIN_NEIGHBORS):
    """Calcule l'exposant de Lyapunov maximal (Rosenstein)."""
    N = len(signal) - (embed_dim - 1) * embed_delay
    if N < 200:
        return 0.0

    # Sous-échantillonnage pour mémoire (garder max 30000 points)
    if N > 30000:
        step = N // 30000
        signal_sub = signal[::step]
        N = len(signal_sub) - (embed_dim - 1) * embed_delay
        dt_sub = dt * step
    else:
        signal_sub = signal
        dt_sub = dt

    X = np.array([signal_sub[i:i + embed_dim * embed_delay:embed_delay] for i in range(N)])
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
    return coeffs[0] / dt_sub


# ============================================================================
# 2. DIMENSION DE CORRÉLATION (Grassberger-Procaccia) - Optionnel
# ============================================================================
def calculer_dimension_correlation(signal, dt, embed_dims=CORR_EMBED_DIMS,
                                   delay=CORR_EMBED_DELAY, n_r=30):
    """Calcule la dimension de corrélation (Grassberger-Procaccia)."""
    N = len(signal)
    if N < 500:
        return 0.0, [], False

    # Sous-échantillonnage pour mémoire
    if N > 15000:
        step = N // 15000
        signal_sub = signal[::step]
        dt_sub = dt * step
    else:
        signal_sub = signal
        dt_sub = dt

    pentes = []
    for m in embed_dims:
        L = len(signal_sub) - (m - 1) * delay
        if L < 100:
            continue
        X = np.array([signal_sub[i:i + m * delay:delay] for i in range(L)])
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


# ============================================================================
# 3. ANALYSE DE RÉCURRENCE (RQA) - Optionnel
# ============================================================================
def recurrence_quantification(signal, embed_dim=RQA_EMBED_DIM,
                              embed_delay=RQA_EMBED_DELAY,
                              threshold=RQA_THRESHOLD):
    """Recurrence Quantification Analysis (RR, DET, Lmax)."""
    N = len(signal) - (embed_dim - 1) * embed_delay
    if N < 100:
        return 0.0, 0.0, 0.0

    # Limiter la taille pour mémoire
    if N > 800:
        step = N // 800
        signal_sub = signal[::step]
        N = len(signal_sub) - (embed_dim - 1) * embed_delay
    else:
        signal_sub = signal

    X = np.array([signal_sub[i:i + embed_dim * embed_delay:embed_delay] for i in range(N)])
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


# ============================================================================
# 4. CYCLE LIBRE POUR UN D DONNÉ (calcul de ⟨A⟩)
# ============================================================================
def cycle_libre_pour_D(D, beta):
    """Simule le cycle libre (sans feedback) à partir des CI fixes."""
    print(f"      [Cycle libre] Simulation sans feedback...", flush=True)

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

    peaks, _ = find_peaks(A_perm[-5000:], distance=50)
    if len(peaks) > 3:
        t_peaks = t_array[idx:][-5000:][peaks]
        periode_libre = np.mean(np.diff(t_peaks))
    else:
        periode_libre = 0

    print(f"      Période libre = {periode_libre:.1f} s, ⟨A⟩ = {A_av:.4f} µM", flush=True)

    return A_av, periode_libre


# ============================================================================
# 5. SIMULATION AVEC FEEDBACK (CI FIXES)
# ============================================================================
def simulate_with_feedback(D, beta, A_av):
    """Simulation avec feedback de Roesky (1993)."""
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
        except Exception:
            break

    return np.array(t_array[:len(sol)]), np.array(sol)


# ============================================================================
# 6. ANALYSE DES RÉSULTATS
# ============================================================================
def analyser_resultats(t_array, sol, D, beta, A_av, figures_dir):
    """Analyse complète avec gestion mémoire."""
    A = sol[:, 0]
    B = sol[:, 1]

    idx = int(t_transient / dt)
    if idx >= len(A):
        idx = len(A) // 2

    t_perm = t_array[idx:]
    A_perm = A[idx:]
    B_perm = B[idx:]

    # Période
    peaks, _ = find_peaks(A_perm, distance=50)
    if len(peaks) > 5:
        t_peaks = t_perm[peaks]
        periodes = np.diff(t_peaks)
        periode = np.mean(periodes)
        cv = np.std(periodes) / periode if periode > 0 else 1.0
    else:
        periode, cv = 0, 1.0

    # Lyapunov
    lyap = calculer_lyapunov(A_perm, dt)

    # SVD (optionnel)
    if USE_SVD:
        d2, pentes, convergence = calculer_dimension_correlation(A_perm, dt)
    else:
        d2 = 0.0

    # RQA (optionnel)
    if USE_RQA:
        RR, DET, Lmax = recurrence_quantification(A_perm)
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

    # Sauvegarde des figures
    if SAVE_FIGURES:
        window_s = 500
        n_window = int(window_s / dt)
        if len(t_perm) > n_window:
            t_plot = t_perm[:n_window] - t_perm[0]
            A_plot = A_perm[:n_window]
        else:
            t_plot = t_perm - t_perm[0]
            A_plot = A_perm

        plt.figure(figsize=(12, 4))
        plt.plot(t_plot, A_plot, 'r-', linewidth=0.8)
        plt.axhline(y=A_av, color='k', linestyle='--', alpha=0.5, label=f'⟨A⟩={A_av:.1f}µM')
        plt.xlabel('Temps (s)')
        plt.ylabel('[O₂] (µM)')
        plt.title(f'Série temporelle - D={D}s, β={beta} [{regime}]')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(figures_dir, f'timeseries_D{D}_beta{beta}.pdf'),
                    dpi=150, bbox_inches='tight')
        plt.close()

        signal_center = A_perm - np.mean(A_perm)
        if len(signal_center) > 100:
            nperseg = min(2048, len(signal_center) // 4)
            f, Pxx = welch(signal_center, fs=1/dt, nperseg=nperseg)
            amplitude = np.sqrt(Pxx)

            plt.figure(figsize=(10, 6))
            mask = f <= 0.2
            plt.plot(f[mask], amplitude[mask], 'k-', linewidth=0.8)
            plt.xlabel('Fréquence (Hz)')
            plt.ylabel('Amplitude (µM)')
            plt.title(f'Spectre d\'amplitude - D={D}s, β={beta} [{regime}]')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(figures_dir, f'spectrum_D{D}_beta{beta}.pdf'),
                        dpi=150, bbox_inches='tight')
            plt.close()

        plt.figure(figsize=(8, 8))
        step = max(1, len(A_perm) // 3000)
        plt.plot(B_perm[::step], A_perm[::step], 'b-', linewidth=0.5, alpha=0.7)
        plt.xlabel('[NADH] (µM)')
        plt.ylabel('[O₂] (µM)')
        plt.title(f'Portrait de phase - D={D}s, β={beta} [{regime}]')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(figures_dir, f'attractor_D{D}_beta{beta}.pdf'),
                    dpi=150, bbox_inches='tight')
        plt.close()

    return periode, cv, lyap, d2, RR, DET, regime


# ============================================================================
# 7. FONCTION PRINCIPALE
# ============================================================================
def main():
    """Interface interactive - compatible cloud et local."""
    
    print("\n" + "="*80)
    print("INTERFACE INTERACTIVE - MODÈLE DOP")
    print(f"Environnement: {ENV_NAME} | RAM: {MEM_TOTAL:.1f} GB")
    print(f"k₁ = {k1_fixed}, β = {beta_fixed}")
    print(f"dt = {dt} s, t_total = {t_total} s")
    print(f"USE_SVD = {USE_SVD}, USE_RQA = {USE_RQA}")
    print("="*80)

    all_results = []
    current_beta = beta_fixed  # Valeur par défaut
    figures_dir = f"figures_DOP_beta{current_beta}"
    os.makedirs(figures_dir, exist_ok=True)

    while True:
        print("\n" + "-"*50)
        print("MENU PRINCIPAL")
        print("-"*50)
        print("1. Changer/entrer la valeur de β (feedback gain)")
        print("2. Simuler une valeur de D")
        print("3. Afficher le résumé des simulations")
        print("4. Sauvegarder et quitter")
        print("5. Quitter sans sauvegarder")

        print(f"\n   État actuel : β = {current_beta}")
        print(f"   Dossier : {figures_dir}/")

        choix = input("\nVotre choix (1-5) : ")

        if choix == '1':
            beta_input = input(f"Entrez la nouvelle valeur de β (ex: 0.0304, 0.0306, ...) : ")
            try:
                current_beta = float(beta_input)
                figures_dir = f"figures_DOP_beta{current_beta}"
                os.makedirs(figures_dir, exist_ok=True)
                print(f"\n✅ β = {current_beta}")
                print(f"   Dossier : {figures_dir}/")
            except ValueError:
                print("❌ Valeur invalide.")

        elif choix == '2':
            D_input = input("Entrez la valeur de D (s) : ")
            try:
                D = float(D_input)
            except ValueError:
                print("❌ Valeur invalide.")
                continue

            start_time = time.time()
            print(f"\n  --- Simulation : β={current_beta}, D={D} s ---")
            print(f"    CI : [O₂]₀={CI_FIXES[0]}, [NADH]₀={CI_FIXES[1]}")
            print(f"    Début : {datetime.now().strftime('%H:%M:%S')}", flush=True)

            try:
                A_av, periode_libre = cycle_libre_pour_D(D, current_beta)
                print(f"      [Feedback] Simulation...", flush=True)
                t_arr, sol = simulate_with_feedback(D, current_beta, A_av)
                periode, cv, lyap, d2, RR, DET, regime = analyser_resultats(
                    t_arr, sol, D, current_beta, A_av, figures_dir
                )

                elapsed = time.time() - start_time
                all_results.append({
                    'beta': current_beta,
                    'D': D,
                    'periode_libre': periode_libre,
                    'periode': periode,
                    'cv': cv,
                    'lyap': lyap,
                    'd2': d2,
                    'RR': RR,
                    'DET': DET,
                    'regime': regime,
                    'elapsed': elapsed
                })

                print(f"\n      --- RÉSULTATS ---")
                print(f"      Période libre = {periode_libre:.1f} s")
                print(f"      Période feedback = {periode:.1f} s")
                print(f"      λ = {lyap:.6f}, d₂ = {d2:.3f}")
                if USE_RQA:
                    print(f"      RR = {RR:.5f}, DET = {DET:.4f}")
                print(f"      → {regime}")
                print(f"      Temps écoulé : {elapsed/60:.1f} min", flush=True)

            except Exception as e:
                print(f"      ❌ ERREUR : {e}")
                import traceback
                traceback.print_exc()

        elif choix == '3':
            if not all_results:
                print("\nAucune simulation effectuée.")
            else:
                print("\n" + "="*80)
                print("RÉSUMÉ DES SIMULATIONS")
                print("="*80)
                print(f"{'β':<10} {'D (s)':<10} {'Période libre':<14} {'Période FB':<12} {'λ Lyap':<12} {'d₂':<10} {'Régime'}")
                print("-"*85)
                for r in all_results:
                    print(f"{r['beta']:<10.4f} {r['D']:<10.1f} {r['periode_libre']:<14.1f} {r['periode']:<12.1f} {r['lyap']:<12.6f} {r['d2']:<10.3f} {r['regime']}")

        elif choix == '4':
            if all_results:
                save_data = {
                    'env': ENV_NAME,
                    'ram_gb': MEM_TOTAL,
                    'dt': dt,
                    't_total': t_total,
                    'beta': [r['beta'] for r in all_results],
                    'D': [r['D'] for r in all_results],
                    'periode_libre': [r['periode_libre'] for r in all_results],
                    'periode': [r['periode'] for r in all_results],
                    'cv': [r['cv'] for r in all_results],
                    'lyap': [r['lyap'] for r in all_results],
                    'd2': [r['d2'] for r in all_results],
                    'RR': [r['RR'] for r in all_results],
                    'DET': [r['DET'] for r in all_results],
                    'regime': [r['regime'] for r in all_results],
                    'elapsed': [r['elapsed'] for r in all_results]
                }
                np.savez('resultats_DOP_complets.npz', **save_data)
                
                with open('resultats_DOP_complets.json', 'w') as f:
                    json.dump(save_data, f, indent=2)
                    
                print(f"\n✅ Résultats sauvegardés dans resultats_DOP_complets.npz et .json")
            print("\nAu revoir.")
            break

        elif choix == '5':
            print("\nAu revoir.")
            break

        else:
            print("\n❌ Choix invalide.")


if __name__ == "__main__":
    main()
