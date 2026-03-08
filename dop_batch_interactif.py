#!/usr/bin/env python3
"""
Modèle DOP (Degn-Olsen-Perram) en réacteur fermé (batch)
Olsen (2024).
SANS flux - k1 est le paramètre de bifurcation
Sans feedback - pour exploration par les doctorants
Version interactive avec sauvegarde automatique des figures

Auteur: Ibrahima Sory Coumbassa
Pour les doctorants de l'Université de Conakry
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.signal import find_peaks, periodogram
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# PARAMÈTRES DU MODÈLE DOP BATCH (MISE À JOUR)
# ============================================================

# k1 sera demandé interactivement (paramètre de bifurcation)
# Plage : 0.01 - 0.06

# Paramètres fixes (d'après le nouveau tableau)
k2 = 800.0                    # k2 (400 - 1250) - valeur moyenne
k3 = 0.05                     # k3 (0.035 - 0.065) - valeur moyenne
k4 = 20.0                     # k4 (fixe)
k5 = 1.5                      # k5 (fixe)
k6 = 0.001                    # k6 (10^-3)
a0 = 8.0                      # a0 (fixe)
k7 = 0.1                      # k7 (fixe)
k8 = 0.61                     # k8 (0.5-0.72) - valeur moyenne

# Plage typique de k1 (paramètre de bifurcation)
k1_min, k1_max = 0.01, 0.06

print("=" * 80)
print("MODÈLE DOP BATCH (Degn-Olsen-Perram) - Sans flux")
print("Équations fournies par Doumbouya et al. (2025)")
print("=" * 80)
print("Paramètres FIXES (valeurs moyennes) :")
print(f"  k2 = {k2}      (plage 400-1250)")
print(f"  k3 = {k3}      (plage 0.035-0.065)")
print(f"  k4 = {k4}")
print(f"  k5 = {k5}")
print(f"  k6 = {k6}")
print(f"  a0 = {a0}")
print(f"  k7 = {k7}")
print(f"  k8 = {k8}      (plage 0.5-0.72)")
print("\nPARAMÈTRE DE BIFURCATION : k1")
print(f"  Plage typique : {k1_min} - {k1_max}")
print("=" * 80)

# Créer un dossier pour les figures s'il n'existe pas
figures_dir = "figures_dop_batch"
if not os.path.exists(figures_dir):
    os.makedirs(figures_dir)
    print(f"Dossier '{figures_dir}' créé pour sauvegarder les figures.")
else:
    print(f"Les figures seront sauvegardées dans le dossier '{figures_dir}'.")
print("=" * 80)


# ============================================================
# CLASSE DU MODÈLE DOP BATCH
# ============================================================

class ModeleDOPBatch:
    """
    Modèle DOP à 4 variables - SANS FLUX (batch)
    
    Variables d'état :
        a : [A] (O₂)
        b : [B] (NADH)
        x : [X] (intermédiaire)
        y : [Y] (intermédiaire)
    
    Équations :
        da/dt = -k₁ a b x - k₃ a b y + k₇ (a₀ - a)
        db/dt = -k₁ a b x - k₃ a b y + k₈
        dx/dt =  k₁ a b x - 2k₂ x² + 2k₃ a b y - k₄ x + k₆
        dy/dt =  2k₂ x² - k₃ a b y - k₅ y
    
    k1 est le paramètre de bifurcation
    """
    
    def __init__(self, k1):
        """
        Initialise le modèle avec k1 comme paramètre de bifurcation
        """
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.k4 = k4
        self.k5 = k5
        self.k6 = k6
        self.a0 = a0
        self.k7 = k7
        self.k8 = k8
        
    def modele(self, y, t):
        """
        Équations différentielles du modèle DOP batch
        y = [a, b, x, y]
        """
        a, b, x, y_var = y  # y_var pour éviter conflit avec variable y
        
        # Protection contre les valeurs négatives
        a = max(a, 1e-10)
        b = max(b, 1e-10)
        x = max(x, 1e-10)
        y_var = max(y_var, 1e-10)
        
        # Équations DOP batch (fournies)
        da = -self.k1 * a * b * x - self.k3 * a * b * y_var + self.k7 * (self.a0 - a)
        db = -self.k1 * a * b * x - self.k3 * a * b * y_var + self.k8
        dx =  self.k1 * a * b * x - 2 * self.k2 * x**2 + 2 * self.k3 * a * b * y_var - self.k4 * x + self.k6
        dy =  2 * self.k2 * x**2 - self.k3 * a * b * y_var - self.k5 * y_var
        
        return [da, db, dx, dy]


# ============================================================
# FONCTIONS D'ANALYSE
# ============================================================

def analyser_oscillations(t, y):
    """
    Analyse simple des oscillations pour le modèle DOP batch
    """
    a = y[:, 0]  # O₂
    
    # Éliminer le transitoire (première moitié)
    miroir = len(t) // 2
    a_stead = a[miroir:]
    
    # Statistiques
    a_min = np.min(a_stead)
    a_max = np.max(a_stead)
    a_mean = np.mean(a_stead)
    amplitude = a_max - a_min
    
    # Détection des pics
    peaks, _ = find_peaks(a_stead, distance=50, height=a_mean)
    nb_pics = len(peaks)
    
    # Période si assez de pics
    periode = 0
    if nb_pics >= 3:
        t_stead = t[miroir:]
        t_pics = t_stead[peaks]
        periodes = np.diff(t_pics)
        periode = np.mean(periodes)
    
    # Type de comportement
    if amplitude < 0.1:
        type_comport = "État stationnaire"
    elif nb_pics < 5:
        type_comport = "Transitoire long"
    elif periode > 0:
        type_comport = f"Oscillations (P{min(5, nb_pics//5+1)})"
    else:
        type_comport = "Comportement complexe"
    
    return {
        'amplitude': amplitude,
        'periode': periode,
        'nb_pics': nb_pics,
        'type': type_comport,
        'a_min': a_min,
        'a_max': a_max,
        'a_mean': a_mean
    }


def tracer_resultats(t, y, k1, figures_dir):
    """
    Trace les résultats de la simulation et sauvegarde les figures
    """
    # Extraire les variables
    a = y[:, 0]  # O₂
    b = y[:, 1]  # NADH
    x = y[:, 2]  # X
    y_var = y[:, 3]  # Y
    
    # Créer un identifiant pour les fichiers à partir de k1
    k1_str = f"{k1:.4f}".replace('.', '_')
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Éliminer le transitoire pour l'affichage
    start = len(t) // 2
    end = start + 5000
    
    # 1. Série temporelle (O₂)
    axes[0, 0].plot(t[start:end], a[start:end], 'b-', lw=0.7)
    axes[0, 0].set_xlabel('Temps (s)')
    axes[0, 0].set_ylabel('[A] (O₂)')
    axes[0, 0].set_title(f'Série temporelle - Oxygène (k₁ = {k1})')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Portrait de phase (O₂ vs B)
    axes[0, 1].plot(b[start:end], a[start:end], 'r-', lw=0.7)
    axes[0, 1].set_xlabel('[B] (NADH)')
    axes[0, 1].set_ylabel('[A] (O₂)')
    axes[0, 1].set_title('Portrait de phase (A vs B)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Toutes les espèces (zoom)
    axes[1, 0].plot(t[start:start+1000], a[start:start+1000], 'b-', 
                    label='A (O₂)', alpha=0.7)
    axes[1, 0].plot(t[start:start+1000], b[start:start+1000], 'g-', 
                    label='B (NADH)', alpha=0.7)
    axes[1, 0].plot(t[start:start+1000], x[start:start+1000], 'm-', 
                    label='X', alpha=0.7)
    axes[1, 0].plot(t[start:start+1000], y_var[start:start+1000], 'c-', 
                    label='Y', alpha=0.7)
    axes[1, 0].set_xlabel('Temps (s)')
    axes[1, 0].set_ylabel('Concentration')
    axes[1, 0].set_title('Toutes les espèces')
    axes[1, 0].legend(loc='upper right', fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Spectre de puissance (O₂)
    f, Pxx = periodogram(a[start:end], fs=10.0)
    axes[1, 1].semilogy(f[1:200], Pxx[1:200], 'k-', lw=0.8)
    axes[1, 1].set_xlabel('Fréquence (Hz)')
    axes[1, 1].set_ylabel('Puissance')
    axes[1, 1].set_title('Spectre de puissance (A)')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xlim([0, 0.5])
    
    plt.tight_layout()
    
    # Sauvegarde dans le dossier figures
    filename = os.path.join(figures_dir, f'dop_batch_k1_{k1_str}.png')
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"✅ Figure sauvegardée : {filename}")
    
    plt.show()


# ============================================================
# PROGRAMME PRINCIPAL INTERACTIF
# ============================================================

if __name__ == "__main__":
    
    print("\n" + "=" * 80)
    print("SIMULATION INTERACTIVE - MODÈLE DOP BATCH")
    print("Équations fournies par Doumbouya et al. (2025)")
    print("=" * 80)
    print("\nCe programme vous permet d'explorer le comportement")
    print("de la réaction Peroxydase-Oxydase en réacteur fermé (batch).")
    print("\nLe PARAMÈTRE DE BIFURCATION est k₁.")
    print("Toutes les figures sont sauvegardées dans le dossier 'figures_dop_batch'")
    print("=" * 80)
    
    # Conditions initiales (spécifiées)
    a0_init = 6.0      # a(0) = 6.0
    b0_init = 150.0    # b(0) = 150.0
    x0_init = 0.1      # x(0) = 0.1
    y0_init = 0.1      # y(0) = 0.1
    y0 = [a0_init, b0_init, x0_init, y0_init]
    
    print(f"\nConditions initiales FIXES :")
    print(f"  a(0) = {a0_init}")
    print(f"  b(0) = {b0_init}")
    print(f"  x(0) = {x0_init}")
    print(f"  y(0) = {y0_init}")
    print("=" * 80)
    
    # Boucle interactive
    continuer = True
    simulation_count = 0
    
    while continuer:
        print("\n" + "-" * 60)
        
        # Demander k1 (paramètre de bifurcation)
        try:
            k1_input = input(f"Entrez k₁ (entre {k1_min} et {k1_max}) [ou 'q' pour quitter, 'l' pour lister] : ")
            
            if k1_input.lower() == 'q':
                continuer = False
                print("\nFin des simulations. À bientôt !")
                break
                
            if k1_input.lower() == 'l':
                print("\nFichiers dans le dossier 'figures_dop_batch' :")
                fichiers = os.listdir(figures_dir)
                for f in sorted(fichiers):
                    print(f"  - {f}")
                continue
            
            k1 = float(k1_input)
            
            if k1 <= 0:
                print("⚠ k₁ doit être positif. Essayez encore.")
                continue
                
        except ValueError:
            print("⚠ Entrée invalide. Veuillez entrer un nombre, 'q' pour quitter, ou 'l' pour lister.")
            continue
        
        simulation_count += 1
        
        print(f"\n✅ k₁ = {k1}")
        
        # Créer le modèle
        modele = ModeleDOPBatch(k1=k1)
        
        # Temps de simulation (à ajuster selon la dynamique)
        t_max = 500  # secondes
        nb_points = 50000
        t = np.linspace(0, t_max, nb_points)
        
        print(f"\nSimulation en cours sur {t_max} s...")
        print(f"({nb_points} points de calcul)")
        
        # Simulation
        y = odeint(modele.modele, y0, t)
        
        print("✅ Simulation terminée.")
        
        # Analyse
        stats = analyser_oscillations(t, y)
        
        print("\n" + "-" * 40)
        print("RÉSULTATS DE L'ANALYSE")
        print("-" * 40)
        print(f"Type de comportement : {stats['type']}")
        print(f"Amplitude (A)        : {stats['amplitude']:.2f}")
        if stats['periode'] > 0:
            print(f"Période moyenne      : {stats['periode']:.1f} s")
        print(f"Nombre de pics       : {stats['nb_pics']}")
        print(f"Min/Max A            : {stats['a_min']:.2f} / {stats['a_max']:.2f}")
        print("-" * 40)
        
        # Tracer et sauvegarder
        tracer_resultats(t, y, k1, figures_dir)
        
        print(f"\n✅ Simulation {simulation_count} terminée pour k₁ = {k1}")
        print("-" * 60)
    
    print("\n" + "=" * 80)
    print(f"PROGRAMME TERMINÉ - {simulation_count} simulation(s) effectuée(s)")
    print(f"Toutes les figures sont dans le dossier '{figures_dir}'")
    print("=" * 80)