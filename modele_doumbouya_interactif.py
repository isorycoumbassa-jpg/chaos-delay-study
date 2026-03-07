#!/usr/bin/env python3
"""
Modèle Oregonator réversible à quatre variables
Doumbouya et al., J. Phys. Chem. 1993, 97, 1025-1031
Volume du réacteur : 20 mL
Sans feedback - pour exploration par les doctorants

Version avec sauvegarde automatique des figures
Entrée interactive du taux de dilution (kf)

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
# PARAMÈTRES FIXES DU MODÈLE (Doumbouya et al. 1993)
# ============================================================

# Concentrations constantes dans le réacteur
BrO3_const = 0.1      # [BrO₃⁻] = 0.1 M
Hplus_const = 0.3     # [H⁺] = 0.3 M
Ce_total = 8.33e-4    # [Ce³⁺ + Ce⁴⁺] = 8.33 × 10⁻⁴ M

# Volume du réacteur
V_reactor = 20.0      # 20 mL (comme dans Doumbouya et al.)

# Calcul des constantes de vitesse
ai1 = 2.0 * BrO3_const * (Hplus_const**2)        # s⁻¹
ai2 = 3.0e6 * Hplus_const                          # M⁻¹s⁻¹
ai3 = 42.0 * BrO3_const * Hplus_const              # s⁻¹
ai4 = 4.2e7                                         # M⁻¹s⁻¹
ai5 = 8.0e4 * Hplus_const                           # M⁻¹s⁻¹
ai6 = 8900.0                                        # M⁻¹s⁻¹
ai7 = 3000.0                                        # M⁻¹s⁻¹
ai8 = 0.1                                           # s⁻¹

# Facteur stoechiométrique
g = 0.833

print("=" * 80)
print("MODÈLE OREGONATOR RÉVERSIBLE À 4 VARIABLES")
print("Doumbouya et al., J. Phys. Chem. 1993, 97, 1025-1031")
print("=" * 80)
print(f"Concentrations constantes :")
print(f"  [BrO₃⁻] = {BrO3_const} M")
print(f"  [H⁺]    = {Hplus_const} M")
print(f"  [Ce]tot = {Ce_total:.2e} M")
print(f"Volume du réacteur : {V_reactor} mL")
print("=" * 80)

# Créer un dossier pour les figures s'il n'existe pas
figures_dir = "figures_doumbouya"
if not os.path.exists(figures_dir):
    os.makedirs(figures_dir)
    print(f"Dossier '{figures_dir}' créé pour sauvegarder les figures.")
else:
    print(f"Les figures seront sauvegardées dans le dossier '{figures_dir}'.")
print("=" * 80)


# ============================================================
# CLASSE DU MODÈLE (sans feedback)
# ============================================================

class ModeleDoumbouya:
    """
    Modèle réversible à 4 variables - sans feedback
    
    Variables d'état (en M) :
        X : [Br⁻]  (ions bromure)
        Y : [HBrO₂] (acide hypobromeux)
        Z : [BrO₂•] (radical dioxyde de brome)
        W : [Ce⁴⁺]  (cérium IV)
    """
    
    def __init__(self, kf):
        """
        Initialise le modèle avec un taux de dilution donné
        kf : taux de dilution (s⁻¹)
        """
        self.a11 = ai1
        self.a12 = ai2
        self.a13 = ai3
        self.a14 = ai4
        self.a15 = ai5
        self.a16 = ai6
        self.a17 = ai7
        self.a18 = ai8
        self.g = g
        self.BrO3 = BrO3_const
        self.Hplus = Hplus_const
        self.kf = kf
        
    def modele(self, y, t):
        """
        Équations différentielles du modèle
        y = [X, Y, Z, W]
        """
        X, Y, Z, W = y
        
        # Protection contre les valeurs négatives
        X = max(X, 1e-20)
        Y = max(Y, 1e-20)
        Z = max(Z, 1e-20)
        W = max(W, 1e-20)
        
        # Équations (d'après Doumbouya et al. 1993)
        dXdt = (- self.a11 * X 
                - self.a12 * X * Y
                + self.g * self.a18 * W
                - self.kf * X)
        
        dYdt = (+ self.a11 * X
                - self.a12 * X * Y
                - self.a13 * Y
                + self.a14 * Z**2
                + self.a15 * Z * (Ce_total - W)
                - self.a16 * Y * W
                - 2 * self.a17 * Y**2
                - self.kf * Y)
        
        dZdt = (+ 2 * self.a13 * Y
                - 2 * self.a14 * Z**2
                - self.a15 * Z * (Ce_total - W)
                + self.a16 * Y * W
                - self.kf * Z)
        
        dWdt = (+ self.a15 * Z * (Ce_total - W)
                - self.a16 * Y * W
                - self.a18 * W
                - self.kf * W)
        
        return [dXdt, dYdt, dZdt, dWdt]


# ============================================================
# FONCTIONS D'ANALYSE SIMPLIFIÉES
# ============================================================

def analyser_oscillations(t, y):
    """
    Analyse simple des oscillations
    """
    W = y[:, 3] * 1e6  # Ce⁴⁺ en µM
    
    # Éliminer le transitoire (première moitié)
    miroir = len(t) // 2
    W_stead = W[miroir:]
    
    # Statistiques
    W_min = np.min(W_stead)
    W_max = np.max(W_stead)
    W_mean = np.mean(W_stead)
    amplitude = W_max - W_min
    
    # Détection des pics
    peaks, _ = find_peaks(W_stead, distance=50, height=W_mean)
    nb_pics = len(peaks)
    
    # Période si assez de pics
    periode = 0
    if nb_pics >= 3:
        t_stead = t[miroir:]
        t_pics = t_stead[peaks]
        periodes = np.diff(t_pics)
        periode = np.mean(periodes)
    
    # Type de comportement
    if amplitude < 0.5:
        type_comport = "État stationnaire"
    elif nb_pics < 5:
        type_comport = "Transitoire long"
    elif periode > 0:
        type_comport = f"Oscillations Périodiques (P{min(5, nb_pics//5+1)})"
    else:
        type_comport = "Comportement complexe"
    
    return {
        'amplitude': amplitude,
        'periode': periode,
        'nb_pics': nb_pics,
        'type': type_comport,
        'W_min': W_min,
        'W_max': W_max,
        'W_mean': W_mean
    }


def tracer_resultats(t, y, kf, figures_dir):
    """
    Trace les résultats de la simulation et sauvegarde les figures
    """
    # Conversion en µM pour lisibilité
    X_um = y[:, 0] * 1e6
    Y_um = y[:, 1] * 1e6
    Z_um = y[:, 2] * 1e6
    W_um = y[:, 3] * 1e6
    
    # Créer un identifiant pour les fichiers à partir de kf
    kf_str = f"{kf:.2e}".replace('.', '_').replace('e', 'E')
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Éliminer le transitoire pour l'affichage
    start = len(t) // 2
    end = start + 5000
    
    # 1. Série temporelle (Ce⁴⁺)
    axes[0, 0].plot(t[start:end], W_um[start:end], 'b-', lw=0.7)
    axes[0, 0].set_xlabel('Temps (s)')
    axes[0, 0].set_ylabel('[Ce⁴⁺] (µM)')
    axes[0, 0].set_title(f'Série temporelle - Ce⁴⁺ (kf = {kf:.2e} s⁻¹)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Portrait de phase (HBrO₂ vs Ce⁴⁺)
    axes[0, 1].plot(Y_um[start:end], W_um[start:end], 'r-', lw=0.7)
    axes[0, 1].set_xlabel('[HBrO₂] (µM)')
    axes[0, 1].set_ylabel('[Ce⁴⁺] (µM)')
    axes[0, 1].set_title('Portrait de phase')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Toutes les espèces (zoom)
    axes[1, 0].plot(t[start:start+1000], X_um[start:start+1000], 'c-', 
                    label='Br⁻', alpha=0.7)
    axes[1, 0].plot(t[start:start+1000], Y_um[start:start+1000], 'g-', 
                    label='HBrO₂', alpha=0.7)
    axes[1, 0].plot(t[start:start+1000], Z_um[start:start+1000], 'm-', 
                    label='BrO₂•', alpha=0.7)
    axes[1, 0].plot(t[start:start+1000], W_um[start:start+1000], 'b-', 
                    label='Ce⁴⁺', alpha=0.7)
    axes[1, 0].set_xlabel('Temps (s)')
    axes[1, 0].set_ylabel('Concentration (µM)')
    axes[1, 0].set_title('Toutes les espèces chimiques')
    axes[1, 0].legend(loc='upper right', fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Spectre de puissance (Ce⁴⁺)
    f, Pxx = periodogram(W_um[start:end], fs=10.0)
    axes[1, 1].semilogy(f[1:200], Pxx[1:200], 'k-', lw=0.8)
    axes[1, 1].set_xlabel('Fréquence (Hz)')
    axes[1, 1].set_ylabel('Puissance')
    axes[1, 1].set_title('Spectre de puissance (Ce⁴⁺)')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xlim([0, 0.5])
    
    plt.tight_layout()
    
    # Sauvegarde dans le dossier figures
    filename = os.path.join(figures_dir, f'simulation_kf_{kf_str}.png')
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"✅ Figure sauvegardée : {filename}")
    
    plt.show()


def sauvegarder_donnees(t, y, kf, figures_dir):
    """
    Sauvegarde les données brutes de la simulation
    """
    kf_str = f"{kf:.2e}".replace('.', '_').replace('e', 'E')
    filename = os.path.join(figures_dir, f'donnees_kf_{kf_str}.npz')
    
    np.savez(filename,
             t=t,
             X=y[:, 0],
             Y=y[:, 1],
             Z=y[:, 2],
             W=y[:, 3],
             kf=kf)
    
    print(f"✅ Données sauvegardées : {filename}")


# ============================================================
# PROGRAMME PRINCIPAL INTERACTIF
# ============================================================

if __name__ == "__main__":
    
    print("\n" + "=" * 80)
    print("SIMULATION INTERACTIVE - MODÈLE DE DOUMBOUYA ET AL. (1993)")
    print("=" * 80)
    print("\nCe programme vous permet d'explorer le comportement")
    print("du modèle en entrant différentes valeurs du taux de dilution kf.")
    print("\nRappel : le temps de séjour τ_res = 1/kf (en secondes)")
    print("Pour τ_res = 11 min (660 s) → kf = 1/660 = 0.00152 s⁻¹")
    print("\nToutes les figures sont sauvegardées dans le dossier 'figures_doumbouya'")
    print("=" * 80)
    
    # Conditions initiales (concentrations typiques en M)
    X0 = 1e-6    # [Br⁻] initial
    Y0 = 1e-6    # [HBrO₂] initial
    Z0 = 1e-6    # [BrO₂•] initial
    W0 = 1e-6    # [Ce⁴⁺] initial
    y0 = [X0, Y0, Z0, W0]
    
    print(f"\nConditions initiales fixes :")
    print(f"  [Br⁻]   = {X0*1e6:.2f} µM")
    print(f"  [HBrO₂] = {Y0*1e6:.2f} µM")
    print(f"  [BrO₂•] = {Z0*1e6:.2f} µM")
    print(f"  [Ce⁴⁺]  = {W0*1e6:.2f} µM")
    print("=" * 80)
    
    # Boucle interactive
    continuer = True
    simulation_count = 0
    
    while continuer:
        print("\n" + "-" * 60)
        
        # Demander le taux de dilution
        try:
            kf_input = input("Entrez le taux de dilution kf (en s⁻¹) [ou 'q' pour quitter, 'l' pour lister] : ")
            
            if kf_input.lower() == 'q':
                continuer = False
                print("\nFin des simulations. À bientôt !")
                break
                
            if kf_input.lower() == 'l':
                print("\nFichiers dans le dossier 'figures_doumbouya' :")
                fichiers = os.listdir(figures_dir)
                for f in sorted(fichiers):
                    print(f"  - {f}")
                continue
            
            kf = float(kf_input)
            
            if kf <= 0:
                print("⚠ Le taux de dilution doit être positif. Essayez encore.")
                continue
                
        except ValueError:
            print("⚠ Entrée invalide. Veuillez entrer un nombre, 'q' pour quitter, ou 'l' pour lister.")
            continue
        
        simulation_count += 1
        
        # Afficher le temps de séjour correspondant
        tau_res = 1.0 / kf
        print(f"\n✅ Taux de dilution kf = {kf:.2e} s⁻¹")
        print(f"   Temps de séjour correspondant : {tau_res:.1f} s ({tau_res/60:.2f} min)")
        
        # Créer le modèle
        modele = ModeleDoumbouya(kf)
        
        # Temps de simulation (proportionnel au temps de séjour)
        t_max = max(5000, 10 * tau_res)  # Au moins 10 fois le temps de séjour
        nb_points = min(100000, int(t_max * 10))  # 10 points par seconde max
        t = np.linspace(0, t_max, nb_points)
        
        print(f"\nSimulation en cours sur {t_max:.0f} s ({t_max/60:.1f} min)...")
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
        print(f"Amplitude (Ce⁴⁺)    : {stats['amplitude']:.2f} µM")
        if stats['periode'] > 0:
            print(f"Période moyenne      : {stats['periode']:.1f} s")
        print(f"Nombre de pics       : {stats['nb_pics']}")
        print(f"Min/Max Ce⁴⁺         : {stats['W_min']:.2f} / {stats['W_max']:.2f} µM")
        print("-" * 40)
        
        # Tracer et sauvegarder
        tracer_resultats(t, y, kf, figures_dir)
        
        # Sauvegarder les données brutes (optionnel - décommentez si souhaité)
        # sauvegarder_donnees(t, y, kf, figures_dir)
        
        print(f"\n✅ Simulation {simulation_count} terminée pour kf = {kf:.2e}")
        print("-" * 60)
    
    print("\n" + "=" * 80)
    print(f"PROGRAMME TERMINÉ - {simulation_count} simulation(s) effectuée(s)")
    print(f"Toutes les figures sont dans le dossier '{figures_dir}'")
    print("=" * 80)
