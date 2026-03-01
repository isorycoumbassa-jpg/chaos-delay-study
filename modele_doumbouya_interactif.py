#!/usr/bin/env python3
"""
Modèle Oregonator réversible à quatre variables
Doumbouya et al., J. Phys. Chem. 1993, 97, 1025-1031
Volume : 20 mL - Sans feedback
Version interactive avec sauvegarde automatique des figures
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.signal import find_peaks, periodogram
import os

# ============================================================
# PARAMÈTRES FIXES
# ============================================================
BrO3_const = 0.1
Hplus_const = 0.3
Ce_total = 8.33e-4
V_reactor = 20.0

ai1 = 2.0 * BrO3_const * (Hplus_const**2)
ai2 = 3.0e6 * Hplus_const
ai3 = 42.0 * BrO3_const * Hplus_const
ai4 = 4.2e7
ai5 = 8.0e4 * Hplus_const
ai6 = 8900.0
ai7 = 3000.0
ai8 = 0.1
g = 0.833

# Dossier pour les figures
figures_dir = "figures_doumbouya"
os.makedirs(figures_dir, exist_ok=True)

class ModeleDoumbouya:
    def __init__(self, kf):
        self.a11 = ai1; self.a12 = ai2; self.a13 = ai3; self.a14 = ai4
        self.a15 = ai5; self.a16 = ai6; self.a17 = ai7; self.a18 = ai8
        self.g = g; self.BrO3 = BrO3_const; self.Hplus = Hplus_const
        self.kf = kf
        
    def modele(self, y, t):
        X, Y, Z, W = y
        X = max(X, 1e-20); Y = max(Y, 1e-20); Z = max(Z, 1e-20); W = max(W, 1e-20)
        
        dXdt = (- self.a11 * X - self.a12 * X * Y + self.g * self.a18 * W - self.kf * X)
        dYdt = (+ self.a11 * X - self.a12 * X * Y - self.a13 * Y + self.a14 * Z**2
                + self.a15 * Z * (Ce_total - W) - self.a16 * Y * W
                - 2 * self.a17 * Y**2 - self.kf * Y)
        dZdt = (+ 2 * self.a13 * Y - 2 * self.a14 * Z**2
                - self.a15 * Z * (Ce_total - W) + self.a16 * Y * W - self.kf * Z)
        dWdt = (+ self.a15 * Z * (Ce_total - W) - self.a16 * Y * W
                - self.a18 * W - self.kf * W)
        return [dXdt, dYdt, dZdt, dWdt]

def tracer_resultats(t, y, kf, figures_dir):
    a, b, x, w = y[:,0]*1e6, y[:,1]*1e6, y[:,2]*1e6, y[:,3]*1e6
    kf_str = f"{kf:.2e}".replace('.', '_').replace('e', 'E')
    
    fig, axes = plt.subplots(2, 2, figsize=(14,10))
    start = len(t)//2; end = start + 5000
    
    axes[0,0].plot(t[start:end], w[start:end], 'b-', lw=0.7)
    axes[0,0].set_xlabel('Temps (s)'); axes[0,0].set_ylabel('[Ce⁴⁺] (µM)')
    axes[0,0].set_title(f'Ce⁴⁺ (kf={kf:.2e})'); axes[0,0].grid(True)
    
    axes[0,1].plot(b[start:end], w[start:end], 'r-', lw=0.7)
    axes[0,1].set_xlabel('[Br⁻] (µM)'); axes[0,1].set_ylabel('[Ce⁴⁺] (µM)')
    axes[0,1].set_title('Portrait de phase'); axes[0,1].grid(True)
    
    for idx, (var, nom, coul) in enumerate([(a,'A','b'), (b,'B','g'), (x,'X','m'), (w,'W','c')]):
        axes[1,0].plot(t[start:start+1000], var[start:start+1000], color=coul, label=nom, alpha=0.7)
    axes[1,0].set_xlabel('Temps (s)'); axes[1,0].set_ylabel('Concentration (µM)')
    axes[1,0].set_title('Toutes espèces'); axes[1,0].legend(); axes[1,0].grid(True)
    
    f, Pxx = periodogram(w[start:end], fs=10.0)
    axes[1,1].semilogy(f[1:200], Pxx[1:200], 'k-')
    axes[1,1].set_xlabel('Fréquence (Hz)'); axes[1,1].set_ylabel('Puissance')
    axes[1,1].set_title('Spectre'); axes[1,1].grid(True); axes[1,1].set_xlim([0,0.5])
    
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, f'doumbouya_kf_{kf_str}.png'), dpi=150)
    plt.show()

if __name__ == "__main__":
    print("\nModèle Doumbouya et al. (1993) - Volume 20 mL")
    y0 = [1e-6, 1e-6, 1e-6, 1e-6]
    
    while True:
        kf_input = input("\nkf (s⁻¹) [q=quitter, l=lister] : ")
        if kf_input == 'q': break
        if kf_input == 'l':
            print("\nFichiers :", os.listdir(figures_dir))
            continue
        try:
            kf = float(kf_input)
            if kf <= 0: continue
            modele = ModeleDoumbouya(kf)
            t = np.linspace(0, max(5000, 10/kf), 50000)
            y = odeint(modele.modele, y0, t)
            tracer_resultats(t, y, kf, figures_dir)
        except: print("Erreur")
