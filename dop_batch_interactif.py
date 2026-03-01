#!/usr/bin/env python3
"""
Modèle DOP (Degn-Olsen-Perram) - Réaction PO
Batch - k₁ paramètre de bifurcation
Version interactive avec sauvegarde automatique
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.signal import find_peaks, periodogram
import os

# ============================================================
# PARAMÈTRES FIXES
# ============================================================
k2, k3, k4, k5, k6 = 800.0, 0.05, 20.0, 1.5, 0.001
a0, k7, k8 = 8.0, 0.1, 0.61

figures_dir = "figures_dop_batch"
os.makedirs(figures_dir, exist_ok=True)

class ModeleDOP:
    def __init__(self, k1):
        self.k1, self.k2, self.k3 = k1, k2, k3
        self.k4, self.k5, self.k6 = k4, k5, k6
        self.a0, self.k7, self.k8 = a0, k7, k8
        
    def modele(self, y, t):
        a, b, x, yv = y
        a = max(a,1e-10); b = max(b,1e-10); x = max(x,1e-10); yv = max(yv,1e-10)
        
        da = -self.k1*a*b*x - self.k3*a*b*yv + self.k7*(self.a0 - a)
        db = -self.k1*a*b*x - self.k3*a*b*yv + self.k8
        dx =  self.k1*a*b*x - 2*self.k2*x**2 + 2*self.k3*a*b*yv - self.k4*x + self.k6
        dy =  2*self.k2*x**2 - self.k3*a*b*yv - self.k5*yv
        return [da, db, dx, dy]

def tracer_resultats(t, y, k1, figures_dir):
    a, b, x, yv = y[:,0], y[:,1], y[:,2], y[:,3]
    k1_str = f"{k1:.4f}".replace('.', '_')
    
    fig, axes = plt.subplots(2,2,figsize=(14,10))
    start, end = len(t)//2, len(t)//2+5000
    
    axes[0,0].plot(t[start:end], a[start:end], 'b-', lw=0.7)
    axes[0,0].set_xlabel('Temps (s)'); axes[0,0].set_ylabel('[A] (O₂)')
    axes[0,0].set_title(f'Oxygène (k₁={k1})'); axes[0,0].grid(True)
    
    axes[0,1].plot(b[start:end], a[start:end], 'r-', lw=0.7)
    axes[0,1].set_xlabel('[B] (NADH)'); axes[0,1].set_ylabel('[A] (O₂)')
    axes[0,1].set_title('Portrait A vs B'); axes[0,1].grid(True)
    
    for val, nom, coul in [(a,'A','b'), (b,'B','g'), (x,'X','m'), (yv,'Y','c')]:
        axes[1,0].plot(t[start:start+1000], val[start:start+1000], color=coul, label=nom, alpha=0.7)
    axes[1,0].set_xlabel('Temps (s)'); axes[1,0].set_ylabel('Concentration')
    axes[1,0].set_title('Toutes espèces'); axes[1,0].legend(); axes[1,0].grid(True)
    
    f, Pxx = periodogram(a[start:end], fs=10.0)
    axes[1,1].semilogy(f[1:200], Pxx[1:200], 'k-')
    axes[1,1].set_xlabel('Fréquence (Hz)'); axes[1,1].set_ylabel('Puissance')
    axes[1,1].set_title('Spectre'); axes[1,1].grid(True); axes[1,1].set_xlim([0,0.5])
    
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, f'dop_k1_{k1_str}.png'), dpi=150)
    plt.show()

if __name__ == "__main__":
    print("\nModèle DOP batch - Paramètre de bifurcation k₁")
    y0 = [6.0, 150.0, 0.1, 0.1]
    
    while True:
        k1_input = input("\nk₁ [q=quitter, l=lister] : ")
        if k1_input == 'q': break
        if k1_input == 'l':
            print("\nFichiers :", os.listdir(figures_dir))
            continue
        try:
            k1 = float(k1_input)
            modele = ModeleDOP(k1)
            t = np.linspace(0, 1000, 100000)
            y = odeint(modele.modele, y0, t)
            tracer_resultats(t, y, k1, figures_dir)
        except: print("Erreur")