## 📗 README pour `dop_batch_interactif.py`

```markdown
# Modèle DOP (Degn-Olsen-Perram) pour la réaction Peroxydase-Oxydase (PO)

Ce script Python permet d'explorer le comportement du modèle DOP en réacteur fermé (batch), basé sur :

> Olsen, L. F. (2024).  
> *Stern–Brocot arithmetic in dynamics of a biochemical reaction model.*  
> Chaos, 34, 123107.

## 🧪 Modèle

- **Variables** : a = [O₂], b = [NADH], x = [X], y = [Y]
- **Réacteur fermé** : pas de flux (batch)
- **Paramètre de bifurcation** : `k₁` (constante de vitesse)
- **Paramètres fixes** : issus du tableau fourni

### Équations

```

da/dt = -k₁ a b x - k₃ a b y + k₇ (a₀ - a)
db/dt = -k₁ a b x - k₃ a b y + k₈
dx/dt =  k₁ a b x - 2k₂ x² + 2k₃ a b y - k₄ x + k₆
dy/dt =  2k₂ x² - k₃ a b y - k₅ y

```

### Paramètres utilisés

| Paramètre | Valeur | Plage d'origine |
|-----------|--------|-----------------|
| k₁ | variable | 0.04 - 0.15 (oscillations) |
| k₂ | 800.0 | 400 - 1250 |
| k₃ | 0.05 | 0.035 - 0.065 |
| k₄ | 20.0 | fixe |
| k₅ | 1.5 | fixe |
| k₆ | 0.001 | 10⁻³ |
| a₀ | 8.0 | fixe |
| k₇ | 0.1 | fixe |
| k₈ | 0.61 | 0.5 - 0.72 |

## 🚀 Utilisation

### Prérequis
```bash
pip install numpy matplotlib scipy
```

Lancer le script

```bash
python3 dop_batch_interactif.py
```

Commandes interactives

· Entrez une valeur de k₁ pour lancer une simulation
· Tapez l pour lister les figures déjà générées
· Tapez q pour quitter

📊 Résultats

Chaque simulation génère automatiquement :

· Une figure avec 4 graphiques (série temporelle, portrait de phase, toutes les espèces, spectre)
· Sauvegarde dans le dossier figures_dop_batch/
· Nom de fichier : dop_k1_0_0500.png (exemple pour k₁ = 0.05)

📈 Plage d'oscillations identifiée

D'après les tests systématiques :

· Zone d'oscillations : k₁ = 0.04 à 0.15
· Amplitude maximale : vers k₁ ≈ 0.05
· En dessous de 0.04 : état stationnaire
· Au-dessus de 0.15 : retour à l'état stationnaire

Valeurs recommandées

· 0.05 → oscillations bien établies
· 0.07 → oscillations plus rapides
· 0.10 → fréquence élevée
· 0.03 → limite inférieure (transitoire long)
· 0.20 → au-delà du seuil

📚 Référence

Olsen, L. F. (2024). Chaos, 34, 123107.
