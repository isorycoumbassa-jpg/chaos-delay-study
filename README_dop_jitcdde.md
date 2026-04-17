# Modèle DOP (Peroxidase-Oxidase) avec Feedback Retardé

## Description

Ce script implémente le modèle DOP (Degn-Olsen-Perram) pour la réaction Peroxidase-Oxydase (PO) avec feedback retardé de type Roesky (1993). Le modèle simule la dynamique de l'oxygène (O₂) et du NADH dans un réacteur batch.

## Référence

- Olsen, L. F., & Degn, H. (1977). Chaos in an enzyme reaction. *Nature*, 267, 177-178.
- Roesky, P. W., Doumbouya, S. I., & Schneider, F. W. (1993). Chaos in the Belousov-Zhabotinsky reaction induced by delayed feedback. *J. Phys. Chem.*, 97, 398-402.

## Paramètres par défaut

| Paramètre | Valeur | Description |
|-----------|--------|-------------|
| `k₁` | 0.08 | Paramètre de bifurcation |
| `β` | 0.0304 | Coupling strength (feedback gain) |
| `dt` | 1.0 s | Pas d'échantillonnage |
| `t_total` | 50000-100000 s | Temps total (adapté à la RAM) |
| `t_transient` | 5000 s | Transitoire éliminé |
| `CI` | [6.0, 150.0, 0.1, 0.1] | Conditions initiales [O₂, NADH, X, Y] |

## Variables du modèle

| Variable | Espèce chimique | Unité |
|----------|-----------------|-------|
| `y(0)` | O₂ (oxygène) | µM |
| `y(1)` | NADH | µM |
| `y(2)` | X (intermédiaire) | µM |
| `y(3)` | Y (intermédiaire) | µM |

## Installation

```bash
# Cloner le dépôt
git clone https://github.com/coumbassa/chaos-delay-study
cd chaos-delay-study

# Installer les dépendances
pip install numpy matplotlib scipy jitcdde psutil
## Utilisation

python3 dop_jitcdde.p

Menu Interactif

MENU PRINCIPAL
--------------------------------------------------
1. Changer/entrer la valeur de β (feedback gain)
2. Simuler une valeur de D
3. Afficher le résumé des simulations
4. Sauvegarder et quitter
5. Quitter sans sauvegarder

## Exemple de session

1. Entrer β : 0.0304
2. Entrer D : 245
3. Attendre la simulation
4. Visualiser les résultats

## Detection automatique de l'environnement

Le script détecte automatiquement 
la RAM disponible et ajuste les paramètres :

Environnement RAM T_TOTAL SVD/RQA
Local (Mac) < 16 GB 50000 s Désactivés
Cloud (GCP) ≥ 16 GB 100000 s Activés

## Auteur
Ibrahima Sory Coumbassa (formerly Sory I. Doumbouya)
Université de Conakry, Guinée

