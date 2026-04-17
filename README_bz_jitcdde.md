
---

## README pour `bz_jitcdde.py`

```markdown
# Modèle BZ (Belousov-Zhabotinsky) avec Feedback Retardé

## Description

Ce script implémente le modèle Oregonator réversible à quatre variables (Doumbouya et al. 1993) pour la réaction de Belousov-Zhabotinsky (BZ) avec feedback retardé de type Roesky (1993). Le modèle simule la dynamique des espèces chimiques dans un CSTR.

## Références

- Doumbouya, S. I., Muenster, A. F., Doona, C. J., & Schneider, F. W. (1993). Reversible Oregonator model for the Belousov-Zhabotinsky reaction coupled to a flow. *J. Phys. Chem.*, 97, 1025-1031.
- Roesky, P. W., Doumbouya, S. I., & Schneider, F. W. (1993). Chaos in the Belousov-Zhabotinsky reaction induced by delayed feedback. *J. Phys. Chem.*, 97, 398-402.

## Paramètres par défaut

| Paramètre | Valeur | Description |
|-----------|--------|-------------|
| `[BrO₃⁻]` | 0.1 M | Concentration en bromate |
| `[H⁺]` | 0.3 M | Concentration en ions H⁺ |
| `Ce_total` | 8.33e-4 M | Concentration totale (Ce³⁺ + Ce⁴⁺) |
| `k₀` | 7.58e-4 s⁻¹ | Débit (τ_res = 22 min) |
| `dt` | 0.1 s | Pas d'échantillonnage |
| `t_total` | 50000-100000 s | Temps total (adapté à la RAM) |
| `CI` | [1e-5, 1e-6, 1e-10, 1e-4] M | Conditions initiales |

## Variables du modèle (Oregonator)

| Variable | Espèce chimique | Unité |
|----------|-----------------|-------|
| `y(0)` | Br⁻ (bromure) | M → µM |
| `y(1)` | HBrO₂ (acide hypobromeux) | M → µM |
| `y(2)` | BrO₂• (radical) | M → µM |
| `y(3)` | Ce⁴⁺ (cérium IV) | M → µM |

## Installation

```bash
# Cloner le dépôt
git clone https://github.com/coumbassa/chaos-delay-study
cd chaos-delay-study

# Installer les dépendances
pip install numpy matplotlib scipy jitcdde psutil