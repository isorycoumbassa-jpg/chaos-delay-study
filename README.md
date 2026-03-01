# Chaos Delay Study

Codes Python pour la simulation des oscillateurs chimiques.

## 📁 Fichiers

- `modele_doumbouya_interactif.py` : Modèle BZ (Oregonator réversible, 20 mL)
- `dop_batch_interactif.py` : Modèle PO (DOP batch, k₁ paramètre de bifurcation)

## 🚀 Utilisation

```bash
python3 modele_doumbouya_interactif.py
python3 dop_batch_interactif.py

---

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
da/dt = -k₁ a b x - k₃ a b y + k₇ (a₀ - a)
db/dt = -k₁ a b x - k₃ a b y + k₈
dx/dt =  k₁ a b x - 2k₂ x² + 2k₃ a b y - k₄ x + k₆
dy/dt =  2k₂ x² - k₃ a b y - k₅ y

## README pour modele_doumbouya_interactif.py
# Modèle de Doumbouya et al. (1993) - Oregonator réversible à 4 variables

Ce script Python permet d'explorer le comportement du modèle de la réaction de Belousov-Zhabotinsky (BZ) publié dans :

> Doumbouya, S. I., Muenster, A. F., Doona, C. J., & Schneider, F. W. (1993).  
> *Deterministic chaos in serially coupled chemical oscillators.*  
> Journal of Physical Chemistry, 97(5), 1025-1031.

## 🧪 Modèle

- **Variables** : X = [Br⁻], Y = [HBrO₂], Z = [BrO₂•], W = [Ce⁴⁺]
- **Volume du réacteur** : 20 mL (comme dans l'article original)
- **Paramètre variable** : taux de dilution `kf` (s⁻¹)
- **Pas de feedback** — modèle de base pour comprendre la dynamique naturelle

## 🚀 Utilisation

### Prérequis
```bash
pip install numpy matplotlib scipy
