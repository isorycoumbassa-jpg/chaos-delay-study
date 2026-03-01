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
