📘 README pour modele_doumbouya_interactif.py

```markdown
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
```

Lancer le script

```bash
python3 modele_doumbouya_interactif.py
```

Commandes interactives

· Entrez une valeur de kf (en s⁻¹) pour lancer une simulation
· Tapez l pour lister les figures déjà générées
· Tapez q pour quitter

📊 Résultats

Chaque simulation génère automatiquement :

· Une figure avec 4 graphiques (série temporelle, portrait de phase, toutes les espèces, spectre)
· Sauvegarde dans le dossier figures_doumbouya/
· Nom de fichier : doumbouya_kf_1_50E-03.png (exemple pour kf = 0.0015)

📈 Exemples de valeurs à tester

kf (s⁻¹) Temps de séjour Comportement typique
0.0015 ~11 min Oscillations lentes
0.0020 ~8.3 min Oscillations plus rapides
0.0030 ~5.6 min Chaos possible
0.0005 ~33 min Très lent

📚 Référence

Doumbouya, S. I. et al. (1993). J. Phys. Chem., 97, 1025-1031.
