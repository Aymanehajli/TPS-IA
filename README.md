## TP IA – Projet global

Ce dépôt regroupe plusieurs **TP d’Intelligence Artificielle / Deep Learning** réalisés en Python avec TensorFlow et d’autres bibliothèques associées.

L’objectif est de couvrir différents cas d’usage :

- Apprentissage par renforcement (DQN)
- Génération de texte avec LSTM
- Classification d’images avec CNN

---

## Structure du dépôt

- `TP_agent/`  
  TP sur l’**apprentissage par renforcement** avec un **Deep Q-Network (DQN)** appliqué à l’environnement `MountainCar-v0` (Gym/Gymnasium).  
  Contient:
  - `cartpoleV1.ipynb` (notebook principal – implémentation du DQN)
  - `models/` (sauvegardes de modèles)
  - `venv/` (environnement virtuel local, à ne pas versionner)
  - `README.md` (description détaillée du TP, exécution, résultats, GitHub)

- `TP_Gutenberg/`  
  TP de **génération de texte** dans le style de Shakespeare à l’aide d’un **LSTM** (modélisation caractère par caractère) sur un corpus du Project Gutenberg.  
  Contient:
  - `gutenberg.py` (script principal)
  - `requirements.txt` (dépendances)
  - `reponse.txt` (compte rendu / réponses)
  - `README.md` (description détaillée, installation, exécution, résultats)

- `TP_mnist/`  
  TP de **classification d’images** MNIST avec un **réseau de neurones convolutionnel (CNN)**.  
  Contient:
  - `mnist.ipynb` (notebook principal)
  - `README.md` (description détaillée, installation, exécution, résultats)

Chaque sous-dossier possède un `README.md` spécifique avec les instructions complètes pour ce TP.

---

## Prérequis généraux

- **Python** 3.9+ recommandé
- **pip** installé
- Connexion Internet (pour le téléchargement des datasets/corpus la première fois)

Il est recommandé de créer un **environnement virtuel par TP** (décrit dans chaque README de sous-dossier) pour isoler les dépendances.

---

## Comment lancer chaque TP

Depuis la racine du projet (`IA/`) :

1. **TP Agent (DQN – MountainCar)**
   - Se placer dans le dossier:
     ```bash
     cd TP_agent
     ```
   - Créer/activer un venv et installer les dépendances (voir `TP_agent/README.md`).
   - Lancer Jupyter:
     ```bash
     jupyter notebook
     ```
   - Ouvrir `cartpoleV1.ipynb` et exécuter les cellules.

2. **TP Gutenberg (LSTM – texte Shakespeare)**
   - Se placer dans le dossier:
     ```bash
     cd TP_Gutenberg
     ```
   - Créer/activer un venv, puis:
     ```bash
     pip install -r requirements.txt
     python gutenberg.py
     ```
   - Tous les détails et interprétations de résultats sont dans `TP_Gutenberg/README.md`.

3. **TP MNIST (CNN – classification de chiffres)**
   - Se placer dans le dossier:
     ```bash
     cd TP_mnist
     ```
   - Créer/activer un venv, installer TensorFlow, etc. (voir `TP_mnist/README.md`).
   - Lancer Jupyter et ouvrir `mnist.ipynb`.

---

## Mise en ligne sur GitHub

Depuis la racine du dépôt (`IA/`), tu peux initialiser et pousser tout le projet sur GitHub:

```bash
git init
git add .
git commit -m "Initialisation des TP IA (DQN, Gutenberg LSTM, MNIST CNN)"
git branch -M main
git remote add origin <URL_DE_TON_DEPOT_GITHUB>
git push -u origin main
```

Pense à ajouter un fichier `.gitignore` à la racine pour **ne pas versionner les environnements virtuels**:

```gitignore
TP_agent/venv/
TP_Gutenberg/venv/
TP_mnist/venv/
```

Ce `README.md` donne une vue d’ensemble du projet. Pour les détails techniques et les commandes spécifiques, se référer aux `README.md` de chaque dossier de TP.

# TPS-IA
