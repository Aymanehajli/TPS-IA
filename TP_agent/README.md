## TP Agent – Deep Q-Network sur `MountainCar-v0`

Ce TP a pour objectif d’implémenter et d’entraîner un agent d’apprentissage par renforcement basé sur un **Deep Q-Network (DQN)** pour résoudre l’environnement `MountainCar-v0` (Gym/Gymnasium).  
L’agent apprend à utiliser l’inertie de la voiture pour atteindre le sommet de la colline.

- **Environnement**: `MountainCar-v0`
- **Type d’algorithme**: DQN (Deep Q-Learning)
- **Objectif pédagogique**:
  - Manipuler un environnement Gym/Gymnasium
  - Implémenter un DQN avec réseau de neurones Keras/TensorFlow
  - Utiliser un **experience replay buffer** et un **target network**
  - Suivre des métriques d’apprentissage et évaluer l’agent

---

## Contenu du TP

Le fichier principal est le notebook:

- `cartpoleV1.ipynb`  
  (malgré le nom, le code travaille sur l’environnement `MountainCar-v0`)

Fonctionnalités principales du notebook :

- **Configuration de l’environnement**:
  - Création de l’environnement `MountainCar-v0`
  - Lecture des dimensions de l’état et de l’espace d’actions
  - Définition des hyperparamètres (epsilon, gamma, learning rate, etc.)
- **Modèle DQN**:
  - Réseau de neurones `Sequential` avec:
    - 2 couches cachées de 24 neurones (activation ReLU)
    - 1 couche de sortie linéaire (une Q-value par action)
  - Optimiseur Adam et loss MSE
- **Experience Replay**:
  - Mémoire `deque` de taille max 100 000 transitions
  - Fonction de stockage et d’échantillonnage de batchs
- **Target Network**:
  - Réseau cible synchronisé périodiquement avec le réseau principal
  - Mise à jour tous les `target_update_frequency` épisodes
- **Politique d’exploration**:
  - Stratégie **epsilon-greedy** avec décroissance exponentielle de `epsilon`
- **Boucle d’entraînement**:
  - Interaction agent–environnement, stockage des transitions
  - Apprentissage par mini-batchs via la fonction de Bellman
  - Suivi de la récompense par épisode, longueur d’épisode, taux de succès
  - Sauvegarde du meilleur modèle (basé sur le taux de réussite / reward moyen)
- **Visualisation**:
  - Tracés:
    - Récompense par épisode et moyenne glissante
    - Taux de réussite
    - Longueur des épisodes
    - Distribution des récompenses
    - Comparaison succès/échecs
- **Évaluation & Vidéo**:
  - Évaluation de l’agent sur plusieurs épisodes sans exploration
  - Tentative d’enregistrement d’une vidéo du comportement de l’agent (`./video`)

Les modèles entraînés sont sauvegardés dans le dossier:

- `models/`

---

## Prérequis

- **Python** 3.9+ recommandé
- Gestionnaire de paquets: **pip**
- Environnement virtuel (recommandé): `venv` ou `conda`

Bibliothèques principales:

- `gym` ou `gymnasium`
- `tensorflow`
- `numpy`
- `matplotlib`
- `imageio`, `imageio-ffmpeg`

---

## Installation et mise en place de l’environnement

Depuis le dossier `TP_agent` :

```bash
python -m venv venv
source venv/bin/activate  # sous macOS / Linux
# ou
# venv\Scripts\activate   # sous Windows PowerShell
```

Installe les dépendances directement depuis le notebook (cellule d’installation) ou manuellement :

```bash
pip install gym gym[classic_control] tensorflow matplotlib imageio imageio-ffmpeg
```

---

## Exécution du TP

### 1. Lancer Jupyter Notebook (ou JupyterLab)

Toujours dans l’environnement virtuel activé:

```bash
pip install notebook  # si nécessaire
jupyter notebook
```

Puis ouvrir le fichier:

- `cartpoleV1.ipynb`

### 2. Exécuter le notebook

1. Lancer les cellules dans l’ordre (Kernel → Restart & Run All recommandé).
2. Vérifier que:
   - L’environnement `MountainCar-v0` se crée correctement.
   - Le modèle DQN est compilé sans erreur.
   - L’entraînement se lance et affiche les récompenses / taux de réussite.
3. À la fin:
   - Un **modèle final** et un **meilleur modèle** sont sauvegardés dans `models/`.
   - Des graphiques récapitulatifs de l’entraînement sont affichés.
   - Une évaluation finale sur plusieurs épisodes est réalisée.
   - Une vidéo de l’agent est éventuellement générée dans `video/`.

---

## Résultats attendus

- L’agent doit atteindre un **taux de réussite d’environ 90 %** sur les 100 derniers épisodes (récompense > -200).
- Les courbes de récompense doivent montrer une **amélioration progressive**.
- Le nombre de pas par épisode diminue généralement au fil de l’apprentissage.

Ce TP permet de:

- Comprendre le fonctionnement d’un DQN.
- Manipuler un buffer d’expérience et un target network.
- Mettre en place une politique d’exploration/d’exploitation.

---

## Préparation pour GitHub

Pour publier ce TP sur GitHub, depuis le dossier racine du projet (où se trouve `TP_agent`) :

```bash
git init
git add TP_agent
git commit -m "Ajout TP Agent DQN MountainCar"
git branch -M main
git remote add origin <URL_DE_TON_DEPOT_GITHUB>
git push -u origin main
```

Veille à **ne pas versionner** l’environnement virtuel (`TP_agent/venv/`) en ajoutant une entrée dans `.gitignore` :

```gitignore
TP_agent/venv/
```

Le fichier `README.md` présent dans `TP_agent` décrit entièrement le TP et est prêt pour être affiché correctement sur GitHub.

