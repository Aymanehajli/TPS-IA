## TP MNIST – Classification de chiffres manuscrits avec CNN

Ce TP a pour objectif de construire et d’entraîner un **réseau de neurones convolutionnel (CNN)** pour classer les images de chiffres manuscrits du célèbre dataset **MNIST**.

- **Dataset**: MNIST (`tensorflow.keras.datasets.mnist`)
- **Tâche**: Classification de chiffres (0–9)
- **Objectif pédagogique**:
  - Charger et prétraiter un dataset d’images
  - Concevoir une architecture CNN simple mais efficace
  - Entraîner, évaluer et visualiser les performances du modèle

---

## Contenu du TP

- `mnist.ipynb`  
  Notebook principal contenant l’intégralité du TP.

Structure générale du notebook:

1. **Introduction**  
   Rappel du but: classifier les images MNIST avec un CNN.

2. **Chargement du dataset**  
   - `train_images`, `train_labels`
   - `test_images`, `test_labels`
   - Affichage de la forme des tableaux et des valeurs de pixels.

3. **Prétraitement des données**  
   - Normalisation des images dans l’intervalle `[0, 1]` (division par 255.0).
   - Reshape pour ajouter la dimension de canal:  
     `(N, 28, 28)` → `(N, 28, 28, 1)` pour compatibilité avec `Conv2D`.

4. **Conception du modèle CNN**  
   Modèle `Sequential` typique:
   - `Conv2D(5, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1))`
   - `MaxPooling2D(pool_size=(2, 2))`
   - `Conv2D(20, kernel_size=(3, 3), activation='relu')`
   - `MaxPooling2D(pool_size=(2, 2))`
   - `Conv2D(30, kernel_size=(3, 3), activation='relu')`
   - `Flatten()`
   - `Dense(50, activation='relu')`
   - `Dense(10, activation='softmax')` (une probabilité par chiffre 0–9)

5. **Compilation du modèle**  
   - Optimiseur: `adam`
   - Loss: `sparse_categorical_crossentropy`
   - Métrique: `accuracy`

6. **Entraînement**  
   - `model.fit(...)` sur quelques époques (ex: 5)
   - Avec `validation_data=(test_images, test_labels)` pour suivre la validation.

7. **Évaluation & visualisations**  
   - `model.evaluate(...)` sur l’ensemble de test (perte et précision).
   - Graphiques:
     - Précision train vs validation
     - Perte train vs validation
   - Affichage de quelques **prédictions** avec images et étiquettes prédites / réelles.

---

## Prérequis

- **Python** 3.9+ recommandé
- Gestionnaire de paquets: **pip**

Bibliothèques principales:

- `tensorflow`
- `numpy`
- `matplotlib`
- `jupyter` (pour exécuter le notebook)

---

## Installation

Depuis le dossier `TP_mnist` :

### 1. Créer et activer un environnement virtuel (recommandé)

```bash
python -m venv venv
source venv/bin/activate  # macOS / Linux
# ou
# venv\Scripts\activate   # Windows
```

### 2. Installer les dépendances nécessaires

```bash
pip install tensorflow matplotlib numpy notebook
```

---

## Exécution du TP

### 1. Lancer Jupyter Notebook

Toujours dans l’environnement virtuel activé:

```bash
jupyter notebook
```

Ouvre ensuite le fichier:

- `mnist.ipynb`

### 2. Exécuter les cellules

1. Exécute les cellules dans l’ordre (Kernel → Restart & Run All est conseillé).
2. Vérifie que:
   - Le dataset MNIST se charge correctement.
   - Le modèle CNN est construit et compilé sans erreur.
   - L’entraînement se déroule sur le nombre d’époques prévu.
3. Observe:
   - La **précision sur le jeu de test** (souvent > 0.97 avec ce type de CNN).
   - Les courbes de **loss/accuracy** pour détecter surapprentissage éventuel.
   - Les **prédictions** affichées (chiffre prédit vs label réel).

---

## Résultats attendus

- Une **précision élevée** sur l’ensemble de test (typiquement > 97 %).
- Des courbes d’apprentissage montrant une convergence rapide.
- Des exemples de prédiction où le modèle reconnaît correctement la majorité des chiffres.

Ce TP illustre les bases d’un pipeline de classification d’images:

- Prétraitement
- Conception d’un CNN
- Entraînement, évaluation, visualisation

---

## Préparation pour GitHub

Pour publier ce TP sur GitHub, depuis la racine de ton projet (où se trouve `TP_mnist`) :

```bash
git init
git add TP_mnist
git commit -m "Ajout TP MNIST CNN"
git branch -M main
git remote add origin <URL_DE_TON_DEPOT_GITHUB>
git push -u origin main
```

Pense à ne pas versionner ton éventuel environnement virtuel dans `TP_mnist` en l’ajoutant dans `.gitignore` :

```gitignore
TP_mnist/venv/
```

Avec ce `README.md`, le TP sera correctement documenté et lisible directement sur GitHub.

