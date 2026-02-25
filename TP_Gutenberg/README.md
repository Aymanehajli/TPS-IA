## TP Gutenberg – Génération de texte de type Shakespeare avec LSTM

Ce TP a pour objectif de construire un modèle de **réseau de neurones récurrent (LSTM)** pour générer du texte dans le style de Shakespeare à partir du corpus du **Project Gutenberg**.  
Le script met l’accent sur une accuracy **réaliste (~70 %) sans surapprentissage**, en corrigeant notamment les problèmes de *data leakage*.

- **Source de données**: `The Complete Works of William Shakespeare` (Project Gutenberg)
- **Tâche**: Modélisation de texte caractère par caractère
- **Objectif pédagogique**:
  - Télécharger et nettoyer un corpus texte brut
  - Construire un dataset caractère par caractère (séquences glissantes)
  - Entraîner un LSTM avec callbacks (early stopping, réduction du LR…)
  - Évaluer la qualité du modèle via des métriques et de la génération de texte

---

## Contenu du dossier

- `gutenberg.py`  
  Script principal qui:
  - Télécharge le corpus (`pg100.txt`) s’il n’existe pas encore (sous le nom `shakespeare.txt`)
  - Nettoie le texte (suppression des entêtes/pieds de Gutenberg, espaces, etc.)
  - Crée un dataset de séquences caractères sans fuite de données entre train/validation
  - Définit et entraîne un modèle LSTM pour la prédiction de caractères
  - Enregistre les graphes d’entraînement et les modèles
  - Génère du texte et effectue plusieurs tests qualitatifs

- `requirements.txt`  
  Liste les dépendances Python nécessaires.

- `reponse.txt`  
  Fichier de compte rendu / réponses liées au TP (analyse des résultats, commentaires, etc.).

Après exécution, le script produit typiquement :

- `training_plots.png` – Graphiques d’entraînement (loss, accuracy, perplexité, overfitting, etc.)
- `best_model.h5` – Meilleur modèle sauvegardé (selon `val_loss`)
- `shakespeare_lstm_corrected.h5` – Modèle final corrigé

---

## Architecture du modèle

Le modèle LSTM utilisé dans `gutenberg.py` est un modèle séquentiel Keras:

- **Embedding**:
  - `Embedding(vocab_size, EMBED_DIM)`
- **LSTM 1**:
  - `LSTM(LSTM_UNITS, return_sequences=True, dropout=0.3, recurrent_dropout=0.2)`
  - Suivi d’une `LayerNormalization`
- **LSTM 2**:
  - `LSTM(LSTM_UNITS // 2, return_sequences=True, dropout=0.3, recurrent_dropout=0.2)`
  - Suivi d’une `LayerNormalization`
- **Sortie**:
  - `Dense(vocab_size)` avec logits (utilisé avec `SparseCategoricalCrossentropy(from_logits=True)`)

Le tout est compilé avec:

- Optimiseur: `Adam(learning_rate=1e-3)`
- Loss: `SparseCategoricalCrossentropy(from_logits=True)`
- Métrique: `accuracy`

Des **callbacks** gèrent l’entraînement:

- `EarlyStopping` (sur `val_loss`, avec `restore_best_weights=True`)
- `ReduceLROnPlateau` (réduction du LR en cas de plateau)
- `ModelCheckpoint` (sauvegarde automatique du meilleur modèle)

---

## Prérequis

- **Python** 3.9+ recommandé
- Gestionnaire de paquets: **pip**

Paquets principaux (déjà listés dans `requirements.txt`) :

- `tensorflow`
- `numpy`
- `matplotlib`
- (éventuellement `urllib3` / dépendances standards pour le téléchargement)

---

## Installation

Depuis le dossier `TP_Gutenberg` :

### 1. Créer et activer un environnement virtuel (recommandé)

```bash
python -m venv venv
source venv/bin/activate  # macOS / Linux
# ou
# venv\Scripts\activate   # Windows
```

### 2. Installer les dépendances

```bash
pip install -r requirements.txt
```

---

## Exécution du TP

Toujours depuis le dossier `TP_Gutenberg` avec l’environnement virtuel activé:

```bash
python gutenberg.py
```

Le script:

1. **Télécharge** le corpus Shakespeare depuis Project Gutenberg (si `shakespeare.txt` n’existe pas).
2. **Nettoie** le texte pour enlever les parties non littéraires (entêtes/pieds Gutenberg).
3. **Construit** deux datasets:
   - Entraînement
   - Validation
   avec un découpage correct (sans overlap entre train et val pour éviter le data leakage).
4. **Entraîne** le modèle LSTM pendant plusieurs époques (avec early stopping).
5. **Affiche et enregistre**:
   - Les courbes de loss et d’accuracy
   - La perplexité
   - L’écart train/val (overfitting gap)
6. **Évalue** la performance:
   - Accuracy validation (objectif ~ 65–75 %)
   - Perplexité réaliste
7. **Génère du texte** avec différentes graines et températures pour juger:
   - La cohérence stylistique
   - La diversité des générations

Les résultats détaillés (accuracy, loss, perplexité, tests de génération) sont affichés dans la console.

---

## Résultats attendus

- **Accuracy de validation**: autour de **65–75 %** (réaliste pour de la génération caractère par caractère).
- **Perplexité**: modérée, reflétant un apprentissage correct sans surapprentissage extrême.
- Générations de texte:
  - Style globalement proche de Shakespeare
  - Diversité raisonnable selon la température choisie

Le script inclut aussi des tests de qualité (plusieurs graines de démarrage, différentes températures, analyse de diversité).

---

## Préparation pour GitHub

Pour publier ce TP sur GitHub, place-toi à la racine de ton projet (où se trouve `TP_Gutenberg`) et exécute, par exemple:

```bash
git init
git add TP_Gutenberg
git commit -m "Ajout TP Gutenberg LSTM texte Shakespeare"
git branch -M main
git remote add origin <URL_DE_TON_DEPOT_GITHUB>
git push -u origin main
```

Pense à ignorer ton environnement virtuel dans `.gitignore` :

```gitignore
TP_Gutenberg/venv/
```

Le fichier `README.md` de `TP_Gutenberg` documente entièrement le TP et s’affichera proprement sur GitHub.

