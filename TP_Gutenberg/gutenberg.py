import os
import re
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
 
# Reproductibilité
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
 
# Dataset
GUTENBERG_URL = "https://www.gutenberg.org/cache/epub/100/pg100.txt"
DATA_PATH = "shakespeare.txt"
 
# Hyperparamètres réalistes pour 70% accuracy
SEQ_LEN = 100  # Séquence raisonnable
BATCH_SIZE = 64  # Batch size réduit pour mieux généraliser
BUFFER_SIZE = 10_000
EMBED_DIM = 256
LSTM_UNITS = 512
EPOCHS = 25
LEARNING_RATE = 1e-3
VALIDATION_SPLIT = 0.15  # Plus de données de validation
 
 
def download_gutenberg(url, path):
    if os.path.exists(path):
        return
    import urllib.request
    print("Téléchargement du corpus...")
    urllib.request.urlretrieve(url, path)
    print("Fichier téléchargé:", path)
 
#  etape 3
def clean_gutenberg_text(text):
    """Nettoyage strict du texte"""
    start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK"
    end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK"
    start_idx = text.find(start_marker)
    end_idx = text.find(end_marker)
    if start_idx != -1 and end_idx != -1:
        text = text[start_idx:end_idx]
   
    # Nettoyage basique uniquement
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
   
    return text
 
#  etape 4
 
def build_char_dataset_correct(text, validation_split=0.15):
    """
    CORRECTION MAJEURE: Découpage séquence-par-séquence correct
    Évite le data leakage qui causait 99% accuracy
    """
    vocab = sorted(set(text))
    char2idx = {u: i for i, u in enumerate(vocab)}
    idx2char = np.array(vocab)
   
    text_as_int = np.array([char2idx[c] for c in text], dtype=np.int32)
   
    # CORRECTION: Split AVANT la création des séquences
    split_idx = int(len(text_as_int) * (1 - validation_split))
    train_text = text_as_int[:split_idx]
    val_text = text_as_int[split_idx:]
   
    print(f"✓ Train: {len(train_text):,} caractères")
    print(f"✓ Val: {len(val_text):,} caractères")
   
    def create_dataset(text_data, shuffle=True):
        """Création correcte du dataset sans overlap entre train/val"""
        # Créer des séquences non-overlapping
        examples_per_epoch = len(text_data) // (SEQ_LEN + 1)
       
        # Tronquer pour avoir des séquences complètes
        char_dataset = tf.data.Dataset.from_tensor_slices(
            text_data[:examples_per_epoch * (SEQ_LEN + 1)]
        )
       
        # Créer les séquences
        sequences = char_dataset.batch(SEQ_LEN + 1, drop_remainder=True)
       
        def split_input_target(chunk):
            input_text = chunk[:-1]
            target_text = chunk[1:]
            return input_text, target_text
       
        dataset = sequences.map(split_input_target)
       
        if shuffle:
            dataset = dataset.shuffle(BUFFER_SIZE)
       
        dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
       
        return dataset
   
    train_dataset = create_dataset(train_text, shuffle=True)
    val_dataset = create_dataset(val_text, shuffle=False)
   
    return train_dataset, val_dataset, vocab, char2idx, idx2char
 
#  etape 5
def build_realistic_model(vocab_size):
    """
    Modèle réaliste pour 70% accuracy
    Moins complexe pour éviter overfitting
    """
    model = tf.keras.Sequential([
        # Embedding
        tf.keras.layers.Embedding(vocab_size, EMBED_DIM),
       
        # LSTM 1
        tf.keras.layers.LSTM(
            LSTM_UNITS,
            return_sequences=True,
            dropout=0.3,
            recurrent_dropout=0.2
        ),
        tf.keras.layers.LayerNormalization(),
       
        # LSTM 2
        tf.keras.layers.LSTM(
            LSTM_UNITS // 2,
            return_sequences=True,
            dropout=0.3,
            recurrent_dropout=0.2
        ),
        tf.keras.layers.LayerNormalization(),
       
        # Dense
        tf.keras.layers.Dense(vocab_size)
    ])
   
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
   
    return model
#  etape 6
 
def plot_training_history(history, save_path='training_plots.png'):
    """Graphiques réalistes"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Entraînement Modèle LSTM - Shakespeare (Accuracy Réaliste)',
                 fontsize=14, fontweight='bold')
   
    # Loss
    axes[0, 0].plot(history.history['loss'], label='Train', linewidth=2, color='#3498db')
    axes[0, 0].plot(history.history['val_loss'], label='Validation', linewidth=2, color='#e74c3c')
    axes[0, 0].set_title('Loss', fontweight='bold')
    axes[0, 0].set_xlabel('Époque')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
   
    # Accuracy
    axes[0, 1].plot(history.history['accuracy'], label='Train', linewidth=2, color='#2ecc71')
    axes[0, 1].plot(history.history['val_accuracy'], label='Validation', linewidth=2, color='#f39c12')
    axes[0, 1].axhline(y=0.70, color='red', linestyle='--', linewidth=1.5, label='Objectif 70%')
    axes[0, 1].set_title('Accuracy (Objectif: ~70%)', fontweight='bold')
    axes[0, 1].set_xlabel('Époque')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_ylim([0.3, 0.8])  # Plage réaliste
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
   
    # Perplexity
    train_perp = np.exp(np.array(history.history['loss']))
    val_perp = np.exp(np.array(history.history['val_loss']))
    axes[1, 0].plot(train_perp, label='Train', linewidth=2, color='#1abc9c')
    axes[1, 0].plot(val_perp, label='Validation', linewidth=2, color='#e74c3c')
    axes[1, 0].set_title('Perplexité (plus bas = meilleur)', fontweight='bold')
    axes[1, 0].set_xlabel('Époque')
    axes[1, 0].set_ylabel('Perplexité')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
   
    # Overfitting Gap
    gap = np.array(history.history['val_loss']) - np.array(history.history['loss'])
    axes[1, 1].plot(gap, linewidth=2, color='#e74c3c')
    axes[1, 1].axhline(y=0, color='green', linestyle='--', linewidth=1)
    axes[1, 1].axhline(y=0.3, color='orange', linestyle='--', linewidth=1, label='Limite acceptable')
    axes[1, 1].fill_between(range(len(gap)), gap, 0, where=(gap > 0), alpha=0.3, color='red')
    axes[1, 1].set_title('Overfitting Gap', fontweight='bold')
    axes[1, 1].set_xlabel('Époque')
    axes[1, 1].set_ylabel('Val Loss - Train Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
   
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Graphiques: {save_path}")
    plt.close()
 
#  etape 7

def generate_text(model, start_string, char2idx, idx2char, num_generate=500, temperature=1.0):
    """Génération de texte"""
    input_eval = [char2idx.get(c, 0) for c in start_string]
    input_eval = tf.expand_dims(input_eval, 0)
    text_generated = []
 
    for _ in range(num_generate):
        predictions = model(input_eval, training=False)
        predictions = predictions[:, -1, :] / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()
        input_eval = tf.expand_dims([predicted_id], 0)
        text_generated.append(idx2char[predicted_id])
 
    return start_string + "".join(text_generated)
 
#  etape 8

def test_model_quality(model, char2idx, idx2char):
    """
    Fonction de test complète du modèle
    Teste la cohérence et la qualité de génération
    """
    print("\n" + "="*70)
    print("TESTS DE QUALITÉ DU MODÈLE")
    print("="*70)
   
    test_seeds = [
        "To be, or not to be",
        "ROMEO: ",
        "What is love? ",
        "The king ",
        "O, ",
        "My lord, ",
        "Fair ",
        "Thou ",
    ]
   
    temperatures = [0.3, 0.7, 1.0]
   
    print("\n" + "─"*70)
    print("TEST 1: GÉNÉRATION AVEC DIFFÉRENTES GRAINES")
    print("─"*70)
   
    for seed in test_seeds[:4]:  # Teste les 4 premières graines
        print(f"\n{'='*70}")
        print(f"🌱 Graine: '{seed}'")
        print('='*70)
       
        for temp in temperatures:
            print(f"\n[Température: {temp}] (0.3=conservateur, 0.7=équilibré, 1.0=créatif)")
            print("─"*70)
            try:
                generated = generate_text(model, seed, char2idx, idx2char,
                                        num_generate=300, temperature=temp)
                print(generated)
            except Exception as e:
                print(f"❌ Erreur: {e}")
   
    print("\n" + "─"*70)
    print("TEST 2: COHÉRENCE DU STYLE SHAKESPEARE")
    print("─"*70)
   
    # Test avec température optimale (0.7)
    shakespeare_seeds = [
        "Wherefore art thou",
        "Friends, Romans, countrymen",
        "All the world's a stage",
    ]
   
    for seed in shakespeare_seeds:
        print(f"\n🎭 Test: '{seed}'")
        print("─"*70)
        generated = generate_text(model, seed, char2idx, idx2char,
                                 num_generate=200, temperature=0.7)
        print(generated)
   
    print("\n" + "─"*70)
    print("TEST 3: ANALYSE DE LA DIVERSITÉ")
    print("─"*70)
   
    # Génère plusieurs fois avec même graine pour tester diversité
    seed = "The "
    generations = []
   
    for i in range(3):
        gen = generate_text(model, seed, char2idx, idx2char,
                          num_generate=100, temperature=0.8)
        generations.append(gen)
        print(f"\nGénération {i+1}:")
        print(gen)
   
    # Vérifie que les générations sont différentes
    unique_gens = set(generations)
    diversity_score = len(unique_gens) / len(generations) * 100
   
    print(f"\n📊 Score de diversité: {diversity_score:.1f}% "
          f"({len(unique_gens)}/{len(generations)} générations uniques)")
   
    if diversity_score == 100:
        print("✅ Excellent: Le modèle génère du contenu varié")
    elif diversity_score > 50:
        print("✅ Bon: Diversité acceptable")
    else:
        print("⚠️  Attention: Le modèle manque de diversité")
   
    print("\n" + "="*70)
    print("FIN DES TESTS")
    print("="*70)
 
#  etape 9
def main():
    print("="*70)
    print("MODÈLE LSTM CORRIGÉ - ACCURACY RÉALISTE (~70%)")
    print("Correction du data leakage et overfitting")
    print("="*70)
   
    download_gutenberg(GUTENBERG_URL, DATA_PATH)
   
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        raw_text = f.read()
   
    text = clean_gutenberg_text(raw_text)
    print(f"\n✓ Corpus total: {len(text):,} caractères")
   
    # CORRECTION: Dataset avec split correct
    train_dataset, val_dataset, vocab, char2idx, idx2char = build_char_dataset_correct(
        text, VALIDATION_SPLIT
    )
    vocab_size = len(vocab)
    print(f"✓ Vocabulaire: {vocab_size} caractères uniques")
   
    model = build_realistic_model(vocab_size)
    print("\n" + "="*70)
    print("ARCHITECTURE (Réaliste)")
    print("="*70)
    model.summary()
   
    # Callbacks réalistes
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'best_model.h5',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]
   
    # Entraînement
    print("\n" + "="*70)
    print("ENTRAÎNEMENT")
    print("="*70)
   
    start_time = datetime.now()
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
   
    training_time = (datetime.now() - start_time).total_seconds()
    print(f"\n✓ Temps: {training_time/60:.2f} minutes")
   
    # Visualisations
    print("\n" + "="*70)
    print("VISUALISATIONS")
    print("="*70)
    plot_training_history(history)
   
    # Rapport
    print("\n" + "="*70)
    print("RÉSULTATS FINAUX")
    print("="*70)
   
    best_val_acc = max(history.history['val_accuracy']) * 100
    final_val_acc = history.history['val_accuracy'][-1] * 100
    final_train_acc = history.history['accuracy'][-1] * 100
    final_val_loss = history.history['val_loss'][-1]
    final_train_loss = history.history['loss'][-1]
   
    print(f"📊 ACCURACY (Génération caractère par caractère):")
    print(f"  - Train: {final_train_acc:.2f}%")
    print(f"  - Validation (finale): {final_val_acc:.2f}%")
    print(f"  - Validation (meilleure): {best_val_acc:.2f}%")
   
    print(f"\n📉 LOSS:")
    print(f"  - Train: {final_train_loss:.4f}")
    print(f"  - Validation: {final_val_loss:.4f}")
    print(f"  - Gap: {final_val_loss - final_train_loss:.4f}")
   
    perplexity = np.exp(final_val_loss)
    print(f"\n🎯 PERPLEXITÉ: {perplexity:.2f}")
   
    # Évaluation
    print(f"\n{'='*70}")
    if 65 <= best_val_acc <= 75:
        print("✅ EXCELLENT! Accuracy réaliste et saine (65-75%)")
        print("   Le modèle a appris sans surapprentissage")
    elif 60 <= best_val_acc < 65:
        print("✅ BON! Performance satisfaisante (60-65%)")
    elif best_val_acc >= 90:
        print("❌ ATTENTION! Accuracy anormalement haute (>90%)")
        print("   Probable data leakage ou overfitting sévère")
    else:
        print("⚠️  Performance à améliorer (<60%)")
    print(f"{'='*70}")
   
    # TESTS DE QUALITÉ
    test_model_quality(model, char2idx, idx2char)
   
    # Sauvegarde
    model.save("shakespeare_lstm_corrected.h5")
    print("\n" + "="*70)
    print("✓ Modèle sauvegardé: shakespeare_lstm_corrected.h5")
    print("✓ Meilleur modèle: best_model.h5")
    print("="*70)
 
 
if __name__ == "__main__":
    main()
 