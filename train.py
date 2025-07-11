from neuralnet import build_parallel_riciannet
from dataloader import load_volumes, extract_patches
import tensorflow as tf
import numpy as np
import os

def train_model():
    print("📦 Loading volumes...")
    vols = load_volumes('.')
    x_train = []
    y_train = []

    for level in ['1', '3', '5', '7', '9']:
        print(f"🔄 Extracting patches for noise level: {level}%")
        noisy = vols[level]
        clean = vols['0']
        x_patches = extract_patches(noisy)
        y_patches = extract_patches(clean)
        x_train.extend(x_patches)
        y_train.extend(y_patches)

    x_train = np.array(x_train)[..., np.newaxis][:200]
    y_train = np.array(y_train)[..., np.newaxis][:200]

    # Normalize using same factor
    max_val = np.max(x_train)
    x_train = x_train / max_val
    y_train = y_train / max_val

    model = build_parallel_riciannet()
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='mse')

    print("🚀 Starting training...")
    model.fit(x_train, y_train, batch_size=1, epochs=10, verbose=1)

    os.makedirs("saved_models", exist_ok=True)
    model.save("saved_models/riciannet_lite.h5")
    print("✅ Model saved to saved_models/riciannet_lite.h5")
