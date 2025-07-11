from neuralnet import build_parallel_riciannet
from dataloader import load_volumes_combined, extract_patches
import tensorflow as tf
import numpy as np
import os

def train_model():
    print("📦 Loading volumes (combined MNC + MR)...")
    vols = load_volumes_combined('.')

    x_train = []
    y_train = []

    # Use all noise levels except '0' which is ground truth
    noise_levels = [k for k in vols.keys() if k != '0']
    print(f"Using noise levels: {noise_levels}")

    for level in noise_levels:
        print(f"🔄 Extracting patches for noise level: {level}%")
        noisy = vols[level]
        clean = vols['0']
        x_patches = extract_patches(noisy)
        y_patches = extract_patches(clean)
        x_train.extend(x_patches)
        y_train.extend(y_patches)

    # Limit number of samples to reduce memory use
    x_train = np.array(x_train)[..., np.newaxis][:200]
    y_train = np.array(y_train)[..., np.newaxis][:200]

    model = build_parallel_riciannet()
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='mse')

    print("🚀 Starting training...")
    model.fit(x_train, y_train, batch_size=1, epochs=10, verbose=1)

    os.makedirs("saved_models", exist_ok=True)
    model.save("saved_models/riciannet_lite.h5")
    print("✅ Model saved to saved_models/riciannet_lite.h5")
