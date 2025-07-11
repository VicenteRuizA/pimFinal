from neuralnet import build_parallel_riciannet
from dataloader import load_volumes, extract_patches
import tensorflow as tf
import numpy as np
import os

def train_model():
    print("Loading volumes...")
    vols = load_volumes('.')
    x_train = []
    y_train = []

    for level in ['1', '3', '5', '7', '9']:
        noisy = vols[level]
        clean = vols['0']
        x_patches = extract_patches(noisy)
        y_patches = extract_patches(clean)
        x_train.extend(x_patches)
        y_train.extend(y_patches)

    x_train = np.array(x_train)[..., np.newaxis]
    y_train = np.array(y_train)[..., np.newaxis]

    model = build_parallel_riciannet()
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='mse')
    print("Training model...")
    model.fit(x_train, y_train, batch_size=4, epochs=10, validation_split=0.1)

    os.makedirs("saved_models", exist_ok=True)
    model.save("saved_models/riciannet_lite.h5")
    print("Model saved to saved_models/riciannet_lite.h5")
