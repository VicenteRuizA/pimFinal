from neuralnet import build_parallel_riciannet
from dataloader import load_volumes, extract_patches
from skimage.metrics import peak_signal_noise_ratio
import numpy as np

def test_model():
    vols = load_volumes('.')
    noisy = vols['5']
    clean = vols['0']

    x_patches = extract_patches(noisy)
    y_patches = extract_patches(clean)

    x_test = np.array(x_patches)[..., np.newaxis]
    y_test = np.array(y_patches)[..., np.newaxis]

    model = build_parallel_riciannet()
    model.load_weights("saved_models/riciannet_lite.h5")

    print("Evaluating...")
    preds = model.predict(x_test)

    total_psnr = 0
    for i in range(len(preds)):
        psnr = peak_signal_noise_ratio(y_test[i].squeeze(), preds[i].squeeze(), data_range=1)
        total_psnr += psnr
    avg_psnr = total_psnr / len(preds)

    print(f"Average PSNR on test set: {avg_psnr:.2f} dB")
