from neuralnet import build_parallel_riciannet
from dataloader import load_volumes, extract_patches
from skimage.metrics import peak_signal_noise_ratio
import numpy as np
import matplotlib.pyplot as plt
import os

def test_model():
    vols = load_volumes('.')
    noisy = vols['5']
    clean = vols['0']

    x_patches = extract_patches(noisy)
    y_patches = extract_patches(clean)

    x_test = np.array(x_patches)[..., np.newaxis]
    y_test = np.array(y_patches)[..., np.newaxis]

    # Normalize to [0,1] to avoid overflow in PSNR
    max_val = np.max(x_test)
    x_test = x_test / max_val
    y_test = y_test / max_val

    model = build_parallel_riciannet()
    model.load_weights("saved_models/riciannet_lite.h5")

    print("🧪 Evaluating...")
    preds = model.predict(x_test, batch_size=1, verbose=1)
    preds = np.clip(preds, 0, 1)

    total_psnr = 0
    for i in range(len(preds)):
        psnr = peak_signal_noise_ratio(y_test[i].squeeze(), preds[i].squeeze(), data_range=1)
        total_psnr += psnr
    avg_psnr = total_psnr / len(preds)
    print(f"📈 Average PSNR on test set: {avg_psnr:.2f} dB")

    # 🎨 Save visual comparison of 3D patch (center slice)
    os.makedirs("denoised_examples", exist_ok=True)
    for idx in range(3):  # Show first 3 examples
        slice_idx = x_test[idx].shape[2] // 2  # middle slice (axis Z)
        noisy_img = x_test[idx][:, :, slice_idx, 0]
        pred_img = preds[idx][:, :, slice_idx, 0]
        target_img = y_test[idx][:, :, slice_idx, 0]

        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        axs[0].imshow(noisy_img, cmap='gray')
        axs[0].set_title("Noisy Input")
        axs[1].imshow(pred_img, cmap='gray')
        axs[1].set_title("Denoised Output")
        axs[2].imshow(target_img, cmap='gray')
        axs[2].set_title("Ground Truth")

        for ax in axs:
            ax.axis('off')
        plt.tight_layout()
        plt.savefig(f"denoised_examples/patch_{idx}_comparison.png", dpi=150)
        plt.close()
        print(f"🖼️ Saved: denoised_examples/patch_{idx}_comparison.png")
