import nibabel as nib
import numpy as np
import os

def load_volumes(data_dir):
    volumes = {}
    for filename in os.listdir(data_dir):
        if filename.endswith(".mnc"):
            level = filename.split("_pn")[1].split("_")[0]
            path = os.path.join(data_dir, filename)
            volumes[level] = nib.load(path).get_fdata()
    return volumes

def extract_patches(volume, patch_size=(32,32,32), stride=16):
    patches = []
    for i in range(0, volume.shape[0] - patch_size[0], stride):
        for j in range(0, volume.shape[1] - patch_size[1], stride):
            for k in range(0, volume.shape[2] - patch_size[2], stride):
                patch = volume[i:i+32, j:j+32, k:k+32]
                patches.append(patch)
    return patches
