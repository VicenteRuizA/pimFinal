import nibabel as nib
import numpy as np
import os
import pydicom
from pydicom.filereader import dcmread

def normalize_volume(volume):
    # Normalize volume to [0,1]
    min_val = np.min(volume)
    max_val = np.max(volume)
    if max_val > min_val:
        return (volume - min_val) / (max_val - min_val)
    else:
        return volume * 0  # all zero volume edge case

def load_mnc_volumes(data_dir):
    volumes = {}
    for filename in os.listdir(data_dir):
        if filename.endswith(".mnc"):
            level = filename.split("_pn")[1].split("_")[0]
            path = os.path.join(data_dir, filename)
            vol = nib.load(path).get_fdata()
            volumes[level] = normalize_volume(vol)
    return volumes

def load_dicom_volume(dicom_dir):
    # Load all DICOM files from a directory, sorted by InstanceNumber or filename
    dicom_files = [os.path.join(dicom_dir, f) for f in os.listdir(dicom_dir) if f.endswith(".dcm")]
    # Sort by InstanceNumber tag if present, else by filename
    def sort_key(f):
        try:
            ds = dcmread(f, stop_before_pixels=True)
            return int(ds.InstanceNumber)
        except Exception:
            return f
    dicom_files.sort(key=sort_key)
    
    slices = [dcmread(f).pixel_array for f in dicom_files]
    volume = np.stack(slices, axis=-1)  # shape: (H, W, num_slices)
    return volume

def load_mr_volumes(base_dir):
    """
    Expects folder structure like:
    base_dir/
        MR/
            noise_level_1/
                DICOM_anon/
                Ground/
            noise_level_3/
                DICOM_anon/
                Ground/
            ...
    Returns dict with keys '0', '1', '3', ... (0 is ground truth).
    """
    volumes = {}
    mr_dir = os.path.join(base_dir, "MR")
    if not os.path.isdir(mr_dir):
        return volumes  # empty if no MR folder found

    for noise_level_folder in os.listdir(mr_dir):
        noise_path = os.path.join(mr_dir, noise_level_folder)
        if not os.path.isdir(noise_path):
            continue

        # Identify noise level from folder name (expecting '0' or '1', '3', etc.)
        noise_level = noise_level_folder

        # Load noisy and clean volumes
        noisy_dir = os.path.join(noise_path, "DICOM_anon")
        clean_dir = os.path.join(noise_path, "Ground")

        if os.path.isdir(noisy_dir) and os.path.isdir(clean_dir):
            noisy_vol = load_dicom_volume(noisy_dir)
            clean_vol = load_dicom_volume(clean_dir)

            # Normalize both
            noisy_vol = normalize_volume(noisy_vol)
            clean_vol = normalize_volume(clean_vol)

            # Store volumes
            volumes[noise_level] = noisy_vol
            volumes['0'] = clean_vol  # always keep clean as key '0' (ground truth)

    return volumes

def load_volumes_combined(base_dir):
    # Load .mnc volumes from base_dir root
    mnc_vols = load_mnc_volumes(base_dir)
    
    # Load MR volumes from Train_Sets/MR or Test_Sets/MR depending on base_dir
    # Assuming base_dir like '.' or './Train_Sets'
    mr_base = os.path.join(base_dir, "Train_Sets") if "Train_Sets" not in base_dir else base_dir
    if not os.path.isdir(mr_base):
        mr_base = base_dir  # fallback

    mr_vols = load_mr_volumes(mr_base)

    # Combine dictionaries, prefer MR volumes if overlap (MR is newer data)
    combined = {**mnc_vols, **mr_vols}

    return combined

def extract_patches(volume, patch_size=(32, 32, 32), stride=16):
    patches = []
    x_max = volume.shape[0] - patch_size[0] + 1
    y_max = volume.shape[1] - patch_size[1] + 1
    z_max = volume.shape[2] - patch_size[2] + 1

    for i in range(0, x_max, stride):
        for j in range(0, y_max, stride):
            for k in range(0, z_max, stride):
                patch = volume[i:i + patch_size[0], j:j + patch_size[1], k:k + patch_size[2]]
                patches.append(patch)
    return patches
