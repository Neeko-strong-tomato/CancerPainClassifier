# CancerPainClassifier
# Copyright (c) 2025 Neeko
# License: MIT
# If used in research, please cite: https://github.com/Neeko-strong-tomato/CancerPainClassifier

import numpy as np
import itertools
import scipy
import logger

def flip_x(scan): return np.flip(scan, axis=0)
def flip_y(scan): return np.flip(scan, axis=1)
def flip_z(scan): return np.flip(scan, axis=2)
def add_noise(scan, noise_level=0.01): return scan + noise_level * np.random.randn(*scan.shape)
def adjust_contrast(scan, factor=1.2): return scan * factor
def adjust_brightness(scan, offset=0.1): return scan + offset
def blur(scan, sigma=1.0): return scipy.ndimage.gaussian_filter(scan, sigma=sigma)
def identity(scan): return scan

AUGMENTATIONS = {
    'flip_x': flip_x,
    'flip_y': flip_y,
    'flip_z': flip_z,
    'noise': add_noise,
    'adjust_contrast': adjust_contrast,
    'adjust_brightness': adjust_brightness,
    'blur': blur,
}


TARGET_SHAPE = (79, 95, 78)

def fix_shape(scan):
    """
    Recadre (crop ou pad) le scan à la forme cible TARGET_SHAPE.
    Supporte les entrées avec ou sans dimensions supplémentaires.
    """
    scan = np.squeeze(scan)  # supprime les dimensions 1 inutiles

    if scan.ndim != 3:
        raise ValueError(f"Le scan doit être 3D après squeeze, reçu shape={scan.shape}")

    fixed = np.zeros(TARGET_SHAPE, dtype=scan.dtype)

    slices_src = tuple(slice(0, min(s, t)) for s, t in zip(scan.shape, TARGET_SHAPE))
    slices_dst = tuple(slice(0, min(s, t)) for s, t in zip(scan.shape, TARGET_SHAPE))

    fixed[slices_dst] = scan[slices_src]
    return fixed


def augment_a_scan(labelized_scan, selected_augmentations=None, keep_original=True, max_combination_size=2):
    """
    Args:
        labelized_scan: dict {'scan': np.ndarray, 'label': int}
        selected_augmentations: list of str (keys from AUGMENTATIONS), or None to use all
        keep_original: bool, include original in output
        max_combination_size: int, how many augmentations to combine at max

    Returns:
        list of dicts with 'scan' and 'label' keys
    """
    scan = labelized_scan['data']
    label = labelized_scan['label']
    
    if selected_augmentations is None:
        selected_augmentations = list(AUGMENTATIONS.keys())

    output = []

    if keep_original:
        output.append({'data': scan.copy(), 'label': label})

    # Combinations of augmentations
    for k in range(1, max_combination_size + 1):
        for combo in itertools.combinations(selected_augmentations, k):
            augmented = scan.copy()
            for aug in combo:
                augmented = AUGMENTATIONS[aug](augmented)
            augmented = fix_shape(augmented)
            output.append({'data': augmented, 'label': label})

    return output


def augmentate_batch(batch, selected_augmentations=None, keep_original=True, max_combination_size=4):

    enlarged_batch = []

    for i in range (len(batch)):
        augmentated_scan = augment_a_scan(batch[i], selected_augmentations, keep_original, max_combination_size)
        for scan in augmentated_scan :
            enlarged_batch.append(scan)
        
    return enlarged_batch

def augmentate_dataset_separated(X, Y, selected_augmentations=None, keep_original=True, max_combination_size=4):
    """
    Args:
        X: list or array of np.ndarray (scans)
        Y: list or array of labels (int)
        selected_augmentations: list of str (from AUGMENTATIONS), or None to use all
        keep_original: bool, whether to keep the original scan
        max_combination_size: int, how many augmentations to combine at max

    Returns:
        X_aug: array of np.ndarray (augmented scans)
        Y_aug: array of int (corresponding labels)
    """
    X_aug = []
    Y_aug = []

    scan_amount = len(X)
    scan_index = 0

    ProgressBar = logger.ProgressReporter('Augmentation')

    for scan, label in zip(X, Y):

        scan_index += 1
        ProgressBar.update((scan_index/scan_amount)*100)

        labelized_scan = {'data': scan, 'label': label}
        augmented_samples = augment_a_scan(labelized_scan, selected_augmentations, keep_original, max_combination_size)

        for sample in augmented_samples:
            scan_data = sample['data']

            scan_data = fix_shape(scan_data)

            if scan_data.ndim == 3:
                scan_data = np.expand_dims(scan_data, axis=0)

            assert scan_data.shape == (1, *TARGET_SHAPE), f"Scan shape mismatch: {scan_data.shape}"

            X_aug.append(scan_data)
            Y_aug.append(sample['label'])
        
    print("\nTerminé.")

    return np.stack(X_aug, axis=0), np.array(Y_aug)