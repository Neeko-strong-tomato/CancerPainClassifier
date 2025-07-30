# CancerPainClassifier
# Copyright (c) 2025 Neeko
# License: MIT
# If used in research, please cite: https://github.com/Neeko-strong-tomato/CancerPainClassifier

import loader
import logger
import numpy as np

def identity(scan): return 0

PREPOCESSINGS = {
    'identity': identity,
}


def apply_normalization(scan, method="zscore"):
        """
        Applique une normalisation à un volume 3D (PET scan, par exemple).

        Paramètres :
            scan (np.ndarray) : volume 3D (shape typique : [D, H, W])
            method (str) : type de normalisation. Options :
                           - "zscore"
                           - "minmax"
                           - "mean"
                           - "none"

        Retour :
            scan normalisé (np.ndarray)
        """
        scan = scan.astype(np.float32)

        if method == "zscore":
            mean = np.mean(scan)
            std = np.std(scan)
            if std > 0:
                return (scan - mean) / std
            else:
                return scan - mean  # ou raise ValueError
        elif method == "minmax":
            min_val = np.min(scan)
            max_val = np.max(scan)
            if max_val > min_val:
                return (scan - min_val) / (max_val - min_val)
            else:
                return np.zeros_like(scan)
        elif method == "mean":
            mean = np.mean(scan)
            return scan - mean
        elif method == "none":
            return scan
        else:
            raise ValueError(f"Unknown normalization method: {method}")


def normalize_a_scan(labelized_scan, normalization_method=None):
    """
    Args:
        labelized_scan: dict {'scan': np.ndarray, 'label': int}
        normalization_method: str (key from PREPROCESSING)

    Returns:
        Nothing, the scan as been preprocessed and modified with Bohr effect
    """

    scan = labelized_scan['data']

    if normalization_method is not None:
       scan = apply_normalization(scan, normalization_method)


def normalize_all_scans(labelized_scans, normalization_method='zscore'
                         , verbose=True):
    """
    Args:
        labelized_scans: list of dict {'scan': np.ndarray, 'label': int}
        preprocessing_method: str (key from PREPROCESSING)

    Returns:
        Nothing, the scan as been preprocessed and modified with Bohr effect
    """

    if verbose:
        scan_amount = len(labelized_scans)
        scan_index = 0
        ProgressBar = logger.ProgressReporter('Normalization')

    for scan in labelized_scans :

        if verbose:
            scan_index += 1
            ProgressBar.update((scan_index/scan_amount)*100)
        normalize_a_scan(scan, normalization_method=normalization_method)

    if verbose:    
        print("\nTerminé.")


def preprocess_a_scan(labelized_scan, preprocessing_method=None):
    """
    Args:
        labelized_scan: dict {'scan': np.ndarray, 'label': int}
        preprocessing_method: str (key from PREPROCESSING)

    Returns:
        Nothing, the scan as been preprocessed and modified with Bohr effect
    """

    scan = labelized_scan['data']
    
    if preprocessing_method is not None:
        preprocess = PREPOCESSINGS[preprocessing_method]
        preprocess(scan)


def preprocess_all_scans(labelized_scans, preprocessing_method=None, normalization_method=None
                         , verbose=True):
    """
    Args:
        labelized_scans: list of dict {'scan': np.ndarray, 'label': int}
        preprocessing_method: str (key from PREPROCESSING)

    Returns:
        Nothing, the scan as been preprocessed and modified with Bohr effect
    """

    if verbose:
        scan_amount = len(labelized_scans)
        scan_index = 0
        ProgressBar = logger.ProgressReporter('Preprocessing')

    for scan in labelized_scans :

        if verbose:
            scan_index += 1
            ProgressBar.update((scan_index/scan_amount)*100)
        preprocess_a_scan(scan, preprocessing_method=preprocessing_method)

    if verbose:
        print("\nTerminé.")

    if normalization_method is not None:
        normalize_all_scans(labelized_scans, normalization_method=normalization_method, verbose=verbose)