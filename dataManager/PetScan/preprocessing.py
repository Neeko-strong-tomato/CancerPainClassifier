# CancerPainClassifier
# Copyright (c) 2025 Neeko
# License: MIT
# If used in research, please cite: https://github.com/Neeko-strong-tomato/CancerPainClassifier

if __name__ == "__main__":
    import loader
    import logger

else :
    import dataManager.PetScan.loader
    import dataManager.PetScan.logger as logger
    import dataManager.PetScan.mask as mask

import numpy as np

from nilearn.image import resample_to_img
import nibabel as nib


#########################################
#          Auxiliar functions           #
#########################################

def compute_mean_template(labelized_scans):
    """
    
     Compute a template with the voxel-wises without empiling all the scan on a stack.
    
    Args:
        labelized_scans : list de dicts {'scan': np.ndarray, 'label': int}

    Returns:
        np.ndarray : mean_template
    """
    total = None
    count = 0

    for s in labelized_scans:
        scan = s['data']
        if total is None:
            total = np.zeros_like(scan, dtype=np.float32)
        total += scan
        count += 1

    return total / count


#########################################
#         Preprocessing method          #
#########################################

def identity(scan): return 0

def mean_template_interpolate(scan, mean_template): 
    scan_nii = nib.Nifti1Image(scan, affine=np.eye(4))
    mean_template_nii = nib.Nifti1Image(mean_template, affine=np.eye(4))
    return resample_to_img(scan_nii, mean_template_nii, 
                           interpolation='continuous', 
                           force_resample=True, 
                           copy_header=True
                           ).get_fdata()

def void_bone_mask(scan):
    scan_mask = mask.create_brain_mask(scan, intensity_threshold=0.02)
    return mask.apply_mask(scan, scan_mask)

PREPROCESSINGS = {
    'identity': identity,
    'mean_template': mean_template_interpolate,
    'mask': void_bone_mask,
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
                return scan - mean  # or raise ValueError
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


def preprocess_a_scan(labelized_scan, preprocessing_methods=None, MEAN_TEMPALTE=np.zeros(1)):
    """
    Args:
        labelized_scan: dict {'scan': np.ndarray, 'label': int}
        preprocessing_method: array of str (key from PREPROCESSING)

    Returns:
        Nothing, the scan as been preprocessed and modified with Bohr effect
    """

    scan = labelized_scan['data']
    
    if preprocessing_methods is None:
        return

    # if the array contains only one element
    if isinstance(preprocessing_methods, str):
        preprocessing_methods = [preprocessing_methods]

    # Check if the methods are correct 
    for method in preprocessing_methods:
        if method not in PREPROCESSINGS:
            raise ValueError(f"Unknown preprocessing methods : '{method}'")

    # Apply the preprocessing methods in the correct order
    for method in preprocessing_methods:
        if method == 'mean_template':
            preprocess_fn = PREPROCESSINGS[method]
            preprocess_fn(scan, MEAN_TEMPALTE)
        else:
            preprocess_fn = PREPROCESSINGS[method]
            preprocess_fn(scan)


def preprocess_all_scans(labelized_scans, preprocessing_method=None, normalization_method=None
                         , verbose=True, visualize_operation=False):
    """
    Args:
        labelized_scans: list of dict {'scan': np.ndarray, 'label': int}
        preprocessing_method: array of str (key from PREPROCESSING)
        normalization_method: a str - "zscore"
                                    - "minmax"
                                    - "mean"
                                    - "none"
    Returns:
        Nothing, the scan as been preprocessed and modified with Bohr effect
    """

    if visualize_operation:
        from dataManager.dataAnalyser import interactive_volume_viewer
        interactive_volume_viewer(labelized_scans[0]['data'])

    if "mean_template" in preprocessing_method:
        MEAN_TEMPLATE = compute_mean_template(labelized_scans)
        if "mask" in preprocessing_method:
            MEAN_TEMPLATE_mask = mask.create_brain_mask(MEAN_TEMPLATE, intensity_threshold=0.02)
            MEAN_TEMPLATE = mask.apply_mask(MEAN_TEMPLATE, MEAN_TEMPLATE_mask)
        
        if visualize_operation:
            interactive_volume_viewer(MEAN_TEMPLATE)


    if verbose:
        scan_amount = len(labelized_scans)
        scan_index = 0
        ProgressBar = logger.ProgressReporter('Preprocessing')

    for scan in labelized_scans :

        if verbose:
            scan_index += 1
            ProgressBar.update((scan_index/scan_amount)*100)
        preprocess_a_scan(scan, preprocessing_methods=preprocessing_method, MEAN_TEMPALTE=MEAN_TEMPLATE)
    
    if visualize_operation:
        interactive_volume_viewer(labelized_scans[0]['data'])

    if verbose:
        print("\nTerminé.")

    if normalization_method is not None:
        normalize_all_scans(labelized_scans, normalization_method=normalization_method, verbose=verbose)
        if visualize_operation:
            interactive_volume_viewer(labelized_scans[0]['data'])