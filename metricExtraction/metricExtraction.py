# pipeline_pet_metrics.py
import os
import numpy as np
import pandas as pd
import ants
import nibabel as nib
from scipy.stats import skew, kurtosis

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from dataManager.PetScan.loader import PETScanLoader
import dataManager.PetScan.logger as logger

# ---- Utils ----
def extract_region_metrics(pet_data, region_mask, region_id):
    """Compute metrics for a given region."""
    voxels = pet_data[region_mask == region_id]
    if len(voxels) == 0:
        return {
            f"r{region_id}_mean": np.nan,
            f"r{region_id}_std": np.nan,
            f"r{region_id}_max": np.nan,
            f"r{region_id}_min": np.nan,
            f"r{region_id}_skew": np.nan,
            f"r{region_id}_kurt": np.nan,
        }
    return {
        f"r{region_id}_mean": float(np.mean(voxels)),
        f"r{region_id}_std": float(np.std(voxels)),
        f"r{region_id}_max": float(np.max(voxels)),
        f"r{region_id}_min": float(np.min(voxels)),
        f"r{region_id}_skew": float(skew(voxels)),
        f"r{region_id}_kurt": float(kurtosis(voxels)),
    }

def process_patient(scan_dict, atlas_img):
    """Register PET to atlas and extract metrics."""
    pet_data = scan_dict["data"]
    label = scan_dict["label"]

    pet_ants = ants.from_numpy(pet_data)


    reg = ants.registration(fixed=atlas_img, moving=pet_ants, type_of_transform="Affine")
    pet_in_atlas = ants.apply_transforms(fixed=atlas_img, moving=pet_ants, transformlist=reg["fwdtransforms"])


    atlas_arr = atlas_img.numpy()
    pet_arr = pet_in_atlas.numpy()

    features = {"label": label}
    for rid in np.unique(atlas_arr)[1:]:  
        features.update(extract_region_metrics(pet_arr, atlas_arr, rid))
    
    return features

# ---- Pipeline ----
def run_pipeline(pet_dir, atlas_path, out_csv="metricExtraction/pet_features.csv"):
    loader = PETScanLoader(pet_dir)
    scans = loader.load_all_labelised()

    # Load atlas
    atlas = ants.image_read(atlas_path)

    # Extract metrics
    results = []

    scan_amount = len(scans)
    scan_index = 0

    ProgressBar = logger.ProgressReporter(f'metric extraction')

    print(len(scans))

    for scan in scans:
        scan_index += 1
        ProgressBar.update((scan_index / scan_amount) * 100)

        try:
            feats = process_patient(scan, atlas)
            results.append(feats)
        except Exception as e:
            print(f"Error processing patient: {e}")

    df = pd.DataFrame(results)
    df.to_csv(out_csv, index=False)
    return df

# ---- Example usage ----
if __name__ == "__main__":
    pet_dir = os.path.expanduser("~/Documents/CancerPain/PETdata/data/")
    atlas_path = "metricExtraction/atlas/BN_Atlas_246_3mm.nii" 
    df = run_pipeline(pet_dir, atlas_path)
    print(df.head())
    print(df.shape)

