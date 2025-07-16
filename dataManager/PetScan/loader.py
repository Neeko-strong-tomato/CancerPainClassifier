import nibabel as nib
import numpy as np
import os

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

if __name__ == "__main__":
    from dataManager.PatientSelector import PatientSelector
else :
    from dataManager.PatientSelector import PatientSelector

from tqdm import tqdm



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


class PETScanLoader:
    def __init__(self, data_dir, normalization):
        """
        data_dir : path to the .nii ou .nii.gz
        normalize : bool telling if the normalization should be applied (min-max)
        """
        self.data_dir = data_dir
        self.normalize = normalization
        self.selector = PatientSelector()
        self.file_list = [f for f in os.listdir(data_dir) if f.endswith('.nii') or f.endswith('.nii.gz')]

    def load_scan(self, filename):
        path = os.path.join(self.data_dir, filename)
        img = nib.load(path)
        data = img.get_fdata()

        data = apply_normalization(data, self.normalize)

        return data
    
    def load_labelised_scan(self, filename):
        """
        Return a 2 - uplet containing the .nii datas and the label of the patient
        """
        path = os.path.join(self.data_dir, filename)
        img = nib.load(path)
        data = img.get_fdata()

        label = self.selector.get_patient_label(os.path.splitext(filename)[0])

        data = apply_normalization(data, self.normalize)

        return {"data": data, "label": label}


    def load_all(self):
        """
        Put all the scans from thr data directory into a list
        """
        scans = []
        for file in tqdm(self.file_list, desc="Loading PET scans"):
            try:
                scans.append(self.load_scan(file))
                if __name__ == "__main__" : 
                    print(f"File added correcty {file}")
            except Exception as e:
                print(f"Error while loading {file} : {e}")
        return scans
    
    def load_all_labelised(self):
        """
        Put all the scans from thr data directory into a list
        """
        labelizedScans = []
        for file in tqdm(self.file_list, desc="Loading PET scans"):
            try:
                newPatient = self.load_labelised_scan(file)
                if newPatient["label"] != None:
                    labelizedScans.append(newPatient)
                #if __name__ == "__main__" : 
                #    print(f"File added correcty {file}")
            except Exception as e:
                print(f"Error while loading {file} : {e}")
        return labelizedScans


if __name__ == "__main__" :
    
    loader = PETScanLoader(os.path.expanduser("~/Documents/CancerPain/PETdata/data/"), 'zscore')
    scan = loader.load_scan("PHC60_8863.nii")
    print(os.path.splitext("PHC60_8863.nii")[0])
    print(loader.load_labelised_scan("PHC60_8863.nii"))

    labelizedScans = loader.load_all_labelised()

    print("=================================")
    print("shape du scan :", scan.shape)
    print("Nombre de dimensions :", scan.ndim)