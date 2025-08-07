import numpy as np
import scipy.ndimage as ndi
from skimage.morphology import ball, binary_closing

from dataManager.PetScan.viewer import show_slice

def create_brain_mask(scan, intensity_threshold=0.02, closing_radius=2):
    """
    Crée un masque binaire du cerveau à partir d’un scan 3D (PET ou autre).

    Args:
        scan: np.ndarray 3D
        intensity_threshold: float (valeur relative à max, ex: 0.02 = 2%)
        closing_radius: int, rayon pour remplir les trous dans le masque

    Returns:
        mask: np.ndarray binaire (True = à garder)
    """
    # Seuillage basique
    thresh = intensity_threshold * np.max(scan)
    binary = scan > thresh

    # Connexité : garder la + grosse région
    labeled, num_features = ndi.label(binary)
    sizes = ndi.sum(binary, labeled, range(1, num_features + 1))

    if num_features == 0:
        raise RuntimeError("Aucune région trouvée, ajuster le seuil")

    largest_label = np.argmax(sizes) + 1
    mask = labeled == largest_label

    # Fermeture morphologique pour remplir les trous
    if closing_radius > 0:
        mask = binary_closing(mask, footprint=ball(closing_radius))

    return mask.astype(np.uint8)


def apply_mask(scan, mask):
    """
    Applique un masque binaire à un scan 3D.
    
    Args:
        scan: np.ndarray 3D
        mask: np.ndarray binaire même shape

    Returns:
        masked scan (np.ndarray)
    """
    return scan * mask


if __name__ == "__main__":
# Chargement de ton scan (3D numpy array)
    import os
    from loader import PETScanLoader

    Loader = PETScanLoader(os.path.expanduser("~/Documents/CancerPain/PETdata/data/"))
    scan = Loader.load_scan("PHC60_8863.nii")

    # Création du masque et visualisation
    mask = create_brain_mask(scan, intensity_threshold=0.02)

    # Visualisation
    show_slice(scan, title="Original")
    show_slice(mask, title="Mask")
    show_slice(apply_mask(scan, mask), title="Masked Scan")
