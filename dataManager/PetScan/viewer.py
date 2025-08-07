import matplotlib.pyplot as plt

def show_slice(scan, axis=2, index=None, cmap='gray', title=None):
    """
    Affiche une coupe d’un volume 3D avec colorbar.
    
    Args:
        scan: np.ndarray 3D (volume)
        axis: int (0=axial, 1=coronal, 2=sagittal)
        index: int or None (si None → coupe au centre)
        cmap: str (colormap matplotlib)
        title: str (titre optionnel)
    """
    if scan.ndim != 3:
        raise ValueError("Le scan doit être un volume 3D.")

    # Choix de l’index si pas fourni
    if index is None:
        index = scan.shape[axis] // 2

    # Extraction de la coupe selon l’axe
    if axis == 0:
        slice_ = scan[index, :, :]
        plane = 'Axial'
    elif axis == 1:
        slice_ = scan[:, index, :]
        plane = 'Coronal'
    elif axis == 2:
        slice_ = scan[:, :, index]
        plane = 'Sagittal'
    else:
        raise ValueError("L’axe doit être 0, 1 ou 2.")

    # Affichage
    plt.figure(figsize=(6, 5))
    im = plt.imshow(slice_.T, cmap=cmap, origin='lower')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title(title or f'{plane} slice at index {index}')
    plt.axis('off')
    plt.show()