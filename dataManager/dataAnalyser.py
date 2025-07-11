import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import dataLoaders.PETScanLoader as Loader
import dataLoaders.PetScanEnlarger as Enlarger


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

from nilearn import plotting
import nibabel as nib

# =======================
# 1. Prétraitement
# =======================

def prepare_data(data_entries):
    X = np.array([entry["data"].flatten() for entry in data_entries])
    y = np.array([entry["label"] for entry in data_entries])
    return X, y


# =======================
# 2. Sélection de features
# =======================

def select_features(X, y, k=100):
    selector = SelectKBest(score_func=f_classif, k=k)
    X_selected = selector.fit_transform(X, y)
    return X_selected, selector


# =======================
# 3. Visualisation PCA
# =======================

def plot_pca(X, y):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette="Set2")
    plt.title("Projection PCA (2D)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True)
    plt.show()


# =======================
# 4. Entraînement modèle
# =======================

def train_and_evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    
    print("📈 Résultats de classification :")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    
    return clf


# =======================
# 5. Pipeline globale
# =======================

def full_pipeline(data_entries, k_best=100):
    X, y = prepare_data(data_entries)
    X_selected, selector = select_features(X, y, k=k_best)
    plot_pca(X_selected, y)
    clf = train_and_evaluate(X_selected, y)
    return clf, selector


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd


# ============================================
# Projections
# ============================================
def project_pca(X, n_components=2):
    if X.ndim > 2:
        X = X.reshape(X.shape[0], -1)

    pca = PCA(n_components=n_components, random_state=42)
    return pca.fit_transform(X), "PCA", pca

def project_tsne(X, n_components=2, perplexity=30):
    if X.ndim > 2:
        X = X.reshape(X.shape[0], -1)

    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
    return tsne.fit_transform(X), "t-SNE"

def project_umap(X, n_components=2, n_neighbors=15):
    if X.ndim > 2:
        X = X.reshape(X.shape[0], -1)
        
    reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, random_state=42)
    return reducer.fit_transform(X), "UMAP"

# ============================================
# Visualisation
# ============================================
def plot_projection(X_proj, y, title="Projection"):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_proj[:, 0], y=X_proj[:, 1], hue=y, palette="Set2")
    plt.title(f"{title} (2D)")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# ============================================
# Classification
# ============================================
def classify_projection(X_proj, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X_proj, y, test_size=0.2, stratify=y, random_state=42
    )
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    return clf


def make_batch(data):
        X = []
        Y = []

        for patient in data:
            X.append(patient["data"])
            Y.append(patient["label"])

        X = np.array(X, dtype=np.float32)
        Y = np.array(Y, dtype=np.int64)


        if len(X.shape) == 4:
            X = np.expand_dims(X, axis=1)  # (N, D, H, W) -> (N, 1, D, H, W)

        return X, Y


# ============================================
# PET Scan visulisation
# ============================================

def contrast_amplify(volume, lower_pct=80, upper_pct=99, low_scale=0.5, high_scale=2.0):
    # Seuils basés sur les percentiles
    threshold_low = np.percentile(np.abs(volume), lower_pct)
    threshold_high = np.percentile(np.abs(volume), upper_pct)
    
    # Applique les échelles selon l’amplitude
    mask_low = np.abs(volume) < threshold_low
    mask_high = np.abs(volume) > threshold_high
    
    result = volume.copy()
    result[mask_low] *= low_scale
    result[mask_high] *= high_scale
    return result


def show_pca_component(pca_component, shape=(79, 95, 78), cut_coords=(0, 0, 0) ,threshold=0.001):
    """
    Affiche une composante PCA reshaped en image 3D avec mise à l'échelle.
    """
    volume = pca_component.reshape(shape)
    
    # Normalisation z-score (optionnelle mais améliore visuellement)
    volume = (volume - volume.mean()) / (volume.std() + 1e-5)

    # Amplifie le contraste
    volume = contrast_amplify(volume, lower_pct=80, upper_pct=99, low_scale=0.5, high_scale=2.0)
    

    img = nib.Nifti1Image(volume, affine=np.eye(4))
    
    plotting.plot_stat_map(
        img,
        display_mode='ortho',
        threshold=np.percentile(np.abs(volume), 95),  # top 5%
        colorbar=True,
        cmap='coolwarm',
        cut_coords=cut_coords
        )
    plotting.show()


import ipywidgets as widgets
from IPython.display import display

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def interactive_volume_viewer(volume):
    """
    volume: numpy array 3D (shape: x, y, z)
    """

    assert volume.ndim == 3, "Volume must be 3D"
    x_max, y_max, z_max = volume.shape

    # Initial slice
    init_x, init_y, init_z = x_max // 2, y_max // 2, z_max // 2

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    plt.subplots_adjust(bottom=0.25)

    # Show initial views
    ax0 = axs[0].imshow(volume[init_x, :, :], cmap='coolwarm')
    axs[0].set_title('Sagittal (X)')
    ax1 = axs[1].imshow(volume[:, init_y, :], cmap='coolwarm')
    axs[1].set_title('Coronal (Y)')
    ax2 = axs[2].imshow(volume[:, :, init_z], cmap='coolwarm')
    axs[2].set_title('Axial (Z)')

    # Sliders
    axcolor = 'lightgoldenrodyellow'
    ax_slider_x = plt.axes([0.15, 0.15, 0.65, 0.03], facecolor=axcolor)
    ax_slider_y = plt.axes([0.15, 0.1, 0.65, 0.03], facecolor=axcolor)
    ax_slider_z = plt.axes([0.15, 0.05, 0.65, 0.03], facecolor=axcolor)

    slider_x = Slider(ax_slider_x, 'X', 0, x_max - 1, valinit=init_x, valstep=1)
    slider_y = Slider(ax_slider_y, 'Y', 0, y_max - 1, valinit=init_y, valstep=1)
    slider_z = Slider(ax_slider_z, 'Z', 0, z_max - 1, valinit=init_z, valstep=1)

    def update(val):
        x = int(slider_x.val)
        y = int(slider_y.val)
        z = int(slider_z.val)

        ax0.set_data(volume[x, :, :])
        ax1.set_data(volume[:, y, :])
        ax2.set_data(volume[:, :, z])
        fig.canvas.draw_idle()

    slider_x.on_changed(update)
    slider_y.on_changed(update)
    slider_z.on_changed(update)

    plt.show()


import seaborn as sns


def plot_component_distributions(X_proj, y, max_components=30):
    """
    Affiche les boxplots des coefficients des composantes principales selon les classes.
    
    Args:
        X_proj: Données projetées par PCA (n_patients x n_components)
        y: étiquettes (classes des patients)
        max_components: nombre de composantes à afficher (troncature)
    """
    n_components = min(X_proj.shape[1], max_components)
    df = pd.DataFrame(X_proj[:, :n_components], columns=[f"PC{i+1}" for i in range(n_components)])
    df["label"] = y

    melted = df.melt(id_vars="label", var_name="Component", value_name="Coefficient")

    plt.figure(figsize=(16, 6))
    sns.boxplot(data=melted, x="Component", y="Coefficient", hue="label")
    plt.title(f"Distribution des {n_components} premières composantes PCA par classe")
    plt.xticks(rotation=90)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


from scipy.stats import f_oneway  # Pour ANOVA

def plot_pca_components_with_anova(X_proj, y, batch_size=16, cols=4):
    """
    Affiche les composantes PCA (boxplots par classe avec p-values ANOVA), 16 par figure.
    """
    n_components = X_proj.shape[1]
    df = pd.DataFrame(X_proj, columns=[f"PC{i+1}" for i in range(n_components)])
    df["label"] = y

    for start in range(0, n_components, batch_size):
        end = min(start + batch_size, n_components)
        current_batch = range(start, end)
        rows = (len(current_batch) + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows), squeeze=False)
        fig.suptitle(f"Composantes PCA {start+1} à {end} avec p-values ANOVA", fontsize=16)

        for idx, i in enumerate(current_batch):
            ax = axes[idx // cols][idx % cols]
            comp_name = f"PC{i+1}"

            # Boxplot
            sns.boxplot(data=df, x="label", y=comp_name, ax=ax)

            # Calcul p-value ANOVA
            groups = [df[df["label"] == label][comp_name] for label in df["label"].unique()]
            try:
                f_val, p_val = f_oneway(*groups)
                p_str = f"p = {p_val:.3e}"
            except Exception:
                p_str = "p = err"

            ax.set_title(f"{comp_name} ({p_str})")
            ax.set_xlabel("Classe")
            ax.set_ylabel("Coefficient")

        # Cacher les axes inutilisés
        for j in range(len(current_batch), rows * cols):
            fig.delaxes(axes[j // cols][j % cols])

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()


def save_pca_components_as_nii(pca, shape, output_dir="components", affine=np.eye(4)):
    """
    Sauvegarde les composantes PCA au format NIfTI (.nii).
    
    Args:
        pca: objet PCA entraîné (de sklearn.decomposition.PCA)
        shape: tuple 3D de la forme d'origine des volumes (ex: (79, 95, 78))
        output_dir: dossier où stocker les fichiers NIfTI
        affine: matrice affine à utiliser pour les fichiers NIfTI (par défaut identité)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for i, component in enumerate(pca.components_):
        volume = component.reshape(shape)
        nii_img = nib.Nifti1Image(volume, affine)
        nib.save(nii_img, os.path.join(output_dir, f"pca_component_{i+1}.nii"))

    print(f"✅ {len(pca.components_)} composantes sauvegardées dans {output_dir}")

# ============================================
# Pipeline principale
# ============================================
def full_analysis_pipeline(data):
    
    for projector in [project_pca, project_tsne, project_umap]:
        X, y = make_batch(data)
        X_proj, name = projector(X)
        print(f"\n📌 Méthode : {name}")
        plot_projection(X_proj, y, title=name)
        print("📈 Résultats de classification :")
        classify_projection(X_proj, y)


def plot_pca_performance(results):
    """
    Affiche l'évolution des métriques de classification en fonction du nombre de composantes PCA.

    Paramètres :
    - results : list de dicts, chaque dict doit avoir :
        {
            "components": int,
            "accuracy": float,
            "macro_f1": float,
            "macro_precision": float,
            "macro_recall": float,
            "percentage_of_keeped_info": float,
        }
    """
    components = [r["components"] for r in results]
    accuracy = [r["accuracy"] for r in results]
    f1 = [r["macro_f1"] for r in results]
    precision = [r["macro_precision"] for r in results]
    recall = [r["macro_recall"] for r in results]
    keeped_info = [r["percentage_of_keeped_info"] for r in results]

    plt.figure(figsize=(10, 6))
    plt.plot(components, accuracy, label="Accuracy", marker='o')
    plt.plot(components, f1, label="Macro F1-score", marker='s')
    plt.plot(components, precision, label="Macro Precision", marker='^')
    plt.plot(components, recall, label="Macro Recall", marker='x')
    plt.plot(components, keeped_info, label="percentage_of_keeped_info", marker='*')

    plt.title("📊 Évolution des performances en fonction du nombre de composantes PCA")
    plt.xlabel("Nombre de composantes PCA")
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

from sklearn.metrics import classification_report
import json

def benchmark_componantes(data, componants_amount, save_path="pca_benchmark_results.csv"):
    results = []

    for amount in componants_amount:
        X, y = make_batch(data)
        X_proj, name, pca = project_pca(X, n_components=amount)

        info_keeped = np.sum(pca.explained_variance_ratio_)
        
        print(f"\n📌 Méthode : {name}, with {amount} components")
        #plot_projection(X_proj, y, title=f"{name} ({amount} components)")
        
        print(f"📈 Résultats de classification with {amount} components:")
        X_train, X_test, y_train, y_test = train_test_split(X_proj, y, test_size=0.2, stratify=y, random_state=42)
        
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        plot_pca_components_with_anova(X_proj, y, batch_size=8, cols=4)
        save_pca_components_as_nii(pca, (79, 95, 78))
        
        report = classification_report(y_test, y_pred, output_dict=True)
        result = {
            "components": amount,
            "accuracy": report["accuracy"],
            "macro_f1": report["macro avg"]["f1-score"],
            "macro_precision": report["macro avg"]["precision"],
            "macro_recall": report["macro avg"]["recall"],
            "percentage_of_keeped_info": float(info_keeped)
        }
        results.append(result)

        print("📈 Résultats de classification :")
        classify_projection(X_proj, y)

        # Affiche aussi le rapport texte pour info
        print(json.dumps(result, indent=2))

    # Sauvegarde des résultats dans un CSV
    df = pd.DataFrame(results)
    df.to_csv(save_path, index=False)
    print(f"\n✅ Résultats sauvegardés dans {save_path}")


    print("=================================")
    print(pca.components_)
    print(pca.components_[0].shape)
    
    for component in pca.components_:
        component = component.reshape((79, 95, 78))
        interactive_volume_viewer(component)

    # Tracé des courbes
    plot_pca_performance(results)


if __name__ == "__main__":
    print("================= Data Analyser ==================")

    
    # ==== 1. DONNÉES 3D ====

    loader = Loader.PETScanLoader("../../Desktop/Cancer_pain_data/PETdata/data/", "zscore")
    labelisedData = loader.load_all_labelised()

    enlargementMethod = ['noise', 'blur', 'flip_x']
    EnlargedData = Enlarger.augmentate_batch(labelisedData, enlargementMethod, True, 2)

    #clf, selector = full_pipeline(EnlargedData, k_best=200)

    benchmark_componantes(EnlargedData, [30])
    print(EnlargedData[0]["data"].shape)
