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
    return pca.fit_transform(X), "PCA"

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
            "macro_recall": float
        }
    """
    components = [r["components"] for r in results]
    accuracy = [r["accuracy"] for r in results]
    f1 = [r["macro_f1"] for r in results]
    precision = [r["macro_precision"] for r in results]
    recall = [r["macro_recall"] for r in results]

    plt.figure(figsize=(10, 6))
    plt.plot(components, accuracy, label="Accuracy", marker='o')
    plt.plot(components, f1, label="Macro F1-score", marker='s')
    plt.plot(components, precision, label="Macro Precision", marker='^')
    plt.plot(components, recall, label="Macro Recall", marker='x')

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
        X_proj, name = project_pca(X, n_components=amount)
        
        print(f"\n📌 Méthode : {name}, with {amount} components")
        plot_projection(X_proj, y, title=f"{name} ({amount} components)")
        
        print(f"📈 Résultats de classification with {amount} components:")
        X_train, X_test, y_train, y_test = train_test_split(X_proj, y, test_size=0.2, stratify=y, random_state=42)
        
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        report = classification_report(y_test, y_pred, output_dict=True)
        result = {
            "components": amount,
            "accuracy": report["accuracy"],
            "macro_f1": report["macro avg"]["f1-score"],
            "macro_precision": report["macro avg"]["precision"],
            "macro_recall": report["macro avg"]["recall"]
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

    # Tracé des courbes
    plot_pca_performance(results)


if __name__ == "__main__":
    print("================= Data Analyser ==================")

    
    # ==== 1. DONNÉES 3D ====

    loader = Loader.PETScanLoader("../../Desktop/Cancer_pain_data/PETdata/data/", "zscore")
    labelisedData = loader.load_all_labelised()

    enlargementMethod = ['noise', 'flip_x']
    EnlargedData = Enlarger.augmentate_batch(labelisedData, enlargementMethod, True, 2)

    #clf, selector = full_pipeline(EnlargedData, k_best=200)

    benchmark_componantes(EnlargedData, [2,3,4,5])