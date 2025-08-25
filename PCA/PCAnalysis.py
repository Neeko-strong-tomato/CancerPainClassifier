import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from dataManager.PetScan.batch import batch

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import classification_report, f1_score
from tqdm import tqdm


# === Flatten 3D ===
class FlattenTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X.reshape(len(X), -1)


# === Pipeline with PCA + SelectKBest + RF ===
def pca_feature_selection_pipeline(X_train, y_train, X_val, y_val, n_features_list):
    results = []

    for n in tqdm(n_features_list, desc="Testing PCA dims"):
        max_n = min(n, X_train.shape[0], X_train.reshape(len(X_train), -1).shape[1])

        pipe = Pipeline([
            ("flatten", FlattenTransformer()),
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=max_n, random_state=42)),
            ("select", SelectKBest(score_func=f_classif, k="all")),  # sélection après PCA
            ("clf", RandomForestClassifier(random_state=42, n_estimators=300))
        ])

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_val)

        # Scores globaux
        f1_macro = f1_score(y_val, y_pred, average="macro")
        f1_class1 = f1_score(y_val, y_pred, pos_label=1, average="binary")

        print(f"\n=== PCA {max_n} components ===")
        print(classification_report(y_val, y_pred, digits=3))

        results.append({
            "n_features": max_n,
            "f1_macro": f1_macro,
            "f1_class1": f1_class1
        })

    return results


# === Visuals ===
def plot_results(results):
    n_features = [r["n_features"] for r in results]
    f1_macro = [r["f1_macro"] for r in results]
    f1_class1 = [r["f1_class1"] for r in results]

    plt.figure(figsize=(8, 5))
    plt.plot(n_features, f1_macro, marker="o", label="F1 Macro (global)")
    plt.plot(n_features, f1_class1, marker="x", label="F1 Classe 1 (anomalie)")
    plt.xlabel("Nombre de composantes PCA conservées")
    plt.ylabel("Score F1")
    plt.title("Impact du nombre de features PCA sur la perf du RF")
    plt.legend()
    plt.grid(True)
    plt.show()




if __name__ == "__main__":
    
    batch = batch(data_dir=os.path.expanduser("~/Documents/CancerPain/PETdata/data"), 
                  preprocessing_method=['mask' ,'mean_template'],
                  normalization='zscore', 
                  show_data_evolution=False,
                  up_sampling=True,
                  verbose=True)
    X_train, X_val, y_train, y_val = batch.split_train_test(['blur', 'adjust_contrast'])

    results = pca_feature_selection_pipeline(
        X_train, y_train,
        X_val, y_val,
        n_features_list=[10, 20, 50, 100]
        )

    plot_results(results)
