# CancerPainClassifier
# Copyright (c) 2025 Neeko
# License: MIT
# If used in research, please cite: https://github.com/Neeko-strong-tomato/CancerPainClassifier

import loader as Loader
import preprocessing as Preprocesser
import enlarger as Enlarger

import numpy as np
from sklearn.model_selection import train_test_split

def make_batch(data):
        X = []
        Y = []

        for patient in data:
            X.append(patient["data"])
            Y.append(patient["label"])

        for i, scan in enumerate(X):
                if scan.shape != X[0].shape:
                    print(f"Inconsistent shape at index {i}: {scan.shape} vs {X[0].shape}")
                    Enlarger.fix_shape(scan)

        X = np.array(X, dtype=np.float32)
        Y = np.array(Y, dtype=np.int64)


        if len(X.shape) == 4:
            X = np.expand_dims(X, axis=1)  # (N, D, H, W) -> (N, 1, D, H, W)

        return X, Y


class batch :
    def __init__(self, data_dir = "../../Desktop/Cancer_pain_data/PETdata/data/", normalization='zscore',
                  preprocessing_method='identity'):
        
        loader = Loader.PETScanLoader(data_dir, normalization)
        self.labelisedData = loader.load_all_labelised()

        Preprocesser.preprocess_all_scans(self.labelisedData, preprocessing_method)


    def split_train_test(self, enlargement_method, keep_originals = True,
                         max_enlargment_combination=2, test_size=0.2, 
                         random_state=42, enlarge_validation_batch = False, 
                         stratify=True):

        X, Y = make_batch(self.labelisedData)
        print("shape de X :", np.shape(X), "shape de Y :", np.shape(Y))
        X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=test_size, 
                                                          random_state=random_state, stratify=Y if stratify else None, shuffle=True)
        X_train, y_train = Enlarger.augmentate_dataset_separated(X_train, y_train, enlargement_method, 
                                                                 keep_originals, max_enlargment_combination)

        if enlarge_validation_batch :
            X_val, y_val = Enlarger.augmentate_dataset_separated(X_val, y_val, enlargement_method, 
                                                                 keep_originals, max_enlargment_combination)

        return X_train, X_val, y_train, y_val
    

if __name__ == "__main__":
     
     import os

     batch = batch(data_dir=os.path.expanduser("~/Documents/CancerPain/PETdata/data"))
     X, x, Y, y = batch.split_train_test(['blur', 'noise'])
     print("shape de X :", np.shape(X), "shape de Y :", np.shape(Y))
     print("shape de x :", np.shape(x), "shape de y :", np.shape(y))
