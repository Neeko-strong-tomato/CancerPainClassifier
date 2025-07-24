
# 📘 Code Reference — Data Manager

This section documents the components of the data management pipeline used during the pre-training phase. It includes data loading, preprocessing, and data augmentation utilities.

---

# 📁 Module Overview


###  Loader


Responsible for locating and loading raw patient data from files, converting them to numerical matrices, and assigning appropriate labels. Then apply a normalization.

🔧 Function Overview:
- list_available_patients( )
- load_scan( )
- apply_normalization( )

For more detail, you should look over the descriptions & prototypes of the wanted functions.

###  Preprocessing

This section is used to pre-process the data in order to make it more intelligible for the model learning process.

🔧 Function Overview:
- preprocess_a_scan() (example, coming soon)

For more detail, you should look over the descriptions & prototypes of the wanted functions.

### Enlarge


As the dataset is limited, this module implements various strategies to artificially expand the training data. This helps prevent overfitting and improves generalization.

🔧 Function Overview:
- augment_a_scan()
- augmentate_dataset_separated()
- fix_shape()

For more detail, you should look over the descriptions & prototypes of the wanted functions.

### Batch

This file contains a set of functions for creating a batch using a scan recovery and transformation pipeline based on the modules described above.

🔧 Function Overview:
- batch( )
- split_train_test( )

For more detail, you should look over the descriptions & prototypes of the wanted functions.