# Cancer Pain Classifier

## Context

According to ameli:

Cancer is a disease caused by an initially normal cell whose program goes awry and transforms it. This cell multiplies and produces other, so-called "abnormal" cells, which proliferate anarchicly and excessively.

Among the various forms of cancer, some can be painful for patients. In the particular case of cancers..., splanchnic nerve blocks (a type of surgical operation) can significantly reduce patients' pain.

However, splanchnic nerve blocks are not equally effective in all patients suffering from.... It can even aggravate certain forms of pain or discomfort.

This is why it is important to try to predict whether splanchnic nerve blocks would be a solution for a patient before performing the operation.



#  CancerPainClassifier

> Deep learning-based classification of pain levels in cancer patients using multimodal data (e.g. PET scans, EEG signals).

![Python](https://img.shields.io/badge/python-3.10-blue)
![Torch](https://img.shields.io/badge/torch-2.0-orange)
![License](https://img.shields.io/badge/license-MIT-green)
![Build](https://img.shields.io/badge/status-Prototype-yellow)
![MadeWith](https://img.shields.io/badge/made%20with-MONAI-blue)

---

## Project Overview

CancerPainClassifier is a research prototype that leverages artificial intelligence to classify cancer patients' pain levels using neuroimaging (PET), EEG recordings, and machine learning techniques. The project aims to improve pain assessment and patient stratification in clinical environments through automated multimodal data analysis.

---

## 📁 Project Structure

CancerPainClassifier/  
│  
├── dataManager/ # Preprocessing modules (EEG, PET)  
│ ├── EEG/ # EEG handling and preprocessing  
│ └── PetScan/ # PET scan loading, normalization, logging  
│  
├── models/ # Model definitions and evaluation  
│ ├── model/ # Deep learning models (ResNet, MONAI, etc.)  
│ └── communs/ # Shared utils: metrics, analysis, saving  
│  
├── main.py # Main training/inference script  
├── requirements.txt # Project dependencies  
└── README.md


---

##  Objectives

- Integrate EEG and PET scan data into a unified machine learning pipeline.
- Compare and benchmark multiple models (naive CNNs, MONAI-based).
- Support future clinical decision tools for pain evaluation.
- Build a modular and extensible framework for medical classification tasks.

---

## Methodology

### Preprocessing
- PET: loading NIfTI files using `nibabel`, normalization
- several preprocessing (soon) and augmentation methods are available
- Combined datasets are formatted for PyTorch models.

### Models
- `naiveModel.py`: standard convolutional neural network
- `Resnet.py`: custom MONAI-based model for medical images

### Training & Evaluation
- Use `main.py` to launch the pipeline.
- `batch.py` prepares batches from different data sources, preprocess the data and split them into a training and a validation batch.
- `metrics.py` and `performanceAnalyser.py` compute evaluation metrics.

---

## 📊 Preliminary Results

| Model         | Accuracy | F1-score | AUC    |
|---------------|----------|----------|--------|
| Naive         | xxxx     | xxxx     | xxxx   |
| ResNet        | xxxx     | xxxx     | xxxx   |

> *those results come from some training session*

---

## 📖 Citation

If you use this repository in your research, please cite it as:

Neeko (2025). CancerPainClassifier: Deep Learning-based Pain Classification in Cancer Patients. GitHub repository: https://github.com/Neeko-strong-tomato/CancerPainClassifier

## ⚙️ Installation

```bash
git clone https://github.com/Neeko-strong-tomato/CancerPainClassifier.git
cd CancerPainClassifier
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
