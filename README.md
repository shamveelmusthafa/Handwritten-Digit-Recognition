# 🔢 Handwritten Digit Recognition
### PRCP-1002 | Machine Learning Project

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://handwritten-digitrecognition.streamlit.app/)

> **Live Demo:** [https://handwritten-digitrecognition.streamlit.app/](https://handwritten-digitrecognition.streamlit.app/)

---

## 📌 Project Overview

This project builds and evaluates multiple machine learning classifiers capable of accurately identifying handwritten digits from the **MNIST dataset** — the de facto benchmark for image classification algorithms. Each input image is classified into one of ten digit categories (0 through 9).

The project covers the full data science pipeline — from exploratory data analysis and preprocessing through model building, hyperparameter tuning, error analysis, and a live interactive web application for real-time digit prediction.

---

## 🗂️ Project Structure

```
Handwritten-Digit-Recognition/
│
├── app.py                        ← Streamlit web application
├── requirements.txt              ← Python dependencies
├── Handwritten.ipynb             ← Full project notebook
├── Dataset.docx                  ← Dataset documentation
│
└── saved_models/
    ├── logistic_regression.pkl
    ├── knn.pkl
    ├── svm.pkl
    ├── random_forest.pkl
    ├── mlp.pkl
    ├── mlp_tuned.pkl             ← Best performing model (deployed)
    ├── svm_tuned.pkl
    └── model_metadata.json
```

---

## 📊 Dataset

The **MNIST (Modified National Institute of Standards and Technology)** dataset is the benchmark dataset used in this project.

| Property | Details |
|---|---|
| Total Images | 70,000 grayscale images |
| Training Set | 60,000 images |
| Test Set | 10,000 images |
| Image Size | 28 × 28 pixels (784 features when flattened) |
| Color Space | Grayscale (pixel values 0–255) |
| Total Classes | 10 (digits 0 through 9) |
| Class Balance | Near-perfectly balanced (~5,900–6,700 images per class) |
| Missing Values | None |

---

## 🔍 Exploratory Data Analysis

The EDA section covers a comprehensive visual and statistical exploration of the MNIST dataset:

- **Class Distribution** — confirmed near-perfect balance across all 10 digit classes
- **Sample Image Visualization** — 3 samples per digit across all classes
- **Average Pixel Intensity Heatmap** — reveals typical stroke patterns per digit
- **Pixel Intensity Distribution** — bimodal distribution confirmed across all classes (background near 0, strokes near 1)
- **Mean and Standard Deviation Images** — highlights high-variance regions most discriminative for classification
- **PCA 2D Projection** — shows clear clustering with expected overlap between visually similar digits (4 & 9, 3 & 8)
- **PCA Scree Plot** — 90% variance captured by ~87 components, 95% by ~154 components
- **Pixel Row Correlation Heatmap** — confirms spatially continuous stroke patterns
- **Brightest and Darkest Images** — identifies most and least ambiguous samples in the training set

---

## ⚙️ Preprocessing

Three preprocessing steps were applied before model training:

1. **Normalization** — Pixel values scaled from (0–255) to (0–1) by dividing by 255.0
2. **Flattening** — Each 28×28 image reshaped into a 1D vector of 784 features
3. **Data Integrity Verification** — Confirmed zero missing values, zero corrupted images, and checked for duplicate samples

**Note on Augmentation:** Data augmentation was deliberately not applied. MNIST images are already standardised and centred, the dataset is large enough (60,000 samples) to prevent overfitting, and the models used are not spatially aware — making augmentation unnecessary and potentially harmful.

---

## 🤖 Models Built & Evaluated

Five machine learning models were trained and evaluated on the full MNIST dataset:

| Rank | Model | Test Accuracy | Train Time |
|---|---|---|---|
| 1st | **SVM (Tuned)** | **98.33%** | ~84 mins (incl. CV) |
| 2nd | MLP Neural Network (Tuned) | 98.05% | ~22 mins (incl. CV) |
| 3rd | MLP Neural Network (Original) | 97.97% | ~82 sec |
| 4th | SVM (Original) | 98.37%* | ~198 sec |
| 5th | Random Forest | 96.90% | ~57 sec |
| 6th | K-Nearest Neighbors | 96.88% | ~16 sec |
| 7th | Logistic Regression | 92.60% | ~90 sec |

> *The original SVM scored marginally higher on the single test set (98.37% vs 98.33%), however the tuned model's cross-validated score of 98.02% is statistically more reliable as it averages performance across 3 independent data splits.

---

## 🎯 Hyperparameter Tuning

The top two models — **SVM** and **MLP Neural Network** — were selected for fine-tuning using `RandomizedSearchCV` (8 combinations × 3-fold CV = 24 fits each).

**SVM Best Parameters:**
- Kernel: RBF
- C: 50
- Gamma: scale

**MLP Best Parameters:**
- Architecture: 512 → 256 → 128
- Activation: tanh
- Learning Rate: 0.0005
- Batch Size: 64
- Alpha: 0.0001

---

## 📉 Error Analysis — Best Model (SVM Tuned)

| Metric | Value |
|---|---|
| Total Test Images | 10,000 |
| Correctly Classified | 9,833 |
| Misclassified | 167 |
| Overall Error Rate | 1.67% |

- **Most difficult digit:** Digits with curved strokes (5, 8, 9) — highest error rates due to shared structural patterns
- **Easiest digit:** Digit 1 — unique single vertical stroke is highly discriminative
- **Most confused pair:** Digit 4 mistaken as Digit 9 — both share a closed upper loop and downward stem

---

## ⚠️ Challenges Faced

1. **SVM Scalability** — Training time scales poorly with dataset size. Full 60,000 sample training took ~198 seconds; fine-tuning required over 84 minutes.
2. **SVM Probability Calibration** — RBF kernel does not natively output class probabilities. Softmax normalisation was applied to convert decision function scores into pseudo-probabilities.
3. **Tuned SVM vs Original** — The tuned SVM scored marginally lower on the test set than the original, highlighting the limitation of single test set evaluation versus cross-validated scoring.
4. **MLP Spatial Blindness** — Fully connected MLP treats every pixel independently with no awareness of spatial structure, unlike CNNs.
5. **Random Forest Pixel Independence** — Random Forest evaluates pixels in isolation rather than as spatially correlated groups, explaining its performance ceiling.

---

## 🚀 Live Demo

The best performing model (MLP Tuned) is deployed as an interactive Streamlit web application:

**[👉 Try the Live App](https://handwritten-digitrecognition.streamlit.app/)**

Draw a digit on the canvas and the model will predict it in real time with confidence scores.

---

## 🛠️ Tech Stack

| Category | Libraries |
|---|---|
| Data Manipulation | NumPy, Pandas |
| Visualisation | Matplotlib, Seaborn, Plotly |
| Machine Learning | Scikit-learn |
| Deep Learning (data loading) | TensorFlow / Keras |
| Web Application | Streamlit, Streamlit-Drawable-Canvas |
| Model Persistence | Joblib |

---

## 💻 Run Locally

1. **Clone the repository:**
```bash
git clone https://github.com/shamveelmusthafa/Handwritten-Digit-Recognition.git
cd Handwritten-Digit-Recognition
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run the Streamlit app:**
```bash
streamlit run app.py
```

---

## 📁 Key Files

| File | Description |
|---|---|
| `app.py` | Streamlit web application for live digit prediction |
| `Handwritten.ipynb` | Complete project notebook with all analysis and models |
| `requirements.txt` | All Python dependencies for deployment |
| `saved_models/mlp_tuned.pkl` | Best deployed model |
| `saved_models/svm_tuned.pkl` | Best overall accuracy model |
| `saved_models/model_metadata.json` | Model performance metadata |

---

## 👤 Author

**Shamveel Musthafa**
- GitHub: [@shamveelmusthafa](https://github.com/shamveelmusthafa)

---

*Project completed as part of the Datamites Data Science program — PRCP-1002*
