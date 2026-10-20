# Machine Learning Projects

**A hands-on portfolio of end-to-end machine learning, deep learning, and time-series projects — covering classification, regression, clustering, dimensionality reduction, computer vision, and forecasting — built with scikit-learn, TensorFlow/Keras, statsmodels, and Plotly.**

This repository collects 18 projects across 23 Jupyter notebooks, organized into six topic folders. Each project walks through the full workflow: data loading, cleaning and preprocessing, model building, hyperparameter tuning, evaluation, and interpretation of results.

> **Project status:** Portfolio collection / actively growing

![Python](https://img.shields.io/badge/python-3.x-blue)
![Jupyter](https://img.shields.io/badge/notebooks-Jupyter-f37626)
![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-f7931e)
![TensorFlow](https://img.shields.io/badge/DL-TensorFlow%20%2F%20Keras-ff6f00)
![statsmodels](https://img.shields.io/badge/time%20series-statsmodels-4b8bbe)
![Plotly](https://img.shields.io/badge/viz-Plotly%20%7C%20Seaborn-3f4f75)

---

## 📑 Table of Contents

- [Highlights](#-highlights)
- [Results at a Glance](#-results-at-a-glance)
- [Repository Structure](#-repository-structure)
- [Projects](#-projects)
  - [1. Classification](#1-classification)
  - [2. Clustering](#2-clustering)
  - [3. Dimensionality Reduction](#3-dimensionality-reduction)
  - [4. Regression](#4-regression)
  - [5. Deep Learning](#5-deep-learning)
  - [6. Time Series Analysis](#6-time-series-analysis)
- [Datasets](#-datasets)
- [Setup](#%EF%B8%8F-setup)
- [Tech Stack](#%EF%B8%8F-tech-stack)
- [Notes & Limitations](#%EF%B8%8F-notes--limitations)
- [License](#-license)
- [Author](#-author)

---

## ✨ Highlights

- **Full-pipeline thinking** — raw data → cleaning → feature engineering → modelling → evaluation → reusable inference code.
- **Classical ML** — logistic regression, K-Nearest Neighbors, decision trees, random forests, Ridge/Lasso/OLS regression.
- **Unsupervised learning** — K-Means with elbow + silhouette tuning, hierarchical clustering with dendrograms and heatmaps.
- **Dimensionality reduction** — PCA, LDA, t-SNE, and UMAP compared on the same dataset, with 2D and interactive 3D visualizations.
- **Deep learning** — dense networks, the Keras Functional API, CNNs trained from scratch, and VGG16 transfer learning.
- **Time series** — LSTM, GRU, and 1D-CNN forecasters, plus classical ARIMA/SARIMAX/GARCH analysis of a real corporate takeover.
- **Rigorous methodology** — `GridSearchCV` tuning, cross-validation, early stopping, model checkpointing, regularization, and OLS assumption checks (linearity, multicollinearity via VIF).
- **Production-minded code** — a custom scikit-learn scaler and a reusable prediction module that loads pickled artifacts (`absenteeism_module.py`).

---

## 📊 Results at a Glance

| Project | Task | Model | Key Result |
|---|---|---|---|
| Employee Absenteeism | Binary classification | Logistic Regression | **76.98%** test accuracy |
| Iris Species | Multiclass classification | KNN + GridSearchCV | **97.78%** test accuracy |
| Breast Cancer Diagnosis | Binary classification | Random Forest + feature selection | **95.61%** test accuracy |
| Titanic Survival | Binary classification | Decision Tree | **77.53%** accuracy, 78.72% precision |
| Market Segmentation | Clustering | K-Means | **4** customer segments identified |
| Wine Dimensionality Reduction | Dimensionality reduction | PCA / LDA / t-SNE / UMAP | 4 PCs capture **73.6%** of variance |
| Used Car Prices | Regression | Ridge / Lasso (one-hot) | **R² ≈ 0.81** on test data |
| MNIST Digits | Image classification | Dense neural network | **96.96%** test accuracy |
| Audiobook Customers | Binary classification | Dense neural network | **80.80%** test accuracy |
| California Housing | Regression | ANN (Sequential & Functional) | **MAE ≈ 0.35** (target in $100k) |
| Cats vs Dogs (CNN) | Image classification | CNN from scratch | ~**71%** validation accuracy |
| Cats vs Dogs (VGG16) | Image classification | Transfer learning | **96.10%** test accuracy |
| Temperature Forecasting | Time-series regression | LSTM / 1D-CNN / GRU | Validation RMSE ≈ **0.8 °C** |

---

## 📁 Repository Structure

```text
Machine-Learning-Projects/
├── Classification/
│   ├── Preprocessing exessively and moderately absent employees datasets.ipynb
│   ├── Applying Machine Learning to the Absenteeism Dataset.ipynb
│   ├── New_absenteeism_predictions.ipynb
│   ├── absenteeism_module.py
│   ├── Classifying species of Iris flowers using KNeigborsClassifier.ipynb
│   ├── Diagnosing malignant and benign tumour from the cancer dataset using random forest.ipynb
│   └── Using decision tree to predict number of survivors based on sex in the titanic dataset.ipynb
├── Clustering/
│   ├── Dendogram and Heatmap.ipynb
│   ├── KMeans Clustering-Market Segmentation.ipynb
│   └── Dimensionality Reduction on Wine Dataset using PCA, LDA, T-SNE and UMAP.ipynb
├── Deep Learning/
│   ├── Minimal Example with TensorFlow.ipynb
│   ├── Image Classification (MNIST Dataset).ipynb
│   ├── Preprocessing_Predicting if a customer would buy again from an audio book company.ipynb
│   ├── Predicting if a customer would buy again from an audio book company.ipynb
│   ├── Regression with ANN.ipynb
│   ├── Regression with ANN using functional API.ipynb
│   ├── Classifying coloured images of cats and dogs in cnn.ipynb
│   └── Classifying_coloured_images_of_cats_and_dogs_using_VGG16.ipynb
├── Dimensionality Reduction using PCA, TSNE, LDA, UMAP/
│   └── Dimensionality Reduction on Wine Dataset using PCA, LDA, T-SNE and UMAP.ipynb
├── Regression/
│   ├── Predicting the price of used cars using label encoding.ipynb
│   └── Predicting the price of used cars using one-hot encoding.ipynb
└── Time Series Analysis/
    ├── Temperature prediction using LSTM.ipynb
    ├── Temperature_and_Pressure_prediction_using_LSTM.ipynb
    ├── Temperature_prediction_1D_CNN_vs_GRU.ipynb
    └── Volkswagen's buyout of Porsche.ipynb
```

---

## 🧪 Projects

### 1. Classification

📂 [`Classification/`](./Classification)

#### 1.1 Employee Absenteeism — End-to-End Pipeline

| Notebook | Purpose |
|---|---|
| [Preprocessing](./Classification/Preprocessing%20exessively%20and%20moderately%20absent%20employees%20datasets.ipynb) | Cleans and engineers features from the raw dataset |
| [Modelling](./Classification/Applying%20Machine%20Learning%20to%20the%20Absenteeism%20Dataset.ipynb) | Trains, interprets, and saves the logistic regression model |
| [New predictions](./Classification/New_absenteeism_predictions.ipynb) | Scores new employee data using the saved model |
| [`absenteeism_module.py`](./Classification/absenteeism_module.py) | Reusable `CustomScaler` and `absenteeism_model` classes |

**Objective:** Predict whether an employee will be *excessively absent* (more than the median of 3 hours of absence).

**Approach**
- **Preprocessing:** removed duplicate rows (692 records kept), dropped the `ID` column, and one-hot encoded the 28 absence reasons before grouping them into **4 reason categories**. Extracted **month** and **day of the week** from the date, and collapsed `Education` into a binary feature.
- **Target engineering:** converted absenteeism hours into a binary label using the median (≈46% positive class).
- **Scaling:** built a **custom scikit-learn transformer** (`CustomScaler`) that standardizes only selected numeric columns and leaves dummy variables untouched.
- **Modelling:** logistic regression on an 80/20 split.
- **Interpretation:** built a coefficient summary table and converted weights to **odds ratios**.
- **Deployment-style workflow:** pickled the model and scaler, then wrapped them in `absenteeism_model` so new data can be cleaned and scored with three lines of code.

**Results**
- Training accuracy: **77.03%** · Test accuracy: **76.98%**
- Confusion matrix (test): `[[65, 15], [17, 42]]`
- Strongest predictors by odds ratio: **Reason 3** (≈20.5×) and **Reason 1** (≈16.8×), followed by Reason 4 (≈2.5×).

```python
from absenteeism_module import *

model = absenteeism_model('model', 'scaler')
model.load_and_clean_data('Absenteeism_new_data.csv')
model.predicted_outputs()
```

---

#### 1.2 Iris Species Classification with K-Nearest Neighbors

📓 [`Classifying species of Iris flowers using KNeigborsClassifier.ipynb`](./Classification/Classifying%20species%20of%20Iris%20flowers%20using%20KNeigborsClassifier.ipynb)

**Objective:** Classify iris flowers as *setosa*, *versicolor*, or *virginica* from petal measurements.

**Approach:** Used `petal_len` and `petal_wd` as features, a stratified 70/30 split, and `GridSearchCV` (5-fold) over `n_neighbors` from 2 to 9. Also demonstrates label prediction and class-probability prediction on new samples.

**Results:** Best `k = 5` · cross-validation accuracy **97.14%** · test accuracy **97.78%** (a single versicolor/virginica confusion).

---

#### 1.3 Breast Cancer Diagnosis with Random Forest

📓 [`Diagnosing malignant and benign tumour from the cancer dataset using random forest.ipynb`](./Classification/Diagnosing%20malignant%20and%20benign%20tumour%20from%20the%20cancer%20dataset%20using%20random%20forest.ipynb)

**Objective:** Diagnose tumours as malignant or benign from 30 cell-nucleus measurements.

**Approach**
- Extensive `GridSearchCV` (5-fold, scored on **F1**) across number of trees, split criterion, depth, leaf sizes, `max_features`, and bootstrap.
- Ranked **feature importances**, then used `SelectFromModel` to reduce the input to the **18 most informative features** and retrained.

**Results**
- Best CV F1: **0.972** · Full-feature test accuracy: **94.74%**
- Reduced-feature model test accuracy: **95.61%** — better performance with fewer inputs.
- Most important features: `worst perimeter`, `worst radius`, `mean concave points`, `worst area`.

---

#### 1.4 Titanic Survival with Decision Trees

📓 [`Using decision tree to predict number of survivors based on sex in the titanic dataset.ipynb`](./Classification/Using%20decision%20tree%20to%20predict%20number%20of%20survivors%20based%20on%20sex%20in%20the%20titanic%20dataset.ipynb)

**Objective:** Predict passenger survival from class, sex, age, family aboard, and fare.

**Approach:** Encoded sex as a boolean feature, tuned a decision tree with `GridSearchCV` (F1 scoring), and visualized the final tree with **Graphviz**.

**Results:** Test accuracy **77.53%** · precision **78.72%** · recall **55.22%**.

---

### 2. Clustering

📂 [`Clustering/`](./Clustering)

#### 2.1 Market Segmentation with K-Means

📓 [`KMeans Clustering-Market Segmentation.ipynb`](./Clustering/KMeans%20Clustering-Market%20Segmentation.ipynb)

**Objective:** Segment customers by **satisfaction** and **loyalty**.

**Approach:** Standardized features, applied the **elbow method** (WCSS for 1–9 clusters), then validated the choice with a **silhouette-score `GridSearchCV`** over cluster count, initialization, `n_init`, and `max_iter`.

**Results:** The search selected **4 clusters** (`k-means++`), visualized as colour-coded customer segments on a satisfaction-vs-loyalty plot.

---

#### 2.2 Hierarchical Clustering — Dendrogram & Heatmap

📓 [`Dendogram and Heatmap.ipynb`](./Clustering/Dendogram%20and%20Heatmap.ipynb)

**Objective:** Group countries by geographic coordinates using hierarchical clustering.

**Approach:** Used standardized latitude/longitude with Seaborn's `clustermap` to produce a combined heatmap and dendrogram.

---

### 3. Dimensionality Reduction

📂 [`Dimensionality Reduction using PCA, TSNE, LDA, UMAP/`](./Dimensionality%20Reduction%20using%20PCA%2C%20TSNE%2C%20LDA%2C%20UMAP)

📓 [`Dimensionality Reduction on Wine Dataset using PCA, LDA, T-SNE and UMAP.ipynb`](./Dimensionality%20Reduction%20using%20PCA%2C%20TSNE%2C%20LDA%2C%20UMAP/Dimensionality%20Reduction%20on%20Wine%20Dataset%20using%20PCA%2C%20LDA%2C%20T-SNE%20and%20UMAP.ipynb)

**Objective:** Compare four dimensionality-reduction techniques on the 13-feature Wine dataset (178 samples, 3 cultivars).

**Approach**
- **PCA** — evaluated three component-selection rules: the 90%-variance threshold (8 components), **Kaiser's rule** (3), and the **scree-plot elbow** (4). Chose 4 components as a balance between simplicity and information retained, then built score plots, loading plots, and an interactive **3D Plotly** scatter (PC4 encoded as point size).
- **LDA** — supervised 2D projection.
- **t-SNE** — 2D and 3D non-linear embeddings.
- **UMAP** — tuned with `n_neighbors=10`, `min_dist=0.1`, `metric='correlation'`.

**Results:** The four retained principal components explain **73.6%** of the variance (PC1 36.2%, PC2 19.2%, PC3 11.1%, PC4 7.1%), and all four methods separate the wine classes visually.

> The same notebook also appears in the `Clustering/` folder.

---

### 4. Regression

📂 [`Regression/`](./Regression)

#### Used Car Price Prediction — Label Encoding vs One-Hot Encoding

| Notebook | Encoding strategy |
|---|---|
| [Label encoding](./Regression/Predicting%20the%20price%20of%20used%20cars%20using%20label%20encoding.ipynb) | `LabelEncoder` on categorical variables |
| [One-hot encoding](./Regression/Predicting%20the%20price%20of%20used%20cars%20using%20one-hot%20encoding.ipynb) | `pd.get_dummies(drop_first=True)` |

**Objective:** Predict used-car prices and measure how categorical-encoding choices affect model quality.

**Approach**
- **Cleaning:** dropped missing values (4,345 → 4,025 rows) and trimmed outliers in price, mileage, engine volume, and year (3,867 rows retained).
- **OLS assumption checks:** inspected linearity and fixed the skewed price relationship with a **log transformation**; detected multicollinearity with **Variance Inflation Factor (VIF)** and removed `Year`.
- **Modelling:** linear regression, plus **Ridge** and **Lasso** tuned with `GridSearchCV` and evaluated with 5-fold cross-validation.
- **Feature selection:** used Lasso coefficients to build a reduced feature set and retrained.
- **Diagnostics:** residual plots, R², adjusted R², MSE, MAE, and back-transformed price predictions (`np.exp`) compared against actual prices.

**Results**

| Metric | Label Encoding | One-Hot Encoding |
|---|---|---|
| Mean CV R² (Ridge) | 0.695 | **0.796** |
| Test MSE (Ridge) | 0.270 | **0.157** |
| Test R² (Ridge / Lasso) | — | **0.809 / 0.807** |

**Takeaway:** One-hot encoding clearly outperforms label encoding for nominal features — label encoding imposes an artificial ordering that a linear model cannot ignore.

---

### 5. Deep Learning

📂 [`Deep Learning/`](./Deep%20Learning)

#### 5.1 Minimal Example with TensorFlow

📓 [`Minimal Example with TensorFlow.ipynb`](./Deep%20Learning/Minimal%20Example%20with%20TensorFlow.ipynb)

Builds a single-neuron linear model on 1,000 synthetic observations (`targets ≈ 2x − 3z + 5` plus noise) and trains it with SGD for 100 epochs. The learned weights (≈ `3.0` and `−3.0`) and bias (≈ `5.0`) recover the underlying relationship; the effective `x` coefficient is 3 because the noise term is itself built from `x`.

---

#### 5.2 MNIST Handwritten Digit Classification

📓 [`Image Classification (MNIST Dataset).ipynb`](./Deep%20Learning/Image%20Classification%20(MNIST%20Dataset).ipynb)

Loads MNIST with **TensorFlow Datasets**, scales pixel values, creates a 10% validation split, and trains a dense network (two hidden layers of 50 ReLU units, softmax output) with Adam for 5 epochs.

**Result:** **96.96%** test accuracy.

---

#### 5.3 Audiobook Customer Retention

| Notebook | Purpose |
|---|---|
| [Preprocessing](./Deep%20Learning/Preprocessing_Predicting%20if%20a%20customer%20would%20buy%20again%20from%20an%20audio%20book%20company.ipynb) | Balances, scales, shuffles, and splits the data; saves `.npz` files |
| [Model](./Deep%20Learning/Predicting%20if%20a%20customer%20would%20buy%20again%20from%20an%20audio%20book%20company.ipynb) | Trains and tests the neural network |

**Objective:** Predict whether a customer will purchase again from an audiobook company.

**Approach:** Balanced the classes to ~50/50 priors, standardized inputs, shuffled, and split 80/10/10 (3,579 / 447 / 448 samples). Trained a network with two hidden layers of 200 ReLU units, a softmax output, and **early stopping**.

**Result:** **80.80%** test accuracy.

---

#### 5.4 Regression with ANN — California Housing

| Notebook | API |
|---|---|
| [Regression with ANN](./Deep%20Learning/Regression%20with%20ANN.ipynb) | Keras Sequential |
| [Regression with ANN using functional API](./Deep%20Learning/Regression%20with%20ANN%20using%20functional%20API.ipynb) | Keras Functional (wide & deep) |

**Objective:** Predict median house value on the California Housing dataset (20,640 samples, 8 features).

**Approach:** Standardized features, used train/validation/test splits (13,209 / 3,303 / 4,128), three hidden layers of 32 units, Adam, **early stopping** (patience 10), and **model checkpointing**. The Functional API variant concatenates the raw inputs with the last hidden layer (a *wide & deep* architecture).

**Results**

| Model | Parameters | Test MSE | Test MAE |
|---|---|---|---|
| Sequential | 2,433 | 0.274 | 0.349 |
| Functional (wide & deep) | 2,441 | 0.273 | 0.347 |

---

#### 5.5 Cats vs Dogs — CNN from Scratch

📓 [`Classifying coloured images of cats and dogs in cnn.ipynb`](./Deep%20Learning/Classifying%20coloured%20images%20of%20cats%20and%20dogs%20in%20cnn.ipynb)

**Dataset:** 2,000 training, 1,000 validation, and 1,000 test images (balanced between cats and dogs), resized to 150×150.

**Approach:** Applied heavy **data augmentation** (rotation, shifts, shear, zoom, horizontal flips) and trained a 4-block convolutional network (3.45M parameters) with dropout (0.5) and a dense head.

**Result:** Validation accuracy reached roughly **71%** after 20 epochs — a baseline that motivates transfer learning.

---

#### 5.6 Cats vs Dogs — Transfer Learning with VGG16

📓 [`Classifying_coloured_images_of_cats_and_dogs_using_VGG16.ipynb`](./Deep%20Learning/Classifying_coloured_images_of_cats_and_dogs_using_VGG16.ipynb)
&nbsp;[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Shakpro10/Machine-Learning-Projects/blob/main/Deep%20Learning/Classifying_coloured_images_of_cats_and_dogs_using_VGG16.ipynb)

**Approach:** Used an ImageNet-pretrained **VGG16** convolutional base with a custom dense head, a very low learning rate (`2e-5`) for fine-tuning, the same augmentation pipeline, and checkpointing with early stopping. Trained on Google Colab (TPU/Drive workflow) with support for resuming training from saved weights.

**Result:** **96.10%** test accuracy — a jump of roughly 25 percentage points over the from-scratch CNN on the same data.

---

### 6. Time Series Analysis

📂 [`Time Series Analysis/`](./Time%20Series%20Analysis)

#### 6.1 Temperature Forecasting with LSTM

📓 [`Temperature prediction using LSTM.ipynb`](./Time%20Series%20Analysis/Temperature%20prediction%20using%20LSTM.ipynb)
&nbsp;[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Shakpro10/Machine-Learning-Projects/blob/main/Time%20Series%20Analysis/Temperature%20prediction%20using%20LSTM.ipynb)

**Dataset:** Jena Climate (2009–2016), resampled from 10-minute to **hourly** readings.

**Approach:** Created a sliding-window dataset (5 previous hours → next hour), split 80/10/10 chronologically, and trained an LSTM (64 units) with Adam, checkpointing, and early stopping. Predictions are plotted against actuals for train, validation, and test sets.

**Result:** Validation RMSE ≈ **0.79 °C**.

---

#### 6.2 Multivariate Forecasting — Temperature & Pressure

📓 [`Temperature_and_Pressure_prediction_using_LSTM.ipynb`](./Time%20Series%20Analysis/Temperature_and_Pressure_prediction_using_LSTM.ipynb)

**Approach:** Forecast **two targets at once** from a 7-hour window of six features. Time of day and time of year are encoded with **cyclical sine/cosine features**, and inputs/targets are **min-max normalized** (then inverse-transformed so plots are in real units). Architecture: LSTM(64) → Dense(32) → Dense(8) → Dense(2).

**Result:** Validation RMSE ≈ **0.010** on the normalized scale.

---

#### 6.3 1D-CNN vs GRU

📓 [`Temperature_prediction_1D_CNN_vs_GRU.ipynb`](./Time%20Series%20Analysis/Temperature_prediction_1D_CNN_vs_GRU.ipynb)

**Approach:** Benchmarks a `Conv1D` model (8,689 parameters) against a `GRU` model (15,217 parameters) on the same hourly temperature forecasting task, with side-by-side prediction plots.

**Result:** Both reach a validation RMSE of roughly **0.8 °C**; the lighter CNN is competitive with the recurrent model at a much smaller parameter count and faster training.

---

#### 6.4 Volkswagen's Buyout of Porsche

📓 [`Volkswagen's buyout of Porsche.ipynb`](./Time%20Series%20Analysis/Volkswagen%27s%20buyout%20of%20Porsche.ipynb)

**Objective:** Analyze the stock behaviour of Volkswagen, Porsche, and BMW around the VW–Porsche takeover and the 2015 "Dieselgate" scandal.

**Key dates analyzed:** start (Apr 2009) → first announcement, 49.9% stake (Dec 2009) → second announcement, 50.1% stake (Jul 2012) → end of window (Jan 2014) → Dieselgate (Sep 2015).

**Approach**
- Pulled daily prices with **`yfinance`** (`VOW3.DE`, `PAH3.DE`, `BMW.DE`), computed returns and squared returns, and set a business-day frequency.
- Compared **correlations** between the three manufacturers in each period.
- Fitted the best **`auto_arima`** models (with the other two stocks as exogenous variables) separately for each period, for both **prices** and **returns**.
- Produced forecasts with and without exogenous variables (Porsche, BMW, or both), including a seasonal specification.
- Modelled **volatility** in each period with **GARCH(1,1)**.

**Findings:** As the takeover progressed, the Volkswagen–Porsche correlation rose from **0.66** (before the first announcement) to **0.74** (between announcements) and **0.94** (after the second announcement), showing the two stocks converging. Over the longer period after 2014, Volkswagen's correlation with BMW dropped to about **0.50**, while its correlation with Porsche remained high at about **0.86**.

---

## 🗂️ Datasets

Datasets are **not stored in this repository**. Notebooks load them as follows:

| Project | Source |
|---|---|
| Employee absenteeism | `Absenteeism_data.csv` / `Absenteeism_new_data.csv` (local files) |
| Iris | Loaded from a hosted CSV (`sololearn.com`) |
| Breast cancer, Wine, California housing | Built into scikit-learn |
| Titanic | Loaded from a hosted CSV (`sololearn.com`) |
| Market segmentation | `3.12. Example.csv` (local file) |
| Country clusters | `Country clusters standardized.csv` (local file) |
| Used cars | `1.04. Real life example.csv` (local file) |
| MNIST | `tensorflow_datasets` |
| Audiobooks | `Audiobooks_data.csv` (local file) |
| Cats vs Dogs | `cats_and_dogs_small/` image folders (`train`, `validation`, `test`) |
| Jena climate | Downloaded automatically via `tf.keras.utils.get_file` |
| VW / Porsche / BMW stocks | Downloaded via `yfinance` |

---

## ⚙️ Setup

### 1. Clone the repository

```bash
git clone https://github.com/Shakpro10/Machine-Learning-Projects.git
cd Machine-Learning-Projects
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

**Windows**

```bash
venv\Scripts\activate
```

**macOS / Linux**

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install numpy pandas matplotlib seaborn plotly scikit-learn statsmodels scipy \
            tensorflow tensorflow-datasets umap-learn pmdarima arch yfinance graphviz jupyter
```

Graphviz notebooks also require the [Graphviz system binaries](https://graphviz.org/download/).

### 4. Launch Jupyter

```bash
jupyter notebook
```

Open any notebook from the project folders. For the deep learning and LSTM notebooks, a GPU or [Google Colab](https://colab.research.google.com/) is recommended.

---

## 🛠️ Tech Stack

- **Language:** Python
- **Data handling:** [NumPy](https://numpy.org/), [pandas](https://pandas.pydata.org/)
- **Machine learning:** [scikit-learn](https://scikit-learn.org/)
- **Deep learning:** [TensorFlow](https://www.tensorflow.org/) / [Keras](https://keras.io/), TensorFlow Datasets
- **Statistical modelling:** [statsmodels](https://www.statsmodels.org/), [pmdarima](https://alkaline-ml.com/pmdarima/), [arch](https://arch.readthedocs.io/)
- **Dimensionality reduction:** scikit-learn (PCA, LDA, t-SNE), [UMAP](https://umap-learn.readthedocs.io/)
- **Visualization:** [Matplotlib](https://matplotlib.org/), [Seaborn](https://seaborn.pydata.org/), [Plotly](https://plotly.com/python/), Graphviz
- **Market data:** [yfinance](https://github.com/ranaroussi/yfinance)
- **Environment:** Jupyter Notebook, Google Colab

---

## ⚠️ Notes & Limitations

1. **Local file paths** — several notebooks read data from machine-specific paths (for example `C:\Users\...` or Google Drive on Colab). Update these paths to match your environment before running.
2. **Datasets not bundled** — see the [Datasets](#-datasets) section for where each one comes from.
3. **Reproducibility** — results reported here come from the saved notebook outputs and may vary slightly with library versions and random seeds.
4. **Library versions** — some code uses APIs that have changed in newer releases (for example `statsmodels` ARIMA imports, `sns.distplot`, and some scikit-learn parameter values), so you may need to adjust them on newer versions.
5. **Small-scale image data** — the cats-vs-dogs experiments use a 4,000-image subset to keep training practical.
6. **Educational scope** — these projects are designed to demonstrate methods and workflow, not as production systems.

---

## 📄 License

MIT License

Copyright (c) 2026 Shakiru Sikiru

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

## 👤 Author

**Shakiru Sikiru** — [GitHub: @Shakpro10](https://github.com/Shakpro10)

If you find these projects useful, consider giving the repository a ⭐.
