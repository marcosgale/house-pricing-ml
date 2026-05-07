# 🏠 House Price Prediction — Full ML Pipeline

A complete machine learning pipeline for house price prediction, built as part of **CS411 at Southern New Hampshire University**. The project benchmarks manual models against AutoML across two real-world datasets of different complexity.

---

## 📌 Project Overview

This notebook walks through every stage of a production-style ML workflow:

- **Exploratory data analysis** and outlier removal
- **Feature engineering** and missing value imputation
- **Data leakage prevention** (removal of `price_per_sqft`)
- **Manual model training** with scikit-learn (Linear Regression & Gradient Boosting)
- **AutoML benchmarking** with PyCaret (`compare_models` + `tune_model`)
- **Model evaluation** using residual plots, prediction error plots, and feature importance

---

## 📊 Datasets

| Dataset | Description | Size |
|---|---|---|
| **SimpleHouse.csv** | Residential listings with price as target | ~117 MB |
| **ComplexHouse.csv** | High-dimensional property data with `ValueofHome` as target | ~42 MB |

> ⚠️ **Datasets are not included** in this repository due to their size. Place them inside a `data/` folder to run the notebook locally.

Three dataset versions were maintained throughout the pipeline:
- `Simple` — cleaned Kaggle residential dataset
- `Complex Raw` — minimal preprocessing
- `Complex Cleaned` — 51 irrelevant columns dropped, outliers removed

---

## 📈 Results

### Manual Models — sklearn

| Model | Dataset | MAE | RMSE | R² |
|---|---|---|---|---|
| Linear Regression | Simple | $130,182 | $188,166 | 0.337 |
| Gradient Boosting | Simple | $126,701 | $168,695 | **0.467** |
| Ridge Regression | Complex Cleaned | $92,831 | $196,970 | 0.358 |
| Gradient Boosting | Complex Cleaned | $80,875 | $178,488 | **0.473** |

> Gradient Boosting consistently outperformed linear baselines on both datasets.

### AutoML — PyCaret

- Used `compare_models()` to benchmark all viable regression algorithms with **5-fold cross-validation**
- Best model per dataset tuned with `tune_model(optimize='MAE')`
- Evaluated with **residual plots**, **prediction error plots**, and **feature importance charts**

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![PyCaret](https://img.shields.io/badge/PyCaret-3178C6?style=flat)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat)

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/cbarney23/house-pricing-ml.git
cd house-pricing-ml
```

### 2. Install dependencies
```bash
pip install pandas numpy scikit-learn pycaret matplotlib
```

### 3. Add the datasets
Place `SimpleHouse.csv` and `ComplexHouse.csv` inside a `data/` folder:
```
house-pricing-ml/
├── data/
│   ├── SimpleHouse.csv
│   └── ComplexHouse.csv
└── House_Pricing_Submit.ipynb
```

### 4. Open and run the notebook
```bash
jupyter notebook House_Pricing_Submit.ipynb
```

---

## 📂 Repository Structure

```
house-pricing-ml/
├── House_Pricing_Submit.ipynb   # Final submission notebook
├── House_Pricing.ipynb          # Working/draft notebook
├── .gitignore                   # Excludes data files and logs
└── README.md                    # This file
```

---

## 🧠 Key Concepts Demonstrated

- **Outlier detection and removal** — filtered prices ≤ $10,000
- **Missing value imputation** — median/mode strategy per column type
- **One-hot encoding** — categorical feature transformation
- **Feature selection** — dropped 51 low-value columns from complex dataset
- **Data leakage prevention** — excluded `price_per_sqft` (encodes the target)
- **Train/Validation/Test split** — 80/20 with stratified sampling
- **Model comparison** — MAE, RMSE, and R² across 4 model-dataset combinations
- **AutoML** — PyCaret pipeline with cross-validation and hyperparameter tuning

---

## 📝 Academic Context

**Course:** CS411 — Applied Machine Learning  
**Institution:** Southern New Hampshire University (SNHU)  
**Language:** Python 3 | Jupyter Notebook
