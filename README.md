# 🧪 Two-Layer Test Case Prioritization (TCP) Framework

This project proposes a dynamic, multi-objective, and hybrid **Two-Layer Test Case Prioritization (TCP)** system. It integrates supervised machine learning models with clustering and reinforcement learning to enhance fault detection efficiency in Continuous Integration (CI) environments.

---

## 📌 Key Features

- **Primary Layer**: Supervised learning models (KNN, SVM, Random Forest, Decision Tree)
- **Secondary Layer**: Reinforcement learning (Q-Learning) and clustering (K-Means)
- **Multi-objective Optimization**: Combines APFD, Accuracy, and Feature Importance
- **Hyperparameter Tuning**: GridSearchCV and ParameterSampler for optimal configurations
- **Custom Evaluation**: APFD, Silhouette Score, CFD, Cost per Fault, and execution time

---

## 🧠 Frameworks Included

| Framework | Primary Layer                    | Secondary Layer       |
|----------|----------------------------------|------------------------|
| 1        | KNN + SVM (Hybrid)               | Optimized K-Means     |
| 2        | KNN + SVM (Hybrid)               | Q-Learning            |
| 3        | Multi-Objective Optimization     | Optimized K-Means     |
| 4        | Multi-Objective Optimization     | Q-Learning            |

---

## 🏗️ Architecture Overview

### Primary Layer
- Learns static test case features (e.g., `Duration`, `LastResults_Mean`, `LastResults_Length`)
- Produces hybrid scores from SVM and KNN (or optimized individual models)
- Evaluated using **Accuracy**, **APFD**, and **Feature Importances**

### Secondary Layer
- Operates on dynamic features (e.g., test cycle performance)
- Uses either:
  - **Optimized K-Means** for clustering test cases by risk
  - **Q-Learning** for reward-driven prioritization based on fault density and hybrid score

---

## 📊 Evaluation Metrics

- **APFD (Average Percentage of Faults Detected)**
- **Accuracy** (classification performance)
- **Silhouette Score** (for K-Means clustering)
- **CFD (Cumulative Fault Detection)**
- **Execution Time** (benchmarking each layer)
- **Cost Per Fault Detected**

---

## 📋 Experimental Results

### 🔹 Primary Layer – Baseline Models

| Model           | Accuracy | APFD   |
|----------------|----------|--------|
| KNN            | 0.9961   | 0.5057 |
| Decision Tree  | 0.9963   | 0.5076 |
| Random Forest  | 0.9960   | 0.5051 |
| SVM            | 0.9980   | 0.5052 |

---

### 🔹 Primary Layer – Optimized Models

| Model           | Accuracy | APFD   | Feature Importances (RF only) |
|----------------|----------|--------|-------------------------------|
| KNN            | 0.9964   | 0.5052 | –                             |
| Decision Tree  | 0.9968   | 0.5048 | –                             |
| Random Forest  | 0.9960   | 0.5050 | `[0.031, 0.0, 0.959, 0.009]`  |
| SVM            | 0.9980   | 0.5052 | –                             |

---

### 🔸 Secondary Layer – Baseline & Optimized

#### 🔹 K-Means Clustering

| Type       | APFD   | Exec Time (s) | Cost/Fault | Clusters | CFD 50% | CFD 100% |
|------------|--------|---------------|------------|----------|---------|----------|
| Baseline   | 0.1142 | 0.4405        | 0.0000     | 2        | 0.01    | 1.0      |
| Optimized  | 0.1142 | 358.6638      | 0.0262     | 4        | 0.56    | 1.0      |

#### 🔹 Q-Learning

| Type       | APFD   | Exec Time (s) | Cost/Fault | CFD 50% | CFD 100% |
|------------|--------|---------------|------------|---------|----------|
| Baseline   | 0.4918 | 1146.3842     | 0.0838     | 0.49    | 1.0      |
| Optimized  | 0.4806 | 24989.5763    | 1.8275     | 0.47    | 1.0      |

---

## 🧪 Framework Comparisons

### ⚫ **Framework 1** – Hybrid (KNN + SVM) with Optimized K-Means

- **Primary Layer**: Hybrid Accuracy: 0.9971, APFD: 0.5049  
- **Secondary Layer**: Accuracy: 0.9980, Best K: 4, Silhouette: 0.6836  
- **Final**: Accuracy: 0.9975, APFD: 0.4246  
- **Top 10 Rankings**: `[910, 913, 914, 916, 918, 922, 929, 930, 931, 932]`

---

### ⚫ **Framework 2** – Hybrid (KNN + SVM) with Q-Learning

- **Primary Layer**: Hybrid Accuracy: 0.9971, APFD: 0.5049  
- **Secondary Layer**: Accuracy: 0.6927  
  - Params: alpha=0.01, gamma=0.8, epsilon_decay=0.9995  
- **Final**: Accuracy: 0.8086, APFD: 0.9917  
- **Top 10 Rankings**: `[22202, 59754, 60982, 63008, 19757, 53632, 39594, 52031, 24886, 42675]`

---

### ⚫ **Framework 3** – Multi-Objective Optimization with Optimized K-Means

- **Primary Layer**: Hybrid Accuracy: 0.9968, APFD: 0.5049  
  - KNN Best Params: `{'n_neighbors': 10, 'weights': 'distance', 'metric': 'euclidean'}`  
  - SVM Best Params: `{'C': 0.01, 'gamma': 0.1, 'kernel': 'rbf'}`  
- **Secondary Layer**: Accuracy: 0.9980, Best K: 4, Silhouette: 0.6836  
- **Final**: Accuracy: 0.9974, APFD: 0.6457  
- **Top 10 Rankings**: `[910, 913, 914, 916, 918, 922, 929, 930, 931, 932]`

---

### ⚫ Framework 4 – Multi-Objective Optimization + Q-Learning

- **Primary Layer**: Hybrid Accuracy: 0.9968, APFD: 0.5049  
  - KNN Best Params: `{'n_neighbors': 10, 'weights': 'distance', 'metric': 'euclidean'}`  
  - SVM Best Params: `{'C': 0.01, 'gamma': 0.1, 'kernel': 'rbf'}`  
- **Secondary Layer**: Accuracy: 0.6965  
  - Params: alpha=0.0538, gamma=0.9709, epsilon_decay=0.9682  
- **Final**: Accuracy: 0.8141, APFD: 0.9771  
- **Top 10 Rankings**: `[8474, 8475, 8476, 8537, 8577, 8596, 8627, 8628, 8630, 8631]`


## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- `scikit-learn`, `pandas`, `numpy`, `imblearn`  
- A test case dataset

### Installation

```bash
git clone https://github.com/fardinik98/Two_Layer_TCP.git
cd Two_Layer_TCP
pip install -r requirements.txt
python main.py
```
## 📬 Feedback

If you have any feedback or suggestions, feel free to reach out via **[Fardin Islam Khan](https://github.com/fardinik98)**, or create an issue in this repository.

---

## 👥 Authors
- **Fardin Islam Khan** – [GitHub](https://github.com/fardinik98)

