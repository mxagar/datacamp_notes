# XGBoost: A Guide

These are my personal notes of the Datacamp course [Extreme Gradient Boosting with XGBoost](https://app.datacamp.com/learn/courses/extreme-gradient-boosting-with-xgboost).

Note that:

- Images are in [`pics`](pics).
- The code/exercises are in [`lab`](lab).
- The data is in [`data`](data).

Table of contents:

- [XGBoost: A Guide](#xgboost-a-guide)
  - [1. Classification with XGBoost](#1-classification-with-xgboost)
    - [1.1 Introduction](#11-introduction)
      - [Example Snippet](#example-snippet)
      - [Installation](#installation)
    - [1.2 How Does It Work?](#12-how-does-it-work)


Mikel Sagardia, 2023.  
No guarantees.

## 1. Classification with XGBoost

Classification is the original supervised learning problem addressed by XGBoost, although it can also handle regression problems.

### 1.1 Introduction

XGBoost is an implementation of the [Gradient Boosting](https://en.wikipedia.org/wiki/Gradient_boosting) algorithm in C++ which has bindings to other languages, such as Python. It has the following properties:

- Fast.
- Best performance.
- Parallelizable, on a computer and across the network. So it can work with huge datasets distributed on several nodes/GPUs.
- We can use it for classification and regression.
- The Python API is analog to the Scikit-Learn API, i.e., we have `fit()` and `predict()` methods.

#### Example Snippet

```python
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Get data, split
class_data = pd.read_csv("classification_data.csv")
X, y = class_data.iloc[:,:-1], class_data.iloc[:,-1].astype(int)
X_train, X_test, y_train, y_test= train_test_split(X, y,
        test_size=0.2, random_state=123)

# XGBoost Classifier instance
xg_cl = xgb.XGBClassifier(objective='binary:logistic',
        n_estimators=10, seed=123)

# Train/Fit
xg_cl.fit(X_train, y_train)

# Predict
preds = xg_cl.predict(X_test)

# Evaluate
accuracy = float(np.sum(preds==y_test))/y_test.shape[0]
print("Accuracy: %f" % (accuracy))
```

#### Installation

```bash
# PIP
pip install xgboost

# Conda: General
conda install -c conda-forge py-xgboost

# Conda: CPU only
conda install -c conda-forge py-xgboost-cpu

# Conda: Use NVIDIA GPU: Linux x86_64
conda install -c conda-forge py-xgboost-gpu
```

### 1.2 How Does It Work?


