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
    - [1.3 Cross Validation](#13-cross-validation)
    - [1.4 When to Use XGBoost](#14-when-to-use-xgboost)


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
class_data = pd.read_csv("../data/ChurnData.csv")
X, y = class_data.iloc[:,:-1], class_data.iloc[:,-1].astype(int)
X_train, X_test, y_train, y_test= train_test_split(X, y,
        test_size=0.2, random_state=123)

# XGBoost Classifier instance
# Parameters:
# https://xgboost.readthedocs.io/en/stable/parameter.html
xg_cl = xgb.XGBClassifier(objective='binary:logistic',
                          n_estimators=10,
                          seed=123)

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

XGBoost works with *weak* or individual base learners underneath; usually, these are **decision trees**, concretely **CARTs: Classification and Regression Trees**.

A decision tree is a binary tree where in each node a feature is used to split the dataset in two; that split is associated to a question. The leaves of the tree contain either a class or a value to be predicted. In particular, CARTs always contain a continuous value in the leaves, which can be used as a classifier value when a threshold is defined.

Therefore, XGBoost is an **ensemble learning** method: many models are used to yield a result. The underlying *weak* learners can be any algorithm, as mentioned, although CARTs are usually employed. The *weak* learner needs to be any model which is better than random chance, i.e., >50% accuracy in a binary classification. Then, the XGBoost converts those *weak* learners into **strong learners**.

*Weak* learners are trained with **boosting**:

- Iteratively learn models on subsets of data.
- Weight each weak prediction based on learner's performance.
- Combine weighted predictions to obtain a single prediction.

### 1.3 Cross Validation

We can use cross-validation with XGBoost, but the API usage is a bit different.

```python
import xgboost as xgb
import pandas as pd

churn_data = pd.read_csv("../data/ChurnData.csv")

# DMatrix is a specific data structure which accelerates the computations
# In the regular API, i.e., without cross-validation, DMatrix is automatically generated
# but with cross-validation we need to do it explicitly
churn_dmatrix = xgb.DMatrix(data=churn_data.iloc[:,:-1], # X
                            label=churn_data.churn) # y

# Define params
params = {"objective":"binary:logistic",
          "max_depth":4}

# Fit with CV and get results of CV
# Parameters:
# https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.cv
cv_results = xgb.cv(dtrain=churn_dmatrix, # DMatrix
                    params=params, # parameters dictionary
                    nfold=4, # number of non-overlapping folds
                    num_boost_round=10, # number of trees
                    metrics="error", # error converts to accuracy
                    as_pandas=True) # if we want results as a pandas object

print("Accuracy: %f" %((1-cv_results["test-error-mean"]).iloc[-1])) # 0.7

# Perform cross-validation with another metric: AUC
cv_results = xgb.cv(dtrain=churn_dmatrix,
                    params=params, 
                    nfold=3,
                    num_boost_round=5, 
                    metrics="auc",
                    as_pandas=True,
                    seed=123)

# Print cv_results
print(cv_results)
#    train-auc-mean  train-auc-std  test-auc-mean  test-auc-std
# 0        0.907307       0.025788       0.694683      0.057410
# 1        0.951466       0.017800       0.720245      0.032604
# 2        0.975673       0.009259       0.722732      0.018837
# 3        0.982302       0.006991       0.735959      0.038124
# 4        0.988113       0.005642       0.732957      0.040420

# Print the AUC
print((cv_results["test-auc-mean"]).iloc[-1]) # 0.732957


```

### 1.4 When to Use XGBoost

We can use XGBoost:

- With large datasets:
  - 1000 data-point of less than 100 features each,
  - however, as long as the number of features < the number of data-points, everything should be fine.
- With numerical or categorical features, or a mixture of both.

XGBoost is suboptimal if:

- Computer vision, Image recognition (better use deep learning)
- NLP (better use deep learning)
- When the number of features is larger than the number of data-points.