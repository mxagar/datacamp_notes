# XGBoost: A Guide

These are my personal notes of the Datacamp course [Extreme Gradient Boosting with XGBoost](https://app.datacamp.com/learn/courses/extreme-gradient-boosting-with-xgboost).

The course has 4 main sections:

1. Classification
2. Regression
3. Fine-tuning XGBoost
4. Using XGBoot in Pipelines

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
  - [2. Regression with XGBoost](#2-regression-with-xgboost)
    - [2.1 Non-Linear and Linear Weak Learners: Ames/Boston Housing Prediction](#21-non-linear-and-linear-weak-learners-amesboston-housing-prediction)
      - [2.1.1 Default Weak Learner (Trees, CART)](#211-default-weak-learner-trees-cart)
      - [2.1.2 Linear Weak Learner](#212-linear-weak-learner)
    - [2.2 Regression with Cross-Validation](#22-regression-with-cross-validation)
    - [2.3 Regression with Regularization](#23-regression-with-regularization)
    - [2.4 Visualizing Trees](#24-visualizing-trees)
    - [2.5 Feature Importances](#25-feature-importances)
  - [3. Fine-Tuning XGBoost](#3-fine-tuning-xgboost)


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
# Objective functions:
# reg:linear - regression (deprecated)
# reg:squarederror - regression
# reg:logistic - classification, class label output
# binary:logistic - classification, class probability output
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

# For tree visualization
pip install graphviz
```

### 1.2 How Does It Work?

XGBoost works with *weak* or individual base learners underneath; usually, these are **decision trees**, concretely **CARTs: Classification and Regression Trees**.

A decision tree is a binary tree where in each node a feature is used to split the dataset in two; that split is associated to a question. The leaves of the tree contain either a class or a value to be predicted. In particular, CARTs always contain a continuous value in the leaves, which can be used as a classifier value when a threshold is defined.

Therefore, XGBoost is an **ensemble learning** method: many models are used to yield a result. The underlying *weak* learners can be any algorithm, as mentioned, although CARTs are usually employed. The *weak* learner needs to be any model which is better than random chance, i.e., >50% accuracy in a binary classification. Then, the XGBoost converts those *weak* learners into **strong learners**: weak/bad effects cancel out and strong/good effects are highlighted.

*Weak* learners are trained with **boosting**:

- Iteratively learn models on subsets of data.
- Weight each weak prediction based on learner's performance.
- Combine weighted predictions to obtain a single prediction.

### 1.3 Cross Validation

We can use cross-validation with XGBoost, but the API usage is a bit different:

- We need to define `DMatrix` objects.
- We call `xgb.cv()`.

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
# Objective functions:
# reg:linear - regression (deprecated)
# reg:squarederror - regression
# reg:logistic - classification, class label output
# binary:logistic - classification, class probability output
params = {"objective":"binary:logistic",
          "max_depth":4}

# Fit with CV and get results of CV
# Parameters:
# https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.cv
cv_results = xgb.cv(dtrain=churn_dmatrix, # DMatrix
                    params=params, # parameters dictionary
                    nfold=4, # number of non-overlapping folds
                    num_boost_round=10, # number of trees
                    metrics="error", # error converts to accuracy; "rmse" or "mae" for regression (below)
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

## 2. Regression with XGBoost

In regression, continuous values are predicted.

Typical metrics for regression:

- Root mean square error (RMSE): `sqrt(sum((y_true-y_pred)^2))`: `"rmse"`.
- Mean absolute error (MAE): `mean(abs(y_true-y_pred))`: `"mae".`

The objective function in regression is `reg:linear` (deprecated) or `reg:squarederror`.

We can use as base learners two types of models:

- Trees, i.e., CARTs: `params["booster"] = "gbtree"`. **These are the learners by default, and are almost the unique ones that are used in practice.**
- Linear learners, i.e., linear models: : `params["booster"] = "gblinear"`.

The ensemble model is a weighted sum of the weak learners; if we use the linear learners, the final model is linear, if use trees as weak learners, the final model is non-linear.

Additionally, we can apply **regularization** to control the complexity of the model:

- `gamma`: minimum loss reduction allowed for a split to occur (refers to trees, I understand).
- `alpha`: L1 regularization on leaf weights, larger values mean more regularization (refers to linear models, I understand).
- `lambda`: L2 regularization on leaf weights (refers to linear models, I understand).

### 2.1 Non-Linear and Linear Weak Learners: Ames/Boston Housing Prediction

In this section, the Ames housing dataset is modeled using:

1. Trees as weak learners.
2. Linear weak learners.

#### 2.1.1 Default Weak Learner (Trees, CART)

```python
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

boston_data = pd.read_csv("../data/ames_housing_trimmed_processed.csv")
boston_data.shape # (1460, 57)

X, y = boston_data.iloc[:,:-1],boston_data.iloc[:,-1]
X_train, X_test, y_train, y_test= train_test_split(X,
                                                   y,
                                                   test_size=0.2,
                                                   random_state=123)

xg_reg = xgb.XGBRegressor(objective='reg:squarederror', # reg:linear
                          n_estimators=10,
                          seed=123)

xg_reg.fit(X_train, y_train)

preds = xg_reg.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test,preds))
print("RMSE: %f" % (rmse)) # 28106.463641
```

#### 2.1.2 Linear Weak Learner

If we want to use linear weak learners, we need to use the learning API, which is different:

- We need to define `DMatrix` objects.
- We call `xgb.train()`.

```python
# After the train/test split from previous example/section
# ...

# Convert to DMatrix: note that both X and y are in the matrix!
DM_train = xgb.DMatrix(data=X_train,label=y_train)
DM_test =  xgb.DMatrix(data=X_test,label=y_test)

# Define params
# The weak learner is defined with booster
# In this case, we use a linear base learner!
params = {"booster":"gblinear", # Trees are "gbtree"
          "objective":"reg:squarederror"} # linear
xg_reg = xgb.train(params=params, dtrain=DM_train, num_boost_round=10)

preds = xg_reg.predict(DM_test)

rmse = np.sqrt(mean_squared_error(y_test,preds))
print("RMSE: %f" % (rmse)) # 40824.166531
```

### 2.2 Regression with Cross-Validation

As for classification, we need to use the CV API with `DMatrix` objects.

```python
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

boston_data = pd.read_csv("../data/ames_housing_trimmed_processed.csv")

X, y = boston_data.iloc[:,:-1],boston_data.iloc[:,-1]

# Create the DMatrix: housing_dmatrix
housing_dmatrix = xgb.DMatrix(data=X, label=y)

# Create the parameter dictionary: params
params = {"objective":"reg:squarederror", "max_depth":4}

# Perform cross-validation: cv_results
cv_results = xgb.cv(dtrain=housing_dmatrix,
                    params=params,
                    nfold=4,
                    num_boost_round=5,
                    metrics="rmse", # or "mae"
                    as_pandas=True,
                    seed=123)

# Print cv_results
print(cv_results)

# Extract and print final boosting round metric
print((cv_results["test-rmse-mean"]).tail(1))
```

### 2.3 Regression with Regularization

Recall that we can apply **regularization** to control the complexity of the model:

- `gamma`: minimum loss reduction allowed for a split to occur (refers to trees, I understand).
- `alpha`: L1 regularization on leaf weights, larger values mean more regularization (refers to linear models, I understand).
- `lambda`: L2 regularization on leaf weights (refers to linear models, I understand).

```python
import xgboost as xgb
import pandas as pd

boston_data = pd.read_csv("../data/ames_housing_trimmed_processed.csv")
X,y = boston_data.iloc[:,:-1],boston_data.iloc[:,-1]
boston_dmatrix = xgb.DMatrix(data=X,label=y)

params={"objective":"reg:squarederror","max_depth":4}
# L1 regularization values to test
l1_params = [1,10,100]
# Results of each regularization parameter
rmses_l1 = []
# CV with each L1 param
for reg in l1_params:
    # Define L1 regularization param alpha
    params["alpha"] = reg
    # We perform CV
    cv_results = xgb.cv(dtrain=boston_dmatrix,
                        params=params,
                        nfold=4,
                        num_boost_round=10,
                        metrics="rmse",
                        as_pandas=True,
                        seed=123)
    # Store result metric
    rmses_l1.append(cv_results["test-rmse-mean"].tail(1).values[0])

print("Best rmse as a function of l1:")
print(pd.DataFrame(list(zip(l1_params,rmses_l1)), columns=["l1","rmse"]))
```

### 2.4 Visualizing Trees

If we use the XGBoost learning API (with `DMatrix` and `xgb.train()`) we can visualize the trees under the hood using `plot_tree()`. Note: we need to `pip install graphviz` if not done yet.

```python
import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt

boston_data = pd.read_csv("../data/ames_housing_trimmed_processed.csv")
X,y = boston_data.iloc[:,:-1],boston_data.iloc[:,-1]

# Create the DMatrix: housing_dmatrix
housing_dmatrix = xgb.DMatrix(data=X, label=y)

# Create the parameter dictionary: params
params = {"objective":"reg:squarederror", "max_depth":2}

# Train the model: xg_reg
xg_reg = xgb.train(params=params,
                   dtrain=housing_dmatrix,
                   num_boost_round=10) # 10 trees in total

# Plot the first tree
# num_trees refers to the tree, starting with 0
xgb.plot_tree(xg_reg, num_trees=0)
plt.show()

# Plot the fifth tree
xgb.plot_tree(xg_reg, num_trees=4)
plt.show()

# Plot the last tree sideways
xgb.plot_tree(xg_reg, num_trees=9, rankdir="LR")
plt.show()
```

### 2.5 Feature Importances

One way of measuring the importance of a feature is counting the number of times each feature is split on across all boosting rounds (trees) in the model. We can visualize the results with `plot_importance()`. For that, we need to use the XGBoost learning API (with `DMatrix` and `xgb.train()`).

```python
import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt

boston_data = pd.read_csv("../data/ames_housing_trimmed_processed.csv")
X,y = boston_data.iloc[:,:-1],boston_data.iloc[:,-1]

# Create the DMatrix: housing_dmatrix
housing_dmatrix = xgb.DMatrix(data=X, label=y)

# Create the DMatrix: housing_dmatrix
housing_dmatrix = xgb.DMatrix(data=X, label=y)

# Create the parameter dictionary: params
params = {"objective":"reg:squarederror", "max_depth":4}

# Train the model: xg_reg
xg_reg = xgb.train(params=params,
                   dtrain=housing_dmatrix,
                   num_boost_round=10) # 10 trees in total

# Plot the feature importances
xgb.plot_importance(xg_reg)
plt.show()
```

## 3. Fine-Tuning XGBoost

