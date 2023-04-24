# XGBoost: A Guide

These are my personal notes of the Datacamp course [Extreme Gradient Boosting with XGBoost](https://app.datacamp.com/learn/courses/extreme-gradient-boosting-with-xgboost).

The course has 4 main sections:

1. Classification
2. Regression
3. Fine-tuning XGBoost
4. Using XGBoost in Pipelines

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
    - [3.1 Manual Hyperparameter Selection](#31-manual-hyperparameter-selection)
      - [Example of Manual Parameter Selection](#example-of-manual-parameter-selection)
      - [Effect of Varying a Hyperparameter: Number of Boosting Rounds](#effect-of-varying-a-hyperparameter-number-of-boosting-rounds)
      - [Automated Selection with Early Stopping](#automated-selection-with-early-stopping)
    - [3.2 Most Common Tunable Hyperparameters](#32-most-common-tunable-hyperparameters)
      - [Example: Variation of the Learning Rate, Max Depth, Number of Features](#example-variation-of-the-learning-rate-max-depth-number-of-features)
    - [3.3 Grid Search and Random Search](#33-grid-search-and-random-search)
      - [Grid Search](#grid-search)
      - [Random Search](#random-search)
  - [4. Using XGBoost in Pipelines](#4-using-xgboost-in-pipelines)
    - [4.1 Example of Pipelines](#41-example-of-pipelines)
    - [4.2 Data Processing in Pipelines](#42-data-processing-in-pipelines)
    - [4.3 Full Pipeline Example: Kidney Disease Dataset](#43-full-pipeline-example-kidney-disease-dataset)
    - [4.4 Housing Pipeline with Random Search Cross-Validation](#44-housing-pipeline-with-random-search-cross-validation)


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
- The [Python API](https://xgboost.readthedocs.io/en/stable/python/python_api.html) is easy to use and has two major flavors or sub-APIs:
  - The **Scikit-Learn API**: We instantiate `XGBRegressor()` or `XGBClassifier` and then we can `fit()` and `predict()`, using the typical Scikit-Learn parameters; we can even use those objects with other Scikit-Learn modules, such as `GridSearchCV`.
  - The **Learning API**: The native XGBoost Python API requires to convert the dataframes into `DMatrix` objects first; then, we have powerful methods which allow for tuning many parameters: `xgb.cv()`, `xgb.train()`. The native/learning API is very easy to use. **Note: the parameter names are different compared to the Scikit-Learn API!**

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

# XGBoost Classifier instance: Scikit-Learn API
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

The XGBoost implementation allows two weak learners:

- The mentioned CART trees; these should be used in most cases, because they capture non-linearities.
- Linear learners.

Notes on the general boosting algorithm:

- Each weak learner is created in a boosting round and it uses a subset of the total dataset.
- If we use trees, we can select the number of features to be selected randomly to build the tree.
- We can apply regularization is the model is overfitting.

### 1.3 Cross Validation

We can use cross-validation with XGBoost, but the API usage is a bit different, i.e., we use the XGBoost native/learning API:

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
# XGBoost native/learning API
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

- `gamma`: minimum loss reduction allowed for a split to occur.
- `alpha`: L1 regularization on (leaf) weights, larger values mean more regularization.
- `lambda`: L2 regularization on (leaf) weights.

Apparently, all these regularization parameters can be used with both base/weak learners; however, their interpretation is different in each case:

1. Decision Trees (`"gbtree"` base learner):
  - Gamma: Controls the minimum loss reduction required to make a split in a decision tree. A higher value of gamma makes the algorithm more conservative and reduces the number of splits.
  - Alpha: L1 regularization parameter on leaf weights. A higher value of alpha makes the leaf weights closer to zero, resulting in a more sparse tree.
  - Lambda: L2 regularization parameter on leaf weights. A higher value of lambda makes the leaf weights smaller, resulting in a smoother tree.

2. Linear Models (`"gblinear"` base learner):
  - Gamma: Controls the L1 regularization strength on the weights. A higher value of gamma results in more sparsity in the weight matrix.
  - Alpha: L1 regularization parameter on the bias. A higher value of alpha makes the bias closer to zero, resulting in a more sparse model.
  - Lambda: L2 regularization parameter on the weights. A higher value of lambda makes the weights smaller, resulting in a smoother model.

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

# Scikit-Learn API
xg_reg = xgb.XGBRegressor(objective='reg:squarederror', # reg:linear
                          n_estimators=10,
                          seed=123)

xg_reg.fit(X_train, y_train)

preds = xg_reg.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test,preds))
print("RMSE: %f" % (rmse)) # 28106.463641
```

#### 2.1.2 Linear Weak Learner

If we want to use linear weak learners, we need to use the learning or XGBoost native/learning API, which is different:

- We need to define `DMatrix` objects.
- We call `xgb.train()`.

```python
# After the train/test split from previous example/section
# ...

# Convert to DMatrix: note that both X and y are in the matrix!
DM_train = xgb.DMatrix(data=X_train,label=y_train)
DM_test =  xgb.DMatrix(data=X_test,label=y_test)

# XGBoost native/learning API
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

# XGBoost native/learning API
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

- `gamma`: minimum loss reduction allowed for a split to occur.
- `alpha`: L1 regularization on (leaf) weights, larger values mean more regularization.
- `lambda`: L2 regularization on (leaf) weights.

Apparently, all these regularization parameters can be used with both base/weak learners; however, their interpretation is different in each case:

1. Decision Trees (`"gbtree"` base learner):
  - Gamma: Controls the minimum loss reduction required to make a split in a decision tree. A higher value of gamma makes the algorithm more conservative and reduces the number of splits.
  - Alpha: L1 regularization parameter on leaf weights. A higher value of alpha makes the leaf weights closer to zero, resulting in a more sparse tree.
  - Lambda: L2 regularization parameter on leaf weights. A higher value of lambda makes the leaf weights smaller, resulting in a smoother tree.

2. Linear Models (`"gblinear"` base learner):
  - Gamma: Controls the L1 regularization strength on the weights. A higher value of gamma results in more sparsity in the weight matrix.
  - Alpha: L1 regularization parameter on the bias. A higher value of alpha makes the bias closer to zero, resulting in a more sparse model.
  - Lambda: L2 regularization parameter on the weights. A higher value of lambda makes the weights smaller, resulting in a smoother model.

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

In general, XGBoost hyperparameters are modifiable via the `params` dictionary. Also, note that the parameters in the `cv()` or `train()` APIs are also hyperparameters! Especially, the `num_boost_round` value, which specifies the number of weak learners or boosting rounds is essential. We can also use the Scikit-Learn API; in that case, we can take advantage of `GridSearchCV` or `RandomSearchCV`, but note that the name of the parameters might change, e.g., `num_boost_round` becomes `n_estimators`. For more, check the documentation: [Python API Reference](https://xgboost.readthedocs.io/en/stable/python/python_api.html).

### 3.1 Manual Hyperparameter Selection

Manual or systematic parameter tuning can significantly improve the results, but it can be also time consuming. So we need to choose according to the application.

#### Example of Manual Parameter Selection

```python
import pandas as pd
import xgboost as xgb
import numpy as np

housing_data = pd.read_csv("../data/ames_housing_trimmed_processed.csv")
X = housing_data[housing_data.columns.tolist()[:-1]]
y = housing_data[housing_data.columns.tolist()[-1]]
housing_dmatrix = xgb.DMatrix(data=X,label=y)

# Manually set paramater values (not default ones)
params = {"objective":"reg:squarederror",
          'colsample_bytree': 0.3,
          'learning_rate': 0.1,
          'max_depth': 5}
cv_results_rmse = xgb.cv(dtrain=housing_dmatrix,
                         params=params,
                         nfold=4,
                         num_boost_round=200, # THIS IS ALSO A PARAM!
                         metrics="rmse",
                         as_pandas=True,
                         seed=123)

print("Tuned rmse: %f" %((tuned_cv_results_rmse["test-rmse-mean"]).tail(1))) # 30370
```

#### Effect of Varying a Hyperparameter: Number of Boosting Rounds

In this example, we try different number of boosting rounds, i.e., `num_boost_round`; these denote the number of weak learners under the hood. Note that we can use a loop with any kind of hyperparameter.

```python
# Create list of number of boosting rounds
num_rounds = [150, 200, 250]

# Empty list to store final round rmse per XGBoost model
final_rmse_per_round = []

# Iterate over num_rounds and build one model per num_boost_round parameter
for curr_num_rounds in num_rounds:

    # Perform cross-validation: cv_results
    cv_results = xgb.cv(dtrain=housing_dmatrix,
                        params=params,
                        nfold=3,
                        num_boost_round=curr_num_rounds, # Several values tried in loop
                        metrics="rmse",
                        as_pandas=True,
                        seed=123)
    
    # Append final round RMSE
    final_rmse_per_round.append(cv_results["test-rmse-mean"].tail().values[-1])

# Print the resultant DataFrame
num_rounds_rmses = list(zip(num_rounds, final_rmse_per_round))
print(pd.DataFrame(num_rounds_rmses,columns=["num_boosting_rounds","rmse"]))
#    num_boosting_rounds          rmse
# 0                  150  29763.123698
# 1                  200  29634.996745
# 2                  250  29639.554036
```

#### Automated Selection with Early Stopping

We can activate activate early stopping with `early_stopping_rounds`: boosting rounds can be stopped before completing the total number of boosting rounds given with `num_boost_round`. The validation metric needs to improve at least once in every `early_stopping_rounds` round(s) to avoid stopping.

```python
# Create the parameter dictionary for each tree: params
params = {"objective":"reg:squarederror", "max_depth":4}

# Perform cross-validation with early stopping: cv_results
cv_results = xgb.cv(dtrain=housing_dmatrix,
                         params=params,
                         nfold=3,
                         num_boost_round=50,
                         early_stopping_rounds=10,
                         metrics="rmse",
                         as_pandas=True,
                         seed=123)

# Print cv_results
# We see the results for each boosting round
print(cv_results)
#     train-rmse-mean  train-rmse-std  test-rmse-mean  test-rmse-std
# 0     141871.635417      403.636200   142640.651042     705.559164
# 1     103057.036458       73.769561   104907.664062     111.112417
# ...
```

### 3.2 Most Common Tunable Hyperparameters

**IMPORTANT**: Have a look at the [Python API](https://xgboost.readthedocs.io/en/stable/python/python_api.html) to see all the parameters for any API (i.e., Scikit-Learn or Learning). In the following, the most common parameters are listed.

Tree weak learner:

- `eta` or `learning_rate`: how quickly we fit the residual error. High values lead to quicker fits.
- `gamma`: min loss reduction to create new tree split. Higher value, less splits, less complexity, less overfitting.
- `lambda`: L2 reg on leaf weights. Higher value, less complexity.
- `alpha`: L1 reg on leaf weights. Higher value, less complexity.
- `max_depth`: max depth per tree; how deep each tree is allowed to grow in each round. Higher value, **more** complexity.
- `subsample`: fraction of total samples used per tree; in each boosting round, a tree takes one subset of all data points, this value refers to the size of this subset. Higher value, **more** complexity. 
- `colsample_bytree`: fraction of features used per each tree or boosting round. Not all features need to be used by each weak learner or boosting round. This value refers to how many from the total amount are used, selected randomly. A low value of this parameter is like more regularization.

Linear weak learner (much less hyperparameters):

- `lambda`: L2 reg on weights. Higher value, less complexity.
- `alpha`: L1 reg on weights. Higher value, less complexity.
- `lambda_bias`: L2 reg term on bias. Higher value, less complexity.

For any type base/weak learner, recall that we can tune the number of boostings or weak learners we want in the `cv()` or `train()` call:

- `num_boost_round`
- `early_stopping_rounds`

#### Example: Variation of the Learning Rate, Max Depth, Number of Features

```python
# Create your housing DMatrix: housing_dmatrix
housing_dmatrix = xgb.DMatrix(data=X, label=y)

# Create the parameter dictionary for each tree (boosting round)
params = {"objective":"reg:squarederror", "max_depth":3}

# Create list of eta values and empty list to store final round rmse per xgboost model
vals = [0.001, 0.01, 0.1] # eta
#vals = [2, 5, 10, 20] # max_depth
#vals = [0.1, 0.5, 0.8, 1] # colsample_bytree
best_rmse = []

# Systematically vary the eta 
for curr_val in vals:

    params["eta"] = curr_val
    #params["max_depth"] = curr_val
    #params["colsample_bytree"] = curr_val
    
    # Perform cross-validation: cv_results
    cv_results = xgb.cv(dtrain=housing_dmatrix,
                         params=params,
                         nfold=3,
                         num_boost_round=10,
                         early_stopping_rounds=5,
                         metrics="rmse",
                         as_pandas=True,
                         seed=123)

    # Append the final round rmse to best_rmse
    best_rmse.append(cv_results["test-rmse-mean"].tail().values[-1])

# Print the resultant DataFrame
print(pd.DataFrame(list(zip(eta_vals, best_rmse)), columns=["eta","best_rmse"]))
#      eta      best_rmse
# 0  0.001  195736.411458
# 1  0.010  179932.192708
# 2  0.100   79759.411458
```

### 3.3 Grid Search and Random Search

We can use `GridSearchCV` and `RandomSearchCV` from Scikit-Learn to systematically obtain the best parameters. To that end, we need to use the Scikit-Learn API, i.e., we instantiate `XGBRegressor` or `XGBClassifier` and user the parameters typical from Scikit-Learn. Note that the parameter seach space increases exponentially as we add parameters, so:

- With `GridSearchCV` we might require much more time to find the optimum parameter set.
- With `RandomSearchCV` we limit the number of sets, but these are random!

There are more advanced techniques for hyperparameter tuning, such as [Bayesian hyperparameter optimization](https://machinelearningmastery.com/scikit-optimize-for-hyperparameter-tuning-in-machine-learning/).

#### Grid Search

```python
import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

housing_data = pd.read_csv("../data/ames_housing_trimmed_processed.csv")
X = housing_data[housing_data.columns.tolist()[:-1]]
y = housing_data[housing_data.columns.tolist()[-1]]
housing_dmatrix = xgb.DMatrix(data=X,label=y)

# The parameter names with the Scikit-Learn API are different
# eta -> learning_rate
# num_boost_round -> n_estimators
gbm_param_grid = {'learning_rate': [0.01,0.1,0.5,0.9],
                  'n_estimators': [200],
                  'subsample': [0.3, 0.5, 0.9]}

gbm = xgb.XGBRegressor()
grid_mse = GridSearchCV(estimator=gbm,
                        param_grid=gbm_param_grid,
                        scoring='neg_mean_squared_error', # negative MSE
                        cv=4,
                        verbose=1)

grid_mse.fit(X, y)

print("Best parameters found: ", grid_mse.best_params_)
# Since we have the negative MSE, we need to compute the RMSE from it
print("Lowest RMSE found: ", np.sqrt(np.abs(grid_mse.best_score_)))
```

#### Random Search

We define the possible hyperparameter values but unlike in the grid search, here we define a number of possible combinations to be tested. Then, for each trial, the hyperparameter values are chosen randomly.

```python
# All possible combinations are: 20 * 1 * 20 = 400
# BUT: we limit to n_iter=25 the number of combinations
# And we will train each of them 4-fold with CV
gbm_param_grid = {'learning_rate': np.arange(0.05,1.05,.05), # arange: 20 values
                  'n_estimators': [200],
                  'subsample': np.arange(0.05,1.05,.05)} # arange: 20 values

gbm = xgb.XGBRegressor()
randomized_mse = RandomizedSearchCV(estimator=gbm,
                                    param_distributions=gbm_param_grid,
                                    n_iter=25, # number of combinations
                                    scoring='neg_mean_squared_error',
                                    cv=4,
                                    verbose=1)

randomized_mse.fit(X, y)
print("Best parameters found: ",randomized_mse.best_params_)
print("Lowest RMSE found: ", np.sqrt(np.abs(randomized_mse.best_score_)))
```

## 4. Using XGBoost in Pipelines

This section is not that relevant: it is not that XGBoost specific, just a generic section on how to build `Pipeline` objects, but using XGBoost models. The main message is that we need to use the Scikit-Learn API to be able to build `Pipelines`.

### 4.1 Example of Pipelines

```python
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

data = pd.read_csv("../data/ames_housing_trimmed_processed.csv")
X = data.iloc[:,:-1]
y = data.iloc[:,-1]

rf_pipeline = Pipeline([("st_scaler", StandardScaler()),
                        ("rf_model", RandomForestRegressor())])

scores = cross_val_score(rf_pipeline,
                         X,
                         y,
                         scoring="neg_mean_squared_error",
                         cv=10)
```

### 4.2 Data Processing in Pipelines

```python
df = pd.read_csv("../data/ames_unprocessed_data.csv")

# There are some NAs
df.isna().sum()

### -- Categoricals: LabelEncoder

# Import LabelEncoder
from sklearn.preprocessing import LabelEncoder

# Fill missing values with 0
df.LotFrontage = df.LotFrontage.fillna(0)

# Create a boolean mask for categorical columns
categorical_mask = (df.dtypes == "object")

# Get list of categorical column names
categorical_columns = df.columns[categorical_mask].tolist()

# Print the head of the categorical columns
print(df[categorical_columns].head())

# Create LabelEncoder object: le
le = LabelEncoder()

# Apply LabelEncoder to categorical columns
df[categorical_columns] = df[categorical_columns].apply(lambda x: le.fit_transform(x))

# Print the head of the LabelEncoded categorical columns
print(df[categorical_columns].head())

### -- Categoricals: OneHotEncoder

# Import OneHotEncoder
from sklearn.preprocessing import OneHotEncoder

# Create OneHotEncoder: ohe
ohe = OneHotEncoder(sparse=False)

# Apply OneHotEncoder to categorical columns - output is no longer a dataframe: df_encoded
df_encoded = ohe.fit_transform(df)

# Print first 5 rows of the resulting dataset - again, this will no longer be a pandas dataframe
print(df_encoded[:5, :])

# Print the shape of the original DataFrame
print(df.shape)

# Print the shape of the transformed array
print(df_encoded.shape)

### -- Categoricals: DictVectorizer

# Import DictVectorizer
from sklearn.feature_extraction import DictVectorizer

# Convert df into a dictionary: df_dict
# Each row becomes a dictionary with key = col name and value = cell value
# All dictionaries/rows are inside a list
df_dict = df.to_dict(orient="records")

# Create the DictVectorizer object: dv
dv = DictVectorizer(sparse=False)

# Apply dv on df: df_encoded
df_encoded = dv.fit_transform(df_dict)

# Print the resulting first five rows
print(df_encoded[:5,:])

# Print the vocabulary
# NOTE: the order of the columns is not preserved!
# The dv.vocabulary_ maps the column index of df_encoded to a column name
# but the original order is not preserved.
# Thus, the df_encoded matrix is compliant with dv.vocabulary_ but the column
# order is new.
print(dv.vocabulary_)

### -- Pipeline

# Here, everything is done in a pipeline. Note that we need to use the Scikit-Learn API in order to pack the XGBoost models intoa pipeline.

import xgboost as xgb

df = pd.read_csv("../data/ames_housing_trimmed_processed.csv")
X = data.iloc[:,:-1]
y = data.iloc[:,-1]

# Fill LotFrontage missing values with 0
X.LotFrontage = X.LotFrontage.fillna(0)

# Setup the pipeline steps: steps
steps = [("ohe_onestep", DictVectorizer(sparse=False)),
         ("xgb_model", xgb.XGBRegressor())]

# Create the pipeline: xgb_pipeline
xgb_pipeline = Pipeline(steps)

# Fit the pipeline
xgb_pipeline.fit(X.to_dict("records"), y)

### -- Cross-Validation

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
import xgboost as xgb

df = pd.read_csv("../data/ames_housing_trimmed_processed.csv")
X = df.iloc[:,:-1]
y = df.iloc[:,-1]

# Fill LotFrontage missing values with 0
X.LotFrontage = X.LotFrontage.fillna(0)

# Setup the pipeline steps: steps
steps = [("ohe_onestep", DictVectorizer(sparse=False)),
         ("xgb_model", xgb.XGBRegressor(max_depth=2, objective="reg:squarederror"))]

# Create the pipeline: xgb_pipeline
xgb_pipeline = Pipeline(steps)

# Cross-validate the model
cross_val_scores = cross_val_score(xgb_pipeline,
                         X.to_dict("records"),
                         y,
                         cv=10,
                         scoring="neg_mean_squared_error")

# Print the 10-fold RMSE
print("10-fold RMSE: ", np.mean(np.sqrt(np.abs(cross_val_scores))))
```

### 4.3 Full Pipeline Example: Kidney Disease Dataset

We need to use the Scikit-Learn API in order to pack the XGBoost models intoa pipeline.

In this section, the [chronic kidney disease dataset](https://archive.ics.uci.edu/ml/datasets/chronic_kidney_disease) is used, which requirems more data processing.

From the dataset web:

```
1.Age(numerical) 
age in years 
2.Blood Pressure(numerical) 
bp in mm/Hg 
3.Specific Gravity(nominal) 
sg - (1.005,1.010,1.015,1.020,1.025) 
4.Albumin(nominal) 
al - (0,1,2,3,4,5) import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
import xgboost as xgb
5.Sugar(nominal) 
su - (0,1,2,3,4,5) 
6.Red Blood Cells(nominal) 
rbc - (normal,abnormal) 
7.Pus Cell (nominal) 
pc - (normal,abnormal) 
8.Pus Cell clumps(nominal) 
pcc - (present,notpresent) 
9.Bacteria(nominal) 
ba - (present,notpresent) 
10.Blood Glucose Random(numerical) 
bgr in mgs/dl 
11.Blood Urea(numerical) 
bu in mgs/dl 
12.Serum Creatinine(numerical) 
sc in mgs/dl 
13.Sodium(numerical) 
sod in mEq/L 
14.Potassium(numerical) 
pot in mEq/L 
15.Hemoglobin(numerical) 
hemo in gms 
16.Packed Cell Volume(numerical) 
17.White Blood Cell Count(numerical) 
wc in cells/cumm 
18.Red Blood Cell Count(numerical) 
rc in millions/cmm 
19.Hypertension(nominal) 
htn - (yes,no) 
20.Diabetes Mellitus(nominal) 
dm - (yes,no) 
21.Coronary Artery Disease(nominal) 
cad - (yes,no) 
22.Appetite(nominal) 
appet - (good,poor) 
23.Pedal Edema(nominal) 
pe - (yes,no) 
24.Anemia(nominal) 
ane - (yes,no) 
25.Class (nominal) 
class - (ckd,notckd)
```

```python
# Import necessary modules
import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
import xgboost as xgb
from sklearn_pandas import DataFrameMapper
from sklearn.impute import SimpleImputer
from sklearn.pipeline import FeatureUnion
from sklearn import preprocessing
from sklearn.feature_extraction import DictVectorizer
#from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin

column_names = ['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc', 'rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane', 'class']
# The CSV has no column names, but I got them from the web/Datacamp
# Also, note that NaN values are marked with a ?
df = pd.read_csv("../data/chronic_kidney_disease.csv", names=column_names, na_values='?')

df.shape # (400, 25)

# Extract features and target
X = df.iloc[:,:-1]
y = df.iloc[:,-1]

# Check number of nulls in each feature column
nulls_per_column = X.isnull().sum()
print(nulls_per_column)

# Create a boolean mask for categorical columns
categorical_feature_mask = X.dtypes == object

# Get list of categorical column names
categorical_columns = X.columns[categorical_feature_mask].tolist()

# Get list of non-categorical column names
non_categorical_columns = X.columns[~categorical_feature_mask].tolist()

# Apply numeric imputer
numeric_imputation_mapper = DataFrameMapper(
                                            [([numeric_feature], SimpleImputer(strategy="median")) for numeric_feature in non_categorical_columns],
                                            input_df=True,
                                            df_out=True
                                           )

# Apply categorical imputer
categorical_imputation_mapper = DataFrameMapper(
                                                [([category_feature], SimpleImputer(strategy="most_frequent")) for category_feature in categorical_columns],
                                                input_df=True,
                                                df_out=True
                                               )

# Transform labels
le = preprocessing.LabelEncoder()
y = le.fit_transform(y).astype("int")

cross_val_scores = cross_val_score(pipeline,
                         X,
                         y,
                         scoring="roc_auc",
                         cv=10)
```

### 4.4 Housing Pipeline with Random Search Cross-Validation

```python
import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV

data = pd.read_csv("../data/ames_housing_trimmed_processed.csv")
X = data.iloc[:,:-1]
y = data.iloc[:,-1]

data.shape # (1460, 57)

xgb_pipeline = Pipeline([("st_scaler", StandardScaler()),
                         ("xgb_model", xgb.XGBRegressor())])
gbm_param_grid = {
    'xgb_model__subsample': np.arange(.05, 1, .05),
    'xgb_model__max_depth': np.arange(3,20,1),
    'xgb_model__colsample_bytree': np.arange(.1,1.05,.05) }

randomized_neg_mse = RandomizedSearchCV(estimator=xgb_pipeline,
                                        param_distributions=gbm_param_grid, 
                                        n_iter=10,
                                        scoring='neg_mean_squared_error', cv=4)

randomized_neg_mse.fit(X, y)

print("Best rmse: ", np.sqrt(np.abs(randomized_neg_mse.best_score_)))

print("Best model: ", randomized_neg_mse.best_estimator_)
```