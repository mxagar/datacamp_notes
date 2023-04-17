### Customer Churn Dataset

This folder contains the dataset and a sample generated during training for inference testing purposes.

Source: [Credit Card Customers](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers/code).

Columns in the raw dataset (22 in total):

- **Target** (renamed to `Churn = {0,1}`)
  - `Attrition_Flag`: `Existing Customer`, `Attrited Customer` (churn).
- **Dropped** (irrelevant information)
  - `Unnamed: 0`: copy of index (not present in the current Kaggle dataset).
  - `CLIENTNUM`: client ID.
- **Categorical** (encoded as churn ratios per category level)
  - `Gender`: `F`, `M`.
  - `Education_Level`: `High School`, `Graduate`, etc.
  - `Marital_Status`: `Married`, `Single`, `Divorced`, `Unknown`
  - `Income_Category`: annual income range: `< $40K`, `$40K - 60K`, `$60K - $80K`, `$80K-$120K`, `> 120k`
  - `Card_Category`: type of card/product: `Blue`, `Silver`, `Gold`, `Platinum`
- **Numerical** (transformed if the absolute skew is larger than 0.75)
  - `Customer_Age`: customer's age in years (quite normal).
  - `Dependent_count`: number of dependents (quite normal).
  - `Months_on_book`: period of relationship with bank (normal with peak).
  - `Total_Relationship_Count`: total number of products held by the customer (quite uniform).
  - `Months_Inactive_12_mon`: number of months inactive in the last 12 months (quite normal).
  - `Contacts_Count_12_mon`: number of Contacts in the last 12 months (quite normal).
  - `Credit_Limit`: credit limit on the credit card (exponential decaying).
  - `Total_Revolving_Bal`: total revolving balance on the credit card, i.e., amount that goes unpaid at the end of the billing cycle (non-normal; peaks at both tails).
  - `Avg_Open_To_Buy`: open to buy credit line (average of last 12 months) (exponential).
  - `Total_Amt_Chng_Q4_Q1`: change in transaction amount (Q4 over Q1) (quite normal, maybe large central peak).
  - `Total_Trans_Amt`: total transaction amount (last 12 months) (tri-modal).
  - `Total_Trans_Ct`: total transaction count (last 12 months) (bi-modal).
  - `Total_Ct_Chng_Q4_Q1`: change in transaction count (Q4 over Q1)  (quite normal, maybe large central peak).
  - `Avg_Utilization_Ratio`: average card utilization ratio (exponential decaying).

The current Kaggle dataset has two additional Naive Bayes variables at the en which can be safely removed.

Both versions of the dataset (Kaggle and Udacity) have 10127 rows; the distributions look the same, so I assume they are the same dataset.