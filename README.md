# Engage 2: Value from Clicks to Conversions

Predicting the purchase value of website sessions from user-behaviour data (Kaggle competition, IIT Madras [course/term name]).

## Problem
Given session-level data (traffic source, device, page views, hits, etc.), predict `purchaseValue` for each session. This is a regression task.

## Approach
1. **EDA:** checked shape, dtypes, missing values, and correlations. Most features had weak correlation with the target.
2. **Cleaning:** dropped duplicates, constant columns, and columns with more than 90% nulls; filled `pageViews` nulls with the median.
3. **Feature engineering:** extracted year, month, day, and day of week from `date`; created interaction features (`hits_pageviews`, `session_pageviews`).
4. **Preprocessing:** scikit-learn `ColumnTransformer` with mean imputation for numeric columns, one-hot encoding for low-cardinality categoricals, and target encoding for high-cardinality ones.
5. **Models tried:** Dummy baseline, LightGBM, Random Forest, XGBoost.
6. **Tuning:** `RandomizedSearchCV` (30 iterations, 3-fold CV) on XGBoost.

## Results
| Model | Validation R² |
|---|---|
| Dummy baseline | [score] |
| XGBoost | [score] |
| XGBoost (tuned) | [score] |

[One or two sentences: what worked, what didn't, e.g. the effect of handling outliers.]

## How to run
- Data: download from the [Kaggle competition page]([link]) (not included in this repo).
- Update the file paths in the notebook (it reads from `/kaggle/input/...` on Kaggle).

## Tech stack
Python, pandas, NumPy, scikit-learn, XGBoost, LightGBM, category_encoders, seaborn, matplotlib

## License
This project was developed as part of an academic assignment for the Machine Learning Practice course at IIT Madras.
