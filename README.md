# Engage 2: Value from Clicks to Conversions

Predicting the purchase value of website sessions from user-behaviour data (Kaggle competition, IIT Madras (MLP).

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
| LightGBM | 0.236 |
| Random Forest Regressor | 0.266 |
| XGBRegressor | 0.283 |

XGBoost (tuned, with outlier handling) performed best with a validation R² of 0.283, ahead of Random Forest (0.266) and LightGBM (0.236). All three models fit the training data much better than the held-out data (e.g. train R² of 0.98 for LightGBM against 0.236 on validation), so overfitting was the main challenge. Cross-validation scores for XGBoost also varied a lot between folds, which suggests that stronger regularization, feature selection, or better outlier handling would be the next steps.

## Tech stack
Python, pandas, NumPy, scikit-learn, XGBoost, LightGBM, category_encoders, seaborn, matplotlib

## License
This project was developed as part of an academic assignment for the Machine Learning Practice course at IIT Madras.
