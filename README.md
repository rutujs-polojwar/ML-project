# Customer Churn Prediction, Segmentation and Retention Framework

### Online Retail II dataset

## 1. Project Overview

This project analyses two years of historical retail transactions to understand customer purchasing behaviour, predict which customers are likely to stop buying (churn), and recommend targeted retention actions. It combines customer analytics, machine learning, explainable AI, customer segmentation, lifetime value estimation, product recommendation and an interactive dashboard.

The data is the **Online Retail II** dataset: transactions of a UK-based online gift-ware retailer between **1 December 2009 and 9 December 2011**. Most customers are wholesalers (businesses), so churn here means a business stopped reordering.

## 2. Project Objectives

| #   | Objective                                                                               | Status                  |
| --- | --------------------------------------------------------------------------------------- | ----------------------- |
| 1   | Analyse historical transactions and identify purchasing patterns and customer behaviour | Done (notebooks 01, 02) |
| 2   | Develop and compare ML classification models for churn prediction                       | Done (notebooks 03, 04) |
| 3   | Identify behavioural factors behind inactivity using Explainable AI (SHAP)              | Planned                 |
| 4   | Customer segmentation and customer lifetime value estimation                            | Planned                 |
| 5   | Personalised product recommendations (collaborative filtering and association rules)    | Planned                 |
| 6   | Intelligent retention framework using churn probability, customer value and preferences | Planned                 |
| 7   | Interactive dashboard integrating all results                                           | Planned                 |
| 8   | Evaluate model performance and simulate the effect of retention strategies              | Planned                 |

## 3. Folder Structure

```
project/
|-- online_retail_II.xlsx            raw data (2 sheets: 2009-2010 and 2010-2011)
|-- 01_data_preparation.ipynb
|-- 02_exploratory_analysis.ipynb
|-- 03_churn_features.ipynb
|-- 04_churn_models.ipynb
|-- analyse_retail.ipynb             optional quick overview (superseded by 01 and 02)
|-- eda_summary.txt                  created by notebook 02
|-- model_summary.txt                created by notebook 04
|-- data/                            created by notebooks 01, 03 and 04
|   |-- clean_sales.csv
|   |-- clean_returns.csv
|   |-- sales_no_customer.csv
|   |-- product_lookup.csv
|   |-- cleaning_summary.csv
|   |-- churn_dataset.csv
|   |-- feature_dictionary.csv
|   |-- model_comparison.csv
|   `-- churn_predictions_test.csv
|-- models/                          created by notebook 04
|   |-- churn_models.joblib          fitted models, calibrators and thresholds
|   `-- churn_scoring.py             helper to score customers (used by later steps and the dashboard)
`-- figures/                         charts created by notebooks 02, 03 and 04
```

## 4. How to Run

1. Install the requirements: `pip install pandas numpy matplotlib openpyxl scikit-learn joblib jupyter`
   (optional, for the XGBoost model: `pip install xgboost`; without it the notebook uses LightGBM or scikit-learn boosting automatically)
2. Put `online_retail_II.xlsx` and the notebooks in the same folder.
3. Open each notebook in Jupyter or VS Code and click **Run All**, **in order: 01, 02, 03, then 04**.

Each notebook reads the files saved by the earlier ones, so they must be run from the same folder. Notebook 01 takes about a minute because the Excel file is large. Notebook 04 takes a few minutes (longer on a machine with few CPU cores) because it tunes three models with cross-validation.

## 5. Notebooks

**01_data_preparation.ipynb** : Loads both Excel sheets, combines them and removes exact duplicates (including the December 2010 overlap between sheets). Every row is labelled as a real sale, cancellation/return, non-product code (postage, fees, test items, gift vouchers), stock adjustment, bad-debt adjustment or zero-price line. Product descriptions are standardised to one per code. Saves the clean tables to `data/` (sales, returns, sales without customer ID, product lookup, cleaning summary) and runs validation checks on them.

**02_exploratory_analysis.ipynb** : Explores the clean data (Objective 1): monthly trends, seasonality by month, weekday and hour, countries, top products and revenue concentration, orders per customer, order value, time between purchases, new vs returning customers, cohort retention and return rates. Removes three very large orders that were cancelled in full. Saves charts to `figures/` and findings to `eda_summary.txt`, and ends with conclusions that guide the churn definition.

**03_churn_features.ipynb** : Defines churn as "no purchase in the 150 days after a cut-off date" and builds the machine-learning table. Features (recency, frequency, spend, purchase gaps, trends, products, returns, tenure, country) use only data before each cut-off. Several snapshot dates are used, with a time-based train/test split. Seven leakage tests confirm no future information is used. Saves `churn_dataset.csv` and `feature_dictionary.csv`.

**04_churn_models.ipynb** : Trains and compares three churn models: logistic regression (baseline), Random Forest and gradient boosting (XGBoost, falling back to LightGBM or scikit-learn if not installed). The split is by time: train on the four 2010 snapshots, test once on the 1 July 2011 snapshot. Hyper-parameters are tuned with customer-grouped cross-validation. It reports ROC-AUC, PR-AUC, F1, precision and recall, calibrates the probabilities, chooses a decision threshold on training data only, and shows gains/lift by risk decile. Saves the models and test predictions.

**analyse_retail.ipynb** (optional) : An early one-click overview covering revenue trends, countries, top products, RFM customer segments and retention. Useful as a quick summary but replaced by notebooks 01 and 02.

## 6. Key Results So Far

**Data**

- 1,067,371 raw rows; 34,335 exact duplicates removed.
- Clean sales: 776,575 lines, 5,852 customers, 4,619 products, 36,594 orders.
- 23% of raw rows have no Customer ID, so customer-level analysis covers the other 77%.

**Business patterns**

- Strong seasonality: September to November brings about 36% of yearly revenue; November is the peak month.
- Revenue for Jan-Nov 2011 is about flat versus 2010 (-1.4%).
- The UK is about 83% of revenue; the next markets are EIRE, Netherlands, Germany and France.
- The top 20% of products make about 78% of revenue; the top 10% of customers make about 63%.
- 28% of customers ordered only once; the median time between purchases is 32 days, and 90% of gaps are under 148 days.
- Returns are about 2.8% of sales value.

**Churn dataset**

- 18,651 customer-snapshot rows with 20 features (14,338 train, 4,313 test).
- Overall churn rate 44.8%, varying by snapshot (32% to 56%) because of seasonality.
- Churn falls as order count rises (66% for one-time buyers, 9% for customers with 10+ orders) and rises with days since last purchase (25% within 30 days, 67% after 150 days).

**Churn models (test snapshot, 1 July 2011)**

| Model                            | CV ROC-AUC (train) | Test ROC-AUC | Test PR-AUC | F1    | Recall |
| -------------------------------- | ------------------ | ------------ | ----------- | ----- | ------ |
| Dummy baseline (average rate)    | -                  | 0.500        | 0.457       | -     | -      |
| Logistic Regression              | 0.748              | 0.781        | 0.709       | 0.714 | 0.920  |
| Random Forest                    | 0.753              | 0.788        | 0.722       | 0.714 | 0.938  |
| Gradient Boosting (scikit-learn) | 0.753              | 0.775        | 0.696       | 0.715 | 0.939  |

- The three models perform almost identically; their 95% confidence intervals overlap. The model picked by cross-validation (gradient boosting) was a near-tie with the Random Forest, which scored slightly higher on the test snapshot. XGBoost was not installed, so scikit-learn's boosting was used.
- Most of the predictive signal comes from simple behaviour (recency and frequency), so a more complex model adds little.
- No calibration method beat the raw probabilities in cross-validation, so none was applied. Logistic regression is the best calibrated on the test snapshot (calibration error 0.018, against 0.047 for Random Forest and 0.050 for gradient boosting).
- Ranking customers by risk works well: the highest-risk 10% churn at 75% (lift 1.65), the lowest-risk 10% at 2%. Contacting only the top 20% reaches 31% of all churners, and the top 50% reaches 73%, against 20% and 50% for random targeting.
- The model works best for customers with 5+ orders (ROC-AUC 0.79) and worst for customers with 1 to 4 orders (0.63 to 0.64), who have little history.

## 7. Method Notes

- **Churn definition:** no purchase in the 150 days after the cut-off date, among customers who bought in the previous 365 days.
- **No data leakage:** features use only data before the cut-off; labels use only the 150 days after. Training outcome windows end before the test cut-off.
- **Outcome columns:** `Outcome_FutureOrders` and `Outcome_FutureSpend` in `churn_dataset.csv` are for evaluation and simulation only and must never be used as model features.
- **Time-based split:** the four 2010 snapshots are used for training and the 1 July 2011 snapshot for testing. The test set is used once, at the end.
- **Grouped cross-validation:** the same customer never appears in both the training and validation fold.
- **Calibration and threshold:** both are chosen using out-of-fold training predictions, never the test set.
- **Threshold:** the F1-maximising threshold is low (about 0.30 to 0.35) because about 45% of customers churn, so F1 rewards catching most churners. Step 8 should choose the contact threshold using campaign cost and customer value instead.

## 8. Limitations

- The data is mostly wholesale, so findings may not transfer to individual consumers.
- No demographics, product categories, channel or marketing history are available.
- Only two years of data, so lifetime value estimates are limited and tests use a single latest snapshot.
- Model test results come from one cut-off date and one 150-day outcome window, so they describe one period only.
- On the test snapshot the tree models predict somewhat higher churn than actually occurred for medium- and low-risk customers (for example, 34% predicted vs 22% actual in the eighth risk decile), because customer behaviour shifted between the training and test periods. Rankings stay reliable; absolute probabilities should be treated as approximate.
- The three churn models are statistically hard to tell apart, so the choice between them should rest on simplicity and explainability as much as on the scores.
- Churn probabilities exist for the 1 July 2011 snapshot. Scoring the customers who are active today needs a new snapshot at the end of the data, which is added with the retention framework (Step 8).
- No real campaign outcomes exist, so the effectiveness of retention strategies (objectives 6 and 8) is estimated by simulation, not measured.
- Product categories, if needed, must be derived from the product descriptions.
