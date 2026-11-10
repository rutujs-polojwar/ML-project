# Customer Churn Prediction, Segmentation and Retention Framework
### Online Retail II dataset

## 1. Project Overview

This project analyses two years of historical retail transactions to understand customer purchasing behaviour, predict which customers are likely to stop buying (churn), and recommend targeted retention actions. It combines customer analytics, machine learning, explainable AI, customer segmentation, lifetime value estimation, product recommendation and an interactive dashboard.

The data is the **Online Retail II** dataset: transactions of a UK-based online gift-ware retailer between **1 December 2009 and 9 December 2011**. Most customers are wholesalers (businesses), so churn here means a business stopped reordering.

## 2. Project Objectives

| # | Objective | Status |
|---|---|---|
| 1 | Analyse historical transactions and identify purchasing patterns and customer behaviour | Done (notebooks 01, 02) |
| 2 | Develop and compare ML classification models for churn prediction | Data ready (notebook 03); modelling next |
| 3 | Identify behavioural factors behind inactivity using Explainable AI (SHAP) | Planned |
| 4 | Customer segmentation and customer lifetime value estimation | Planned |
| 5 | Personalised product recommendations (collaborative filtering and association rules) | Planned |
| 6 | Intelligent retention framework using churn probability, customer value and preferences | Planned |
| 7 | Interactive dashboard integrating all results | Planned |
| 8 | Evaluate model performance and simulate the effect of retention strategies | Planned |

## 3. Folder Structure

```
project/
|-- online_retail_II.xlsx            raw data (2 sheets: 2009-2010 and 2010-2011)
|-- 01_data_preparation.ipynb
|-- 02_exploratory_analysis.ipynb
|-- 03_churn_features.ipynb
|-- analyse_retail.ipynb             optional quick overview (superseded by 01 and 02)
|-- eda_summary.txt                  created by notebook 02
|-- data/                            created by notebooks 01 and 03
|   |-- clean_sales.csv
|   |-- clean_returns.csv
|   |-- sales_no_customer.csv
|   |-- product_lookup.csv
|   |-- cleaning_summary.csv
|   |-- churn_dataset.csv
|   `-- feature_dictionary.csv
`-- figures/                         charts created by notebooks 02 and 03
```

## 4. How to Run

1. Install the requirements: `pip install pandas numpy matplotlib openpyxl jupyter`
2. Put `online_retail_II.xlsx` and the notebooks in the same folder.
3. Open each notebook in Jupyter or VS Code and click **Run All**, **in order: 01, then 02, then 03**.

Notebooks 02 and 03 read the files that notebook 01 saves into `data/`, so they must be run from the same folder. Notebook 01 takes about a minute because the Excel file is large.

## 5. Notebooks

**01_data_preparation.ipynb** : Loads both Excel sheets, combines them and removes exact duplicates (including the December 2010 overlap between sheets). Every row is labelled as a real sale, cancellation/return, non-product code (postage, fees, test items, gift vouchers), stock adjustment, bad-debt adjustment or zero-price line. Product descriptions are standardised to one per code. Saves the clean tables to `data/` (sales, returns, sales without customer ID, product lookup, cleaning summary) and runs validation checks on them.

**02_exploratory_analysis.ipynb** : Explores the clean data (Objective 1): monthly trends, seasonality by month, weekday and hour, countries, top products and revenue concentration, orders per customer, order value, time between purchases, new vs returning customers, cohort retention and return rates. Removes three very large orders that were cancelled in full. Saves charts to `figures/` and findings to `eda_summary.txt`, and ends with conclusions that guide the churn definition.

**03_churn_features.ipynb** : Defines churn as "no purchase in the 150 days after a cut-off date" and builds the machine-learning table. Features (recency, frequency, spend, purchase gaps, trends, products, returns, tenure, country) use only data before each cut-off. Several snapshot dates are used, with a time-based train/test split. Seven leakage tests confirm no future information is used. Saves `churn_dataset.csv` and `feature_dictionary.csv`.

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

## 7. Method Notes

- **Churn definition:** no purchase in the 150 days after the cut-off date, among customers who bought in the previous 365 days.
- **No data leakage:** features use only data before the cut-off; labels use only the 150 days after. Training outcome windows end before the test cut-off.
- **Outcome columns:** `Outcome_FutureOrders` and `Outcome_FutureSpend` in `churn_dataset.csv` are for evaluation and simulation only and must never be used as model features.

## 8. Planned Next Steps

4. Train and compare churn models (logistic regression baseline, Random Forest, XGBoost or LightGBM) with grouped cross-validation, ROC-AUC, PR-AUC, F1, recall and probability calibration.
5. Explain the best model with SHAP and summarise the behavioural drivers of churn.
6. Segment customers (K-Means on RFM features) and estimate lifetime value (BG/NBD and Gamma-Gamma).
7. Build the recommender: FP-Growth association rules and collaborative filtering, evaluated with Precision@K, Recall@K and hit rate.
8. Build the retention framework: rules combining churn probability, customer value and preferred products.
9. Simulate retention campaigns (assumed uplift, sensitivity analysis, cost versus revenue saved, targeted versus random).
10. Build the dashboard (Streamlit or Power BI): overview, churn risk list, customer profile, segments, recommendations and retention actions.

## 9. Limitations

- The data is mostly wholesale, so findings may not transfer to individual consumers.
- No demographics, product categories, channel or marketing history are available.
- Only two years of data, so lifetime value estimates are limited and tests use a single latest snapshot.
- No real campaign outcomes exist, so the effectiveness of retention strategies (objectives 6 and 8) is estimated by simulation, not measured.
- Product categories, if needed, must be derived from the product descriptions.
