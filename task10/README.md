# Customer retention dashboard (Step 10)

A Streamlit dashboard over the results of Steps 3 to 9. It only **reads** the files the notebooks write; it never changes them.

## Run it

From the project folder (the one that contains `data/` and `models/`):

```bash
pip install -r dashboard/requirements.txt
streamlit run dashboard/app.py
```

The browser opens on `http://localhost:8501`. If a file is missing, the sidebar lists it together with the notebook that creates it, and the affected page says what to run.
After re-running a notebook, restart the app (or press `C` then *Clear cache* in the app menu) so it reloads the files.

## Pages

| Page | What it shows |
|---|---|
| Overview | Key numbers, model quality (measured), churn-probability distribution, risk x value grid, segments at a glance |
| Churn risk list | Filterable, sortable customer list with CSV download; a tab explaining what drives churn (SHAP importance and plain statements) |
| Customer profile | One customer: churn probability, CLV, segment, SHAP waterfall (why at risk), recommended action and offer text, usual and recommended products, facts used by the model |
| Segments | Segment table, recency-frequency map, customers vs spend vs CLV, segments by risk tier, top customers of a chosen segment |
| Recommendations | Recommended products per customer, most recommended products, association rules explorer, recommender quality (measured) |
| Retention actions | The Step 8 rules re-applied live: change capacity, risk threshold, minimum CLV for a discount, unit costs and discounts, and see the action list, the cost and the offers update. CSV download |
| Evaluation & simulation | Model scorecard (measured) and the Step 9 campaign simulation (clearly labelled SIMULATED): by action, break-even, strategies, sensitivity |

## Files it reads

`data/`: `churn_predictions_test.csv`, `churn_dataset.csv`, `product_lookup.csv`, `customer_value_test.csv`, `segment_profiles.csv`, `clv_validation.csv`,
`shap_values_test.csv`, `shap_top_drivers_test.csv`, `shap_feature_importance.csv`, `recommendations_test_snapshot.csv`, `association_rules.csv`,
`recommender_comparison.csv`, `retention_actions_test.csv`, `retention_rules.csv`, `retention_action_summary.csv`, `simulation_by_action.csv`,
`simulation_breakeven.csv`, `simulation_strategies.csv`, `simulation_sensitivity.csv`.
`models/`: `retention_framework.json`, `simulation_settings.json`.

## Things to know

* **It is a backtest.** The snapshot is 1 July 2011, so what customers really did afterwards is known; it appears only where labelled *backtest*. In real use those columns would not exist.
* **Costs, discounts and capacity are placeholders.** The effect of any retention action is simulated, never measured.
* The Retention actions page uses the same rules as `08_retention_framework.ipynb`; with the default settings it reproduces `retention_actions_test.csv` exactly.
* `customer_value_current.csv` (customers at the end of the data) is not used, because churn probabilities for that date have not been computed.

## Files

* `app.py` - the Streamlit pages (thin layer).
* `lib.py` - data loading, metrics, the rule engine and all charts (no Streamlit inside, so it can be tested without a browser).
