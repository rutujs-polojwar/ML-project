"""Score customers with the churn models saved by 04_churn_models.ipynb.

    import churn_scoring as cs
    bundle = cs.load_bundle("models/churn_models.joblib")
    df["ChurnProbability"] = cs.predict_proba(bundle, df)      # df needs the Step 3 feature columns
    df["PredictedChurn"] = cs.predict_label(bundle, df)
"""
import joblib
import numpy as np


def load_bundle(path="models/churn_models.joblib"):
    return joblib.load(path)


def add_flags(df):
    df = df.copy()
    df["GapMissing"] = df["AvgGapDays"].isna().astype(int)
    df["StdGapMissing"] = df["StdGapDays"].isna().astype(int)
    return df


def _logit(s):
    s = np.clip(s, 1e-6, 1 - 1e-6)
    return np.log(s / (1 - s))


def predict_proba(bundle, df, model=None):
    m = bundle["models"][model or bundle["best"]]
    s = m["pipeline"].predict_proba(add_flags(df)[bundle["features"]])[:, 1]
    if m["method"] == "isotonic":
        p = m["calibrator"].predict(s)
    elif m["method"] == "sigmoid":
        p = m["calibrator"].predict_proba(_logit(s).reshape(-1, 1))[:, 1]
    else:
        p = s
    return np.clip(p, 0.001, 0.999)


def predict_label(bundle, df, model=None):
    m = bundle["models"][model or bundle["best"]]
    return (predict_proba(bundle, df, model) >= m["threshold"]).astype(int)
