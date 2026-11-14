"""Helpers for the Streamlit dashboard.

Nothing in this file imports Streamlit, so every function can be tested without a browser.
The dashboard only READS files produced by the earlier steps (notebooks 03 to 09); it never changes them.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

# --------------------------------------------------------------------------------------------
# Files the dashboard reads:  key -> (file name in data/, notebook that creates it)
# --------------------------------------------------------------------------------------------
FILES = {
    "predictions": ("churn_predictions_test.csv", "Step 4 (04_churn_models.ipynb)"),
    "dataset": ("churn_dataset.csv", "Step 3 (feature table)"),
    "lookup": ("product_lookup.csv", "Step 1 (data preparation)"),
    "value": ("customer_value_test.csv", "Step 6 (06_segmentation_clv.ipynb)"),
    "segments": ("segment_profiles.csv", "Step 6 (06_segmentation_clv.ipynb)"),
    "clv_validation": ("clv_validation.csv", "Step 6 (06_segmentation_clv.ipynb)"),
    "shap_values": ("shap_values_test.csv", "Step 5 (05_shap_explainability.ipynb)"),
    "shap_drivers": ("shap_top_drivers_test.csv", "Step 5 (05_shap_explainability.ipynb)"),
    "shap_importance": ("shap_feature_importance.csv", "Step 5 (05_shap_explainability.ipynb)"),
    "recs": ("recommendations_test_snapshot.csv", "Step 7 (07_recommendations.ipynb)"),
    "assoc_rules": ("association_rules.csv", "Step 7 (07_recommendations.ipynb)"),
    "rec_compare": ("recommender_comparison.csv", "Step 7 (07_recommendations.ipynb)"),
    "actions": ("retention_actions_test.csv", "Step 8 (08_retention_framework.ipynb)"),
    "retention_rules": ("retention_rules.csv", "Step 8 (08_retention_framework.ipynb)"),
    "action_summary": ("retention_action_summary.csv", "Step 8 (08_retention_framework.ipynb)"),
    "sim_by_action": ("simulation_by_action.csv", "Step 9 (09_evaluation_simulation.ipynb)"),
    "sim_breakeven": ("simulation_breakeven.csv", "Step 9 (09_evaluation_simulation.ipynb)"),
    "sim_strategies": ("simulation_strategies.csv", "Step 9 (09_evaluation_simulation.ipynb)"),
    "sim_sensitivity": ("simulation_sensitivity.csv", "Step 9 (09_evaluation_simulation.ipynb)"),
}
FRAMEWORK_JSON = "retention_framework.json"
SIM_JSON = "simulation_settings.json"

RISK_COLORS = {"High": "#E45756", "Medium": "#F2B134", "Low": "#4C78A8"}
ACTION_COLORS = {"PERSONAL": "#E45756", "WINBACK_DISCOUNT": "#F58518", "REMINDER": "#EECA3B", "CHECKIN": "#72B7B2",
                 "LOYALTY": "#4C78A8", "CROSSSELL": "#9D7660", "MONITOR": "#BAB0AC"}
FACT_COLUMNS = ["Country", "Recency", "Frequency", "Monetary", "AvgOrderValue", "AvgGapDays", "StdGapDays", "RecencyGapRatio",
                "Orders90", "Spend90", "OrdersTrend", "SpendTrend", "DistinctProducts", "AvgItemsPerOrder", "AvgLinesPerOrder",
                "ReturnCount", "ReturnRate", "Tenure", "PurchaseRate"]


def find_root(start: Path | None = None) -> Path:
    """Project root = the folder that contains data/churn_predictions_test.csv (works from the root or from dashboard/)."""
    here = Path(start) if start else Path(__file__).resolve().parent
    for cand in (here, here.parent, Path.cwd(), Path.cwd().parent):
        if (cand / "data" / FILES["predictions"][0]).exists():
            return cand
    return here.parent


class Store:
    """All input tables, loaded once. Missing files are recorded instead of raising."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.data: dict[str, pd.DataFrame] = {}
        self.missing: list[tuple[str, str, str]] = []
        for key, (fname, src) in FILES.items():
            path = self.root / "data" / fname
            if not path.exists():
                self.missing.append((key, fname, src))
                continue
            kw = {"dtype": {"StockCode": str}} if key in ("recs", "lookup") else {}
            df = pd.read_csv(path, **kw)
            if key == "dataset":
                df = df[df["Split"] == "test"][["CustomerID"] + [c for c in FACT_COLUMNS if c in df.columns]].reset_index(drop=True)
            self.data[key] = df
        self.framework = self._json(FRAMEWORK_JSON)
        self.sim_settings = self._json(SIM_JSON)

    def _json(self, name: str):
        p = self.root / "models" / name
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    def get(self, key: str):
        return self.data.get(key)

    def has(self, *keys: str) -> bool:
        return all(k in self.data for k in keys)

    def missing_text(self, *keys: str) -> str:
        rows = [f"{fname} (created by {src})" for key, fname, src in self.missing if key in keys]
        return "; ".join(rows)


# --------------------------------------------------------------------------------------------
# Master customer table
# --------------------------------------------------------------------------------------------
def build_customers(store: Store) -> pd.DataFrame | None:
    """One row per customer of the backtest snapshot: churn, value, segment, reasons, action. None if the core files are missing."""
    if not store.has("predictions", "value", "actions"):
        return None
    a = store.get("actions").copy()
    p = store.get("predictions")[["CustomerID", "Churn", "Outcome_FutureSpend"]]
    v = store.get("value")[["CustomerID", "SegmentID", "PAlive", "ExpPurchases150", "ExpPurchases12m", "ExpRevenue150", "ExpValuePerPurchase"]]
    c = a.merge(p, on="CustomerID", validate="1:1").merge(v, on="CustomerID", validate="1:1")
    if store.has("shap_drivers"):
        d = store.get("shap_drivers")[["CustomerID", "RiskDriver1", "RiskDriver2", "RiskDriver3", "Protective1", "Protective2", "WhyProtected"]]
        c = c.merge(d, on="CustomerID", how="left", validate="1:1")
    for col in ("WhyAtRisk", "WhyProtected", "RiskDriver1", "RiskDriver2", "RiskDriver3"):
        if col in c:
            c[col] = c[col].fillna("")
    return c.sort_values("ValueAtRisk", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------------------------
# Measured model quality (recomputed from the prediction file, no scikit-learn needed)
# --------------------------------------------------------------------------------------------
def roc_auc(y, p) -> float:
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    ranks = pd.Series(p).rank().to_numpy()
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def average_precision(y, p) -> float:
    """Average precision (area under the precision-recall curve); tied scores are treated as one threshold."""
    p = np.asarray(p, dtype=float)
    order = np.argsort(-p, kind="stable")
    ps, ys = p[order], np.asarray(y)[order]
    tp, fp = np.cumsum(ys), np.cumsum(1 - ys)
    last = np.r_[np.where(np.diff(ps) != 0)[0], len(ps) - 1]          # last index of each distinct score
    tp, fp = tp[last], fp[last]
    precision, recall = tp / (tp + fp), tp / ys.sum()
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def churn_quality(c: pd.DataFrame, threshold: float = 0.35) -> dict:
    y, p = c["Churn"].to_numpy(), c["ChurnProbability"].to_numpy()
    pred = p >= threshold
    tp, fp, fn = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum()), int((~pred & (y == 1)).sum())
    top = np.argsort(-p, kind="stable")[: int(0.2 * len(p))]
    return {"churn_rate": float(y.mean()), "roc_auc": roc_auc(y, p), "pr_auc": average_precision(y, p),
            "precision": tp / (tp + fp), "recall": tp / (tp + fn), "top20_capture": float(y[top].sum() / y.sum()), "threshold": threshold}


# --------------------------------------------------------------------------------------------
# Retention framework (same rules as notebook 08)
# --------------------------------------------------------------------------------------------
DRIVER_ANGLE = {"Recency": "timing", "RecencyGapRatio": "timing", "AvgGapDays": "timing", "StdGapDays": "timing",
                "Frequency": "frequency", "PurchaseRate": "frequency", "Orders90": "frequency", "OrdersTrend": "frequency",
                "Monetary": "spend", "Spend90": "spend", "SpendTrend": "spend", "AvgOrderValue": "spend", "AvgItemsPerOrder": "spend",
                "AvgLinesPerOrder": "spend", "DistinctProducts": "spend", "ReturnRate": "service", "ReturnCount": "service"}
ANGLE_TEXT = {"timing": "it has been longer than usual since the last order",
              "frequency": "orders are less frequent than for loyal customers",
              "spend": "the spend and order-size pattern resembles customers who leave",
              "service": "returns are unusually high",
              "onetime": "the customer has placed only one order so far, so the aim is a second order",
              "general": "the overall buying pattern points to a risk of leaving"}


def framework_tables(store: Store):
    """(actions table indexed by ActionCode, settings dict, risk cut-offs, value cut-offs) from models/retention_framework.json."""
    fw = store.framework
    if not fw:
        return None
    actions = pd.DataFrame(fw["actions_ASSUMED_costs_gbp"]).set_index("ActionCode")
    return actions, fw["settings"], fw["risk_tier_cutoffs"], fw["value_tier_cutoffs_clv12m"]


def offer_text(r, actions: pd.DataFrame) -> str:
    fav, rec = r["FavouriteProducts"].split(" | "), r["RecommendedProducts"].split(" | ")
    disc = actions.loc[r["ActionCode"], "DiscountPct"]
    service = " Resolve the returns / quality issue first." if r["MessageAngle"] == "service" else ""
    a = r["ActionCode"]
    if a == "PERSONAL":
        lead = ("Account manager call, starting with service recovery (returns are unusually high)" if r["MessageAngle"] == "service"
                else f"Account manager call: {ANGLE_TEXT[r['MessageAngle']]}")
        offer = f"offer a tailored {disc:.0%} discount on the next order." if disc > 0 else "no discount: focus on the relationship and the products."
        return f"{lead}. Then suggest restocking {', '.join(fav)}; introduce {', '.join(rec[:2])}; {offer}"
    if a == "WINBACK_DISCOUNT":
        if disc <= 0:
            return f"Win-back email featuring the usual products (no discount): {', '.join(fav)}.{service}"
        return f"Email with a {disc:.0%} discount code on the next order, featuring the usual products: {', '.join(fav)}.{service}"
    if a == "REMINDER":
        return f"Reorder reminder email featuring the usual products: {', '.join(fav)}.{service}"
    if a == "CHECKIN":
        return f"Check-in email from the account manager: new products picked for them ({', '.join(rec)}); usual products {', '.join(fav[:2])}.{service}"
    if a == "LOYALTY":
        return f"Thank-you reward (priority service or credit) and early access to new products: {', '.join(rec)}."
    if a == "CROSSSELL":
        return f"Email: new products picked for this customer: {', '.join(rec)}."
    return "No contact; keep monitoring."


def apply_framework(c: pd.DataFrame, actions: pd.DataFrame, risk_cuts: dict, value_cuts: dict,
                    p_elevated: float, n_personal: int, min_clv_discount: float) -> pd.DataFrame:
    """Apply the Step 8 rule table (first matching rule wins) with the given settings and action costs."""
    d = c.copy()
    p, clv = d["ChurnProbability"], d["CLV12m"]
    d["RiskTier"] = np.select([p >= risk_cuts["high_from"], p >= risk_cuts["medium_from"]], ["High", "Medium"], default="Low")
    d["ValueTier"] = np.select([clv >= value_cuts["high_from"], clv >= value_cuts["medium_from"]], ["High", "Medium"], default="Low")
    d["ValueAtRisk"] = p * clv
    elevated = p >= p_elevated
    var_rank = d["ValueAtRisk"].where(elevated).rank(ascending=False, method="first")
    r_high, r_med = d["RiskTier"] == "High", d["RiskTier"] == "Medium"
    v_high, v_med, v_low = d["ValueTier"] == "High", d["ValueTier"] == "Medium", d["ValueTier"] == "Low"
    conds = [elevated & (var_rank <= n_personal), r_high & (clv >= min_clv_discount), r_high,
             (r_med & ~v_low) | (elevated & v_high), r_med, v_high, v_med]
    d["ActionCode"] = np.select(conds, ["PERSONAL", "WINBACK_DISCOUNT", "REMINDER", "CHECKIN", "REMINDER", "LOYALTY", "CROSSSELL"], default="MONITOR")
    d["RuleFired"] = np.select(conds, [1, 2, 3, 4, 5, 6, 7], default=8)
    d["Priority"] = d["ValueAtRisk"].where(d["ActionCode"] != "MONITOR").rank(ascending=False, method="first")
    for col in ("Action", "Channel", "ContactCostGBP", "DiscountPct"):
        d[col] = d["ActionCode"].map(actions[col])
    d["MaxDiscountCostGBP"] = d["DiscountPct"] * d["ExpValuePerPurchase"]
    d["Offer"] = d.apply(lambda r: offer_text(r, actions), axis=1)
    return d


def action_summary(d: pd.DataFrame, actions: pd.DataFrame) -> pd.DataFrame:
    g = d.groupby("ActionCode")
    t = pd.DataFrame({"Customers": g.size(), "AvgChurnProb": g["ChurnProbability"].mean(), "AvgCLV": g["CLV12m"].mean(),
                      "TotalValueAtRisk": g["ValueAtRisk"].sum(), "ContactCost": g["ContactCostGBP"].sum(),
                      "MaxDiscountCost": g["MaxDiscountCostGBP"].sum()}).reindex(actions.index).fillna(0)
    t.insert(0, "Action", actions["Action"])
    t.insert(2, "ShareOfCustomers", t["Customers"] / max(len(d), 1))
    t["ShareOfValueAtRisk"] = t["TotalValueAtRisk"] / max(t["TotalValueAtRisk"].sum(), 1e-9)
    return t


# --------------------------------------------------------------------------------------------
# SHAP helpers
# --------------------------------------------------------------------------------------------
def shap_features(shap_df: pd.DataFrame) -> list[str]:
    return [c[5:] for c in shap_df.columns if c.startswith("SHAP_")]


def fmt_num(v) -> str:
    if isinstance(v, str):
        return v
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "no history"
    return f"{v:,.2f}".rstrip("0").rstrip(".") if abs(v) < 100 else f"{v:,.0f}"


# --------------------------------------------------------------------------------------------
# Charts (matplotlib Figure objects; Streamlit shows them with st.pyplot)
# --------------------------------------------------------------------------------------------
def _style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.3)


def fig_prob_hist(c: pd.DataFrame) -> Figure:
    fig = Figure(figsize=(6.4, 3.6)); ax = fig.subplots()
    tiers = ["Low", "Medium", "High"]
    ax.hist([c.loc[c["RiskTier"] == t, "ChurnProbability"] for t in tiers], bins=40, range=(0, 1), stacked=True,
            color=[RISK_COLORS[t] for t in tiers], label=[f"{t} risk" for t in tiers])
    ax.set_xlabel("Churn probability"); ax.set_ylabel("Customers"); ax.legend(fontsize=8); _style(ax)
    ax.set_title("How likely are customers to churn?", fontsize=10)
    fig.tight_layout(); return fig


def fig_risk_value_grid(c: pd.DataFrame) -> Figure:
    order = ["High", "Medium", "Low"]
    cnt = pd.crosstab(c["RiskTier"], c["ValueTier"]).reindex(index=order, columns=order).fillna(0)
    share = c.pivot_table(index="RiskTier", columns="ValueTier", values="CLV12m", aggfunc="sum").reindex(index=order, columns=order).fillna(0) / c["CLV12m"].sum()
    fig = Figure(figsize=(5.4, 3.6)); ax = fig.subplots()
    ax.imshow(share.to_numpy(), cmap="Blues", vmin=0, vmax=max(share.to_numpy().max(), 1e-9)); ax.grid(False)
    ax.set_xticks(range(3)); ax.set_xticklabels([f"{o} value" for o in order]); ax.set_yticks(range(3)); ax.set_yticklabels([f"{o} risk" for o in order])
    for i, r in enumerate(order):
        for j, v in enumerate(order):
            dark = share.loc[r, v] > 0.5 * max(share.to_numpy().max(), 1e-9)
            ax.text(j, i, f"{int(cnt.loc[r, v]):,}\n{share.loc[r, v]:.0%} of CLV", ha="center", va="center", fontsize=8, color="white" if dark else "black")
    ax.set_title("Customers and share of 12-month CLV", fontsize=10)
    fig.tight_layout(); return fig


def fig_shap_importance(imp: pd.DataFrame, top: int = 12) -> Figure:
    t = imp.sort_values("MeanAbsSHAP_pp", ascending=False).head(top).iloc[::-1]
    fig = Figure(figsize=(6.4, 4.2)); ax = fig.subplots()
    ax.barh(t["Feature"], t["MeanAbsSHAP_pp"], color="#4C78A8"); _style(ax)
    ax.set_xlabel("Average effect on churn probability (percentage points)"); ax.set_title("What the churn model relies on (SHAP)", fontsize=10)
    fig.tight_layout(); return fig


def fig_waterfall(shap_row: pd.Series, feats: list[str], values: dict, top: int = 8) -> Figure:
    """Contribution of each feature to one customer's churn probability. shap_row needs BaseValue, ChurnProbability and SHAP_<feature>."""
    ph = np.array([shap_row[f"SHAP_{f}"] for f in feats])
    base, prob = float(shap_row["BaseValue"]), float(shap_row["ChurnProbability"])
    order = np.argsort(-np.abs(ph))
    show, rest = order[:top], order[top:]
    labels = [f"{feats[j]} = {fmt_num(values.get(feats[j]))}" for j in show]
    vals = [ph[j] * 100 for j in show]
    if len(rest):
        labels.append(f"{len(rest)} other features"); vals.append(ph[rest].sum() * 100)
    fig = Figure(figsize=(8, 0.42 * len(vals) + 1.3)); ax = fig.subplots()
    left = base * 100
    path = [left]
    for r, v in enumerate(vals):
        ax.barh(r, v, left=left, color="#E45756" if v > 0 else "#4C78A8")
        ax.text(left + v + (0.4 if v > 0 else -0.4), r, f"{v:+.1f}", va="center", ha="left" if v > 0 else "right", fontsize=8)
        left += v
        path.append(left)
    span = max(max(path) - min(path), 1.0)
    ax.set_xlim(min(path) - 0.12 * span, max(path) + 0.12 * span)          # room for the value labels
    ax.axvline(base * 100, color="grey", ls=":", lw=1); ax.axvline(prob * 100, color="k", ls="--", lw=1)
    ax.set_yticks(range(len(vals))); ax.set_yticklabels(labels, fontsize=9); ax.invert_yaxis(); ax.grid(axis="y", alpha=0)
    ax.set_xlabel("Churn probability (%)"); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.set_title(f"From a typical customer ({base:.0%}, dotted) to this customer ({prob:.0%}, dashed)", fontsize=10)
    fig.tight_layout(); return fig


def fig_segment_scatter(c: pd.DataFrame) -> Figure:
    fig = Figure(figsize=(6.4, 4.2)); ax = fig.subplots()
    segs = c[["SegmentID", "Segment"]].drop_duplicates().sort_values("SegmentID")
    cmap = [f"C{i}" for i in range(10)]
    for _, r in segs.iterrows():
        m = c["SegmentID"] == r["SegmentID"]
        ax.scatter(c.loc[m, "Recency"].clip(lower=1), c.loc[m, "Frequency"], s=6, alpha=0.4, color=cmap[(int(r["SegmentID"]) - 1) % 10], label=f"{int(r['SegmentID'])}. {r['Segment']}")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("Recency (days since last purchase)"); ax.set_ylabel("Frequency (orders)")
    ax.legend(fontsize=7, markerscale=2); _style(ax); ax.set_title("Segments in recency-frequency space", fontsize=10)
    fig.tight_layout(); return fig


def fig_segment_bars(seg: pd.DataFrame) -> Figure:
    fig = Figure(figsize=(6.4, 3.6)); ax = fig.subplots()
    x = np.arange(len(seg)); w = 0.27
    ax.bar(x - w, seg["ShareOfCustomers"] * 100, w, label="% of customers", color="#4C78A8")
    ax.bar(x, seg["ShareOfPastSpend"] * 100, w, label="% of past spend", color="#F58518")
    ax.bar(x + w, seg["ShareOfTotalCLV"] * 100, w, label="% of future value (CLV)", color="#59A14F")
    ax.set_xticks(x); ax.set_xticklabels([str(int(s)) for s in seg["SegmentID"]]); ax.set_xlabel("Segment"); ax.set_ylabel("%"); ax.legend(fontsize=8); _style(ax)
    ax.set_title("Customers, past spend and future value by segment", fontsize=10)
    fig.tight_layout(); return fig


def fig_action_mix(summary: pd.DataFrame) -> Figure:
    fig = Figure(figsize=(9, 3.4)); ax = fig.subplots(1, 2)
    colors = [ACTION_COLORS.get(a, "#999999") for a in summary.index]
    ax[0].bar(range(len(summary)), summary["Customers"], color=colors); ax[0].set_title("Customers per action", fontsize=10)
    ax[1].bar(range(len(summary)), summary["ContactCost"], color=colors); ax[1].set_title("Contact cost per action (GBP, assumed unit costs)", fontsize=10)
    for a in ax:
        a.set_xticks(range(len(summary))); a.set_xticklabels(summary.index, rotation=45, ha="right", fontsize=8); _style(a)
    fig.tight_layout(); return fig


def fig_top_recommended(recs: pd.DataFrame, top: int = 15) -> Figure:
    t = recs.groupby("Description").size().sort_values(ascending=False).head(top).iloc[::-1]
    fig = Figure(figsize=(7, 4.4)); ax = fig.subplots()
    ax.barh([str(s)[:40] for s in t.index], t.values, color="#4C78A8"); _style(ax)
    ax.set_xlabel("Number of customers it is recommended to"); ax.set_title("Most recommended products", fontsize=10)
    fig.tight_layout(); return fig


def fig_breakeven(be: pd.DataFrame) -> Figure:
    b = be[be["ActionCode"] != "MONITOR"].reset_index(drop=True)
    fig = Figure(figsize=(8, 3.6)); ax = fig.subplots()
    x = np.arange(len(b)); w = 0.38
    ax.bar(x - w / 2, b["AssumedW"] * 100, w, label="assumed win-back rate (ASSUMPTION)", color="#4C78A8")
    ax.bar(x + w / 2, b["BreakEvenW"].replace(np.inf, np.nan) * 100, w, label="break-even rate", color="#F28E2B")
    ax.set_xticks(x); ax.set_xticklabels(b["ActionCode"], rotation=35, ha="right", fontsize=8); ax.set_ylabel("% of would-be churners retained"); ax.legend(fontsize=8); _style(ax)
    ax.set_title("SIMULATED: assumed vs break-even win-back rate", fontsize=10)
    fig.tight_layout(); return fig


def fig_strategies(strat: pd.DataFrame) -> Figure:
    s = strat[strat["Strategy"] != "No campaign"].reset_index(drop=True)
    fig = Figure(figsize=(8.5, 0.45 * len(s) + 1.2)); ax = fig.subplots()
    cols = ["#4C78A8" if "Framework (Step" in n else ("#A0CBE8" if "What-if" in n else "#BBBBBB") for n in s["Strategy"]]
    ax.barh(range(len(s))[::-1], s["Net"], color=cols)
    ax.set_yticks(range(len(s))[::-1]); ax.set_yticklabels([n if len(n) < 62 else n[:59] + "..." for n in s["Strategy"]], fontsize=8)
    ax.axvline(0, color="k", lw=0.8); ax.set_xlabel("Net benefit (GBP)"); _style(ax)
    ax.set_title("SIMULATED: strategies compared (light blue = what-if hypotheses)", fontsize=10)
    fig.tight_layout(); return fig


def scorecard(store: Store, c: pd.DataFrame) -> pd.DataFrame:
    """Measured quality of the three models, recomputed or read from the files of the earlier steps."""
    q = churn_quality(c)
    rows = [("Churn model (Step 4)", "ROC-AUC", f"{q['roc_auc']:.3f}", "ranking quality; 0.5 = random"),
            ("Churn model (Step 4)", "PR-AUC", f"{q['pr_auc']:.3f}", f"a random model scores about the churn rate, {q['churn_rate']:.1%}"),
            ("Churn model (Step 4)", f"Precision / recall at threshold {q['threshold']}", f"{q['precision']:.3f} / {q['recall']:.3f}", ""),
            ("Churn model (Step 4)", "Churners reached by contacting the top 20% by risk", f"{q['top20_capture']:.1%}", "random targeting reaches 20%")]
    cv = store.get("clv_validation")
    if cv is not None:
        for _, r in cv[cv["Group"] == "All active customers"].iterrows():
            rows.append(("CLV model (Step 6)", f"Predicted / actual purchase days, cut-off {r['Cutoff']}", f"{r['PurchaseRatio']:.2f}",
                         "1.00 = perfectly calibrated; the model has no seasonal term"))
    rc = store.get("rec_compare")
    if rc is not None:
        d = rc[rc["Setting"].str.startswith("Discovery")].set_index("Method")
        ci = d["Precision@10 [95% CI]"]
        best = ci.str.split(" ").str[0].astype(float).drop("Popularity").idxmax()
        rows.append(("Recommender (Step 7)", f"Precision@10, {best} (new products only)", ci[best], "95% interval in brackets"))
        rows.append(("Recommender (Step 7)", "Precision@10, popularity baseline", ci["Popularity"], ""))
    return pd.DataFrame(rows, columns=["Component", "Metric", "Value", "Note"])
