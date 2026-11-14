"""Customer retention dashboard (Step 10).

Run from the project folder (the one that contains data/ and models/):

    streamlit run dashboard/app.py

The dashboard only reads the files written by notebooks 03 to 09. Actual outcomes (who really churned) exist because this is a
backtest of the 1 July 2011 snapshot; they are shown only where labelled "backtest". All effects of retention actions are SIMULATED.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lib  # noqa: E402

st.set_page_config(page_title="Customer Retention Dashboard", page_icon="📊", layout="wide")


@st.cache_resource(show_spinner="Loading data ...")
def load_everything(root: str):
    store = lib.Store(Path(root))
    return store, lib.build_customers(store)


ROOT = lib.find_root()
S, C = load_everything(str(ROOT))

PAGES = ["Overview", "Churn risk list", "Customer profile", "Segments", "Recommendations", "Retention actions", "Evaluation & simulation"]


# ----------------------------------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------------------------------
def gbp(v: float) -> str:
    return f"£{v:,.0f}"


def need(*keys: str) -> bool:
    """True if the files exist; otherwise explain which notebook creates them."""
    if S.has(*keys):
        return True
    st.warning("This page needs files that were not found: " + S.missing_text(*keys) + ". Run that notebook first.")
    return False


def customer_picker(label: str, key: str, n: int = 200):
    top = C.head(n)
    labels = {int(r.CustomerID): f"{int(r.CustomerID)} | {r.Segment} | churn {r.ChurnProbability:.0%} | value at risk {gbp(r.ValueAtRisk)}" for r in top.itertuples()}
    return st.selectbox(label, list(labels), format_func=lambda x: labels[x], key=key)


def show_table(df: pd.DataFrame):
    st.dataframe(df, hide_index=True)


# ----------------------------------------------------------------------------------------------
# pages
# ----------------------------------------------------------------------------------------------
def page_overview():
    st.title("Customer retention dashboard")
    snap = str(C["SnapshotDate"].iloc[0])
    st.caption(f"Online Retail II | customers as of {snap} | churn = no purchase in the following 150 days")
    st.info("This is a **backtest**: the snapshot lies in the past, so what customers really did afterwards is known and is used only to check the models "
            "(always labelled *backtest*). The effect of retention actions is **simulated**, never measured (see the last page).")

    k = st.columns(5)
    k[0].metric("Customers scored", f"{len(C):,}")
    k[1].metric("Average churn probability", f"{C['ChurnProbability'].mean():.0%}")
    k[2].metric("High-risk customers", f"{(C['RiskTier'] == 'High').sum():,}", help="Top 20% of churn probabilities (Step 4 risk tier)")
    k[3].metric("Expected revenue, next 12 months", gbp(C["CLV12m"].sum()), help="Sum of customer lifetime value (CLV): expected revenue, not profit")
    k[4].metric("Customers contacted in the plan", f"{(C['ActionCode'] != 'MONITOR').sum():,}", help="All customers except 'no action'")

    q = lib.churn_quality(C)
    st.subheader("How good is the churn model? (measured in the backtest)")
    m = st.columns(4)
    m[0].metric("ROC-AUC", f"{q['roc_auc']:.3f}", help="0.5 = random, 1 = perfect")
    m[1].metric("PR-AUC", f"{q['pr_auc']:.3f}", help=f"A random model scores about {q['churn_rate']:.0%}")
    m[2].metric("Churners reached by contacting the top 20% by risk", f"{q['top20_capture']:.0%}", help="Random targeting reaches 20%")
    m[3].metric("Actual churn rate (backtest)", f"{q['churn_rate']:.0%}")

    a, b = st.columns(2)
    with a:
        st.pyplot(lib.fig_prob_hist(C))
    with b:
        st.pyplot(lib.fig_risk_value_grid(C))
    st.caption("Right: customers by churn risk and value tier, with the share of total 12-month CLV in each cell. High-value customers are almost all low risk, "
               "so the money at risk is spread over all risk tiers.")
    if S.has("segments"):
        st.subheader("Segments at a glance")
        seg = S.get("segments")[["SegmentID", "Segment", "Customers", "ShareOfCustomers", "ShareOfPastSpend", "ActualChurnRate", "MeanCLV"]].copy()
        for col in ("ShareOfCustomers", "ShareOfPastSpend", "ActualChurnRate"):
            seg[col] = (seg[col] * 100).round(0).astype(int).astype(str) + "%"
        seg["MeanCLV"] = seg["MeanCLV"].round(0)
        show_table(seg)


def page_risk():
    st.title("Churn risk")
    tab_list, tab_why = st.tabs(["Customer list", "What drives churn"])
    with tab_list:
        c1, c2, c3, c4 = st.columns(4)
        segs_all = sorted(C["Segment"].unique())
        tiers = c1.multiselect("Risk tier", ["High", "Medium", "Low"], default=["High", "Medium", "Low"])
        segs = c2.multiselect("Segment", segs_all, default=segs_all)
        min_p = c3.slider("Minimum churn probability", 0.0, 1.0, 0.0, 0.05)
        min_clv = c4.number_input("Minimum 12-month CLV (£)", min_value=0, max_value=100000, value=0, step=100)
        c5, c6, c7 = st.columns(3)
        sort_by = c5.selectbox("Sort by", ["Value at risk", "Churn probability", "12-month CLV"])
        n_rows = c6.selectbox("Rows to show", [50, 100, 500, 1000], index=1)
        countries = c7.multiselect("Country (empty = all)", sorted(C["Country"].unique()), default=[])

        f = C[C["RiskTier"].isin(tiers) & C["Segment"].isin(segs) & (C["ChurnProbability"] >= min_p) & (C["CLV12m"] >= min_clv)]
        if countries:
            f = f[f["Country"].isin(countries)]
        key = {"Value at risk": "ValueAtRisk", "Churn probability": "ChurnProbability", "12-month CLV": "CLV12m"}[sort_by]
        f = f.sort_values(key, ascending=False)
        st.write(f"**{len(f):,}** of {len(C):,} customers match; showing the first {min(n_rows, len(f)):,}. "
                 f"Their expected 12-month revenue is {gbp(f['CLV12m'].sum())} and their value at risk {gbp(f['ValueAtRisk'].sum())}.")
        view = f.head(n_rows)[["CustomerID", "Segment", "Country", "ChurnProbability", "RiskTier", "CLV12m", "ValueAtRisk", "Action", "WhyAtRisk"]].copy()
        view["ChurnProbability"] = view["ChurnProbability"].round(3)
        view["CLV12m"] = view["CLV12m"].round(0)
        view["ValueAtRisk"] = view["ValueAtRisk"].round(0)
        show_table(view)
        st.download_button("Download the filtered list (CSV)", f.to_csv(index=False).encode("utf-8"), file_name="churn_risk_list.csv", mime="text/csv")
        st.caption("Value at risk = churn probability x 12-month CLV: a priority index in GBP, not a forecast. Open **Customer profile** for the reasons behind one customer.")
    with tab_why:
        if need("shap_importance"):
            imp = S.get("shap_importance")
            a, b = st.columns([1, 1])
            with a:
                st.pyplot(lib.fig_shap_importance(imp))
            with b:
                st.markdown("**How to read this.** Each bar is how much, on average, a feature moves a customer's churn probability up or down "
                            "(percentage points). It shows what the *model* relies on, which is evidence of association, not proof of cause.")
                st.warning("`Tenure` ranks high here but it is partly an artefact: it is counted from the dataset's start (Dec 2009) and 48% of the test customers "
                           "are older than any training customer. Read it together with Frequency and PurchaseRate.")
            t = imp[["Feature", "Meaning", "MeanAbsSHAP_pp", "Statement"]].copy()
            t["MeanAbsSHAP_pp"] = t["MeanAbsSHAP_pp"].round(2)
            st.subheader("Plain statements (from the SHAP analysis)")
            show_table(t.rename(columns={"MeanAbsSHAP_pp": "Avg effect (pp)"}))


def page_profile():
    st.title("Customer profile")
    mode = st.radio("Find a customer by", ["Priority list (top 200 by value at risk)", "Customer ID"], horizontal=True)
    if mode.startswith("Priority"):
        cid = customer_picker("Customer", key="profile_pick")
    else:
        txt = st.text_input("Customer ID", value=str(int(C["CustomerID"].iloc[0])))
        try:
            cid = int(txt)
        except ValueError:
            st.warning("Please enter a numeric customer ID.")
            return
        if not (C["CustomerID"] == cid).any():
            st.warning(f"Customer {cid} is not in the scored snapshot.")
            return
    r = C[C["CustomerID"] == cid].iloc[0]

    k = st.columns(5)
    k[0].metric("Churn probability", f"{r['ChurnProbability']:.0%}", help=f"Risk tier: {r['RiskTier']}")
    k[1].metric("Risk tier", r["RiskTier"])
    k[2].metric("12-month CLV", gbp(r["CLV12m"]), help=f"Value tier: {r['ValueTier']}")
    k[3].metric("Value at risk", gbp(r["ValueAtRisk"]))
    k[4].metric("Segment", r["Segment"])

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Why is this customer at risk? (SHAP)")
        if S.has("shap_values", "dataset"):
            sv = S.get("shap_values")
            row = sv[sv["CustomerID"] == cid]
            ds = S.get("dataset")
            vals = ds[ds["CustomerID"] == cid]
            if len(row) and len(vals):
                st.pyplot(lib.fig_waterfall(row.iloc[0], lib.shap_features(sv), vals.iloc[0].to_dict()))
                st.caption("Red bars raise the churn probability, blue bars lower it, compared with a typical customer. Contributions add up to this customer's probability.")
            else:
                st.info("No SHAP values stored for this customer.")
        else:
            st.info("SHAP charts need " + S.missing_text("shap_values", "dataset"))
        if r["WhyAtRisk"]:
            st.markdown(f"**Raises the risk:** {r['WhyAtRisk']}")
        if r["WhyProtected"]:
            st.markdown(f"**Lowers the risk:** {r['WhyProtected']}")
    with right:
        st.subheader("Recommended action")
        st.markdown(f"**{r['Action']}**" + (f" (priority #{int(r['Priority'])})" if pd.notna(r["Priority"]) else ""))
        st.write(r["Offer"])
        st.caption(f"Channel: {r['Channel']} | assumed contact cost {gbp(r['ContactCostGBP'])} | message angle: {r['MessageAngle']}")
        st.subheader("Usual products")
        for name in str(r["FavouriteProducts"]).split(" | "):
            st.write("- " + name)
        if S.has("recs"):
            st.subheader("Recommended new products")
            rec = S.get("recs")
            rec = rec[rec["CustomerID"] == cid][["Rank", "Description", "Score"]].copy()
            rec["Score"] = rec["Score"].round(3)
            show_table(rec)

    if S.has("dataset"):
        ds = S.get("dataset")
        d = ds[ds["CustomerID"] == cid]
        if len(d):
            facts = pd.DataFrame({"Measure": [c for c in lib.FACT_COLUMNS if c in d.columns],
                                  "Value": [lib.fmt_num(d.iloc[0][c]) for c in lib.FACT_COLUMNS if c in d.columns]})
            with st.expander("Customer facts used by the model"):
                show_table(facts)
    outcome = "churned (no purchase in the next 150 days)" if r["Churn"] == 1 else "kept buying"
    st.caption(f"Backtest only: this customer actually {outcome}; spend in the next 150 days {gbp(r['Outcome_FutureSpend'])}.")


def page_segments():
    st.title("Segments")
    if not need("segments"):
        return
    seg = S.get("segments")
    st.write("Customers are grouped with K-Means on log-scaled Recency, Frequency and Monetary value. The 'actual churn' column is what happened afterwards (backtest).")
    show = seg[["SegmentID", "Segment", "Customers", "MedianRecency", "MedianFrequency", "MedianMonetary", "ShareOfPastSpend", "ActualChurnRate",
                "AvgPredictedChurn", "MeanCLV", "ShareOfTotalCLV"]].copy()
    for col in ("ShareOfPastSpend", "ActualChurnRate", "AvgPredictedChurn", "ShareOfTotalCLV"):
        show[col] = (show[col] * 100).round(0).astype(int).astype(str) + "%"
    show[["MedianRecency", "MedianFrequency", "MedianMonetary", "MeanCLV"]] = show[["MedianRecency", "MedianFrequency", "MedianMonetary", "MeanCLV"]].round(0)
    show_table(show)
    a, b = st.columns(2)
    with a:
        st.pyplot(lib.fig_segment_scatter(C))
    with b:
        st.pyplot(lib.fig_segment_bars(seg))
    st.subheader("Segments by churn risk")
    show_table(pd.crosstab(C["Segment"], C["RiskTier"]).reindex(columns=["High", "Medium", "Low"]).reset_index())
    st.subheader("Explore one segment")
    name = st.selectbox("Segment", sorted(C["Segment"].unique()))
    g = C[C["Segment"] == name]
    st.write(f"**{len(g):,}** customers, average churn probability {g['ChurnProbability'].mean():.0%}, expected 12-month revenue {gbp(g['CLV12m'].sum())}.")
    v = g.head(25)[["CustomerID", "ChurnProbability", "RiskTier", "CLV12m", "ValueAtRisk", "Action"]].copy()
    v["ChurnProbability"] = v["ChurnProbability"].round(3)
    v[["CLV12m", "ValueAtRisk"]] = v[["CLV12m", "ValueAtRisk"]].round(0)
    st.caption("The 25 customers of this segment with the highest value at risk:")
    show_table(v)


def page_recs():
    st.title("Recommendations")
    if not need("recs"):
        return
    recs = S.get("recs")
    st.write(f"Recommended new products come from **{recs['Method'].iloc[0]}** (Step 7), trained on purchases before the snapshot; "
             "products a customer has bought before are not recommended.")
    st.subheader("For one customer")
    cid = customer_picker("Customer", key="recs_pick")
    r = C[C["CustomerID"] == cid].iloc[0]
    a, b = st.columns(2)
    with a:
        st.markdown("**Recommended new products**")
        t = recs[recs["CustomerID"] == cid][["Rank", "StockCode", "Description", "Score"]].copy()
        t["Score"] = t["Score"].round(3)
        show_table(t)
    with b:
        st.markdown("**Usual products (most orders)**")
        for name in str(r["FavouriteProducts"]).split(" | "):
            st.write("- " + name)
    st.subheader("Across all customers")
    st.pyplot(lib.fig_top_recommended(recs))
    if S.has("assoc_rules"):
        st.subheader("Products bought together (association rules)")
        rules = S.get("assoc_rules")
        x, y = st.columns(2)
        min_conf = x.slider("Minimum confidence", 0.0, 1.0, 0.3, 0.05)
        min_lift = y.slider("Minimum lift", 1.0, float(max(rules["lift"].max(), 2.0)), 5.0, 0.5)
        f = rules[(rules["confidence"] >= min_conf) & (rules["lift"] >= min_lift)].sort_values("lift", ascending=False).head(25)
        st.write(f"{len(f)} rules shown (top 25 by lift). Confidence = share of baskets with A that also contain B; lift = how much more often than chance.")
        t = f[["antecedent_name", "consequent_name", "support", "confidence", "lift"]].copy()
        t[["support", "confidence", "lift"]] = t[["support", "confidence", "lift"]].round(3)
        show_table(t.rename(columns={"antecedent_name": "If a basket has", "consequent_name": "it often also has"}))
    if S.has("rec_compare"):
        st.subheader("How well do the recommenders work? (measured)")
        rc = S.get("rec_compare")
        show_table(rc[["Setting", "Method", "Precision@10 [95% CI]", "Recall@10", "Hit rate@10", "NDCG@10", "Coverage"]].round(3))


def page_actions():
    st.title("Retention actions")
    fw = lib.framework_tables(S)
    if fw is None:
        st.warning("models/retention_framework.json was not found. Run 08_retention_framework.ipynb first.")
        return
    actions, settings, risk_cuts, value_cuts = fw
    st.warning("Costs, discounts and the personal-outreach capacity are **placeholder assumptions** (the data contains none). Change them below to see the effect; "
               "nothing here measures whether an action works.")
    c1, c2, c3 = st.columns(3)
    n_personal = c1.slider("Personal-outreach capacity (customers)", 0, 300, int(settings["N_PERSONAL"]), 5)
    p_elev = c2.slider("Elevated-risk threshold (churn probability)", 0.10, 0.80, float(settings["P_ELEVATED"]), 0.05)
    min_clv = c3.number_input("Minimum 12-month CLV for a discount (£)", min_value=0, max_value=5000, value=int(settings["MIN_CLV_FOR_DISCOUNT"]), step=25)
    edited = actions.copy()
    with st.expander("Edit unit costs and discounts (assumptions)"):
        st.caption("Tip: set the discount of the personal call to 0 to see a call without a blanket discount (the simulation in Step 9 suggests this is cheaper).")
        for code in actions.index:
            if code == "MONITOR":
                continue
            x, y = st.columns(2)
            edited.loc[code, "ContactCostGBP"] = x.number_input(f"{actions.loc[code, 'Action']}: cost per contact (£)", min_value=0.0, max_value=500.0,
                                                                value=float(actions.loc[code, "ContactCostGBP"]), step=0.1, key=f"cost_{code}")
            edited.loc[code, "DiscountPct"] = y.number_input(f"{actions.loc[code, 'Action']}: discount (%)", min_value=0.0, max_value=50.0,
                                                             value=float(actions.loc[code, "DiscountPct"] * 100), step=1.0, key=f"disc_{code}") / 100

    D = lib.apply_framework(C, edited, risk_cuts, value_cuts, p_elev, int(n_personal), float(min_clv))
    summary = lib.action_summary(D, edited)
    contacted = D["ActionCode"] != "MONITOR"
    k = st.columns(4)
    k[0].metric("Customers contacted", f"{contacted.sum():,}", help=f"of {len(D):,}")
    k[1].metric("Contact cost (assumed)", gbp(D["ContactCostGBP"].sum()))
    k[2].metric("Discount cost if every buyer uses it (upper bound)", gbp(D["MaxDiscountCostGBP"].sum()))
    k[3].metric("Share of value at risk covered", f"{D.loc[contacted, 'ValueAtRisk'].sum() / D['ValueAtRisk'].sum():.0%}")
    st.pyplot(lib.fig_action_mix(summary))
    t = summary.reset_index().copy()
    t["ShareOfCustomers"] = (t["ShareOfCustomers"] * 100).round(0).astype(int).astype(str) + "%"
    t["ShareOfValueAtRisk"] = (t["ShareOfValueAtRisk"] * 100).round(0).astype(int).astype(str) + "%"
    t[["AvgChurnProb"]] = t[["AvgChurnProb"]].round(2)
    t[["AvgCLV", "TotalValueAtRisk", "ContactCost", "MaxDiscountCost"]] = t[["AvgCLV", "TotalValueAtRisk", "ContactCost", "MaxDiscountCost"]].round(0)
    show_table(t)

    st.subheader("Action list")
    x, y, z = st.columns(3)
    pick = x.multiselect("Action", list(actions["Action"]), default=[a for a in actions["Action"] if a != actions.loc["MONITOR", "Action"]])
    segs_all = sorted(D["Segment"].unique())
    segs = y.multiselect("Segment", segs_all, default=segs_all)
    n_rows = z.selectbox("Rows to show", [50, 100, 500, 1000], index=1)
    f = D[D["Action"].isin(pick) & D["Segment"].isin(segs)].sort_values("Priority")
    st.write(f"{len(f):,} customers; sorted by priority (value at risk).")
    v = f.head(n_rows)[["Priority", "CustomerID", "Segment", "ChurnProbability", "CLV12m", "ValueAtRisk", "Action", "Offer"]].copy()
    v["Priority"] = v["Priority"].astype("Int64")
    v["ChurnProbability"] = v["ChurnProbability"].round(3)
    v[["CLV12m", "ValueAtRisk"]] = v[["CLV12m", "ValueAtRisk"]].round(0)
    show_table(v)
    st.download_button("Download the action list (CSV)", f.to_csv(index=False).encode("utf-8"), file_name="retention_action_list.csv", mime="text/csv")
    if S.has("retention_rules"):
        with st.expander("The rule table (first matching rule wins)"):
            show_table(S.get("retention_rules")[["Order", "ActionCode", "Condition", "Why"]])


def page_eval():
    st.title("Evaluation and simulation")
    st.subheader("Measured: how well do the models work?")
    show_table(lib.scorecard(S, C))
    st.error("**SIMULATED, NOT MEASURED.** Everything below replays the 1 July 2011 snapshot as if the retention actions had been carried out, using *assumed* "
             "effects, an assumed margin and placeholder costs. Only a controlled pilot (contact a random part of the at-risk customers, hold out the rest) "
             "can measure whether an action works.")
    if not need("sim_by_action", "sim_breakeven", "sim_strategies"):
        return
    if S.sim_settings:
        eff = pd.DataFrame(S.sim_settings["effects_assumed"])
        st.caption(f"Assumptions: gross margin {S.sim_settings['gross_margin']:.0%}; win-back rate = share of would-be churners an action retains. "
                   "The optional spend uplift is switched off in the base scenario.")
        with st.expander("Assumed win-back rates"):
            show_table(eff)
    byact = S.get("sim_by_action")
    st.subheader("Simulated result by action")
    t = byact[["ActionCode", "Customers", "BaselineChurnRate", "SavedCustomers", "SavedRevenue", "ContactCost", "DiscountCost", "Net", "ROI"]].copy()
    t["BaselineChurnRate"] = (t["BaselineChurnRate"] * 100).round(0).astype(int).astype(str) + "%"
    t[["SavedCustomers", "ROI"]] = t[["SavedCustomers", "ROI"]].round(1)
    t[["SavedRevenue", "ContactCost", "DiscountCost", "Net"]] = t[["SavedRevenue", "ContactCost", "DiscountCost", "Net"]].round(0)
    show_table(t)
    a, b = st.columns(2)
    with a:
        st.pyplot(lib.fig_breakeven(S.get("sim_breakeven")))
        st.caption("An action pays off only if it retains more than its break-even share of would-be churners.")
    with b:
        st.pyplot(lib.fig_strategies(S.get("sim_strategies")))
        st.caption("Light blue bars are hypotheses suggested by this same simulation, so their advantage is optimistic.")
    st.subheader("Strategies compared (simulated)")
    s = S.get("sim_strategies")[["Strategy", "Contacted", "Spend", "SavedCustomers", "ExtraRevenue", "Net", "ROI"]].round(1)
    show_table(s)
    if S.has("sim_sensitivity"):
        sens = S.get("sim_sensitivity")
        st.subheader("Sensitivity")
        grid = sens[sens["Table"] == "grid_net_by_effect_scale_and_margin"][["EffectScale", "margin 15%", "margin 20%", "margin 30%", "margin 40%", "margin 50%"]].round(0)
        st.caption("Total net benefit (GBP) when all effect sizes are scaled (rows) and for different gross margins (columns).")
        show_table(grid)
        tor = sens[sens["Table"] == "tornado_one_at_a_time"][["Assumption", "LowSetting", "NetAtLow", "HighSetting", "NetAtHigh"]].round(0)
        st.caption("One assumption at a time.")
        show_table(tor)


PAGE_FUNCS = {"Overview": page_overview, "Churn risk list": page_risk, "Customer profile": page_profile, "Segments": page_segments,
              "Recommendations": page_recs, "Retention actions": page_actions, "Evaluation & simulation": page_eval}

# ----------------------------------------------------------------------------------------------
# navigation
# ----------------------------------------------------------------------------------------------
st.sidebar.title("Retention dashboard")
page = st.sidebar.radio("Page", PAGES)
st.sidebar.caption(f"Data folder: {ROOT / 'data'}")
if S.missing:
    with st.sidebar.expander(f"{len(S.missing)} input file(s) missing"):
        for _, fname, src in S.missing:
            st.write(f"- {fname} ({src})")

if C is None:
    st.title("Customer retention dashboard")
    st.error("The core files are missing: " + S.missing_text("predictions", "value", "actions") + ". Run the notebooks of Steps 4 to 8 first, "
             "then start the dashboard from the project folder.")
    st.stop()

try:
    PAGE_FUNCS[page]()
except Exception as e:  # show the problem on the page instead of a blank screen
    st.error("This page could not be drawn.")
    st.exception(e)
