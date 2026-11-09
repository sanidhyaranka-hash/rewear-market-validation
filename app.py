"""
app.py — Streamlit dashboard
ReWear UAE: pre-owned fashion marketplace — market-validation analytics
Run:  streamlit run app.py
"""
import warnings
warnings.filterwarnings("ignore")
import os
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

os.chdir(os.path.dirname(os.path.abspath(__file__)))  # so data/ is found when run from anywhere (incl. Streamlit Cloud)
import analysis as rc


def P(name):
    """Data file path: works whether the CSVs are in data/ or next to app.py."""
    return f"data/{name}" if os.path.exists(f"data/{name}") else name


st.set_page_config(page_title="ReWear UAE — Market Validation", page_icon="♻️", layout="wide")
BLUE = rc.BLUE


def insight(text):
    st.markdown(f"<div style='background:#f3f6fb;border-left:4px solid {BLUE};padding:10px 14px;border-radius:4px;"
                f"margin:-6px 0 22px 0;font-size:0.93rem'>💡 {text}</div>", unsafe_allow_html=True)


def show(fig_ins, key=None):
    """Draw a chart and the explanation computed from the same (filtered) data."""
    fig, ins = fig_ins
    st.plotly_chart(fig, width="stretch", key=key)
    insight(f"<b>What it shows:</b> {ins['what']}<br><b>Why it matters:</b> {ins['why']}<br><b>So what for ReWear:</b> {ins['so']}")


# ----------------------------------------------------------------- data & models (cached)
@st.cache_data
def get_data():
    clean = rc.load(P("cleaned_survey_data.csv"))
    raw = pd.read_csv(P("raw_survey_responses.csv"), low_memory=False)
    log = pd.read_csv(P("cleaning_log.csv"))
    return raw, clean, log


@st.cache_resource
def get_models(_df):
    cls, clu, ar, reg = rc.run_classification(_df), rc.run_clustering(_df), rc.run_association(_df), rc.run_regression(_df)
    return cls, clu, ar, reg, rc.run_strategy(_df, clu, ar)


raw, df_all, cleaning_log = get_data()
with st.spinner("Training models (first load only, about 30–60 seconds)…"):
    CLS, CLU, ARM, REG, STR = get_models(df_all)

# ----------------------------------------------------------------- sidebar filters
st.sidebar.header("Filters (apply to EDA tabs)")
f_em = st.sidebar.multiselect("Emirate", rc.ordered(df_all, "Emirate"), default=rc.ordered(df_all, "Emirate"))
f_age = st.sidebar.multiselect("Age group", rc.ORDERS["Age_Group"], default=rc.ORDERS["Age_Group"])
f_occ = st.sidebar.multiselect("Occupation", sorted(df_all["Occupation"].unique()), default=sorted(df_all["Occupation"].unique()))
SP = rc.ORDERS["Monthly_Fashion_Spend_Band"]
f_sp = st.sidebar.multiselect("Monthly fashion spend (AED)", SP, default=SP)
df = df_all[df_all["Emirate"].isin(f_em) & df_all["Age_Group"].isin(f_age) & df_all["Occupation"].isin(f_occ)
            & df_all["Monthly_Fashion_Spend_Band"].isin(f_sp)]
st.sidebar.caption(f"Showing **{len(df):,}** of {len(df_all):,} cleaned respondents.")
st.sidebar.warning("All respondent data is **synthetic** (simulated) for coursework. Market figures in tab ⑨ are real and cited.")
if len(df) < 100:
    st.warning("Too few respondents for the current filters — widen them.")
    st.stop()

# ----------------------------------------------------------------- header
st.title("ReWear UAE — Market Validation Dashboard")
st.caption("Pre-owned fashion marketplace for the UAE  |  20-question consumer survey  |  "
           f"Synthetic data (n = {len(raw):,} raw → {len(df_all):,} clean)")

tabs = st.tabs(["① Overview & verdict", "② Data quality", "③ Descriptive EDA", "④ Diagnostic & correlation",
                "⑤ Classification", "⑥ Regression", "⑦ Clustering", "⑧ Association rules", "⑨ Feasibility & break-even"])

q19 = df_all.loc[df_all[rc.WTP_TARGET] > 0, rc.WTP_TARGET]
SC = rc.scenarios(df_all)
SM = STR["summary"].set_index("Strategy")
REC = SM.loc["S3"]

# ===================================================================== ① OVERVIEW
with tabs[0]:
    intent = df[rc.TARGET].mean()
    c = st.columns(5)
    c[0].metric("Respondents (filtered)", f"{len(df):,}")
    c[1].metric("Would use ReWear (Q20)", f"{intent:.0%}")
    c[2].metric("Willing to sell (Q15)", f"{df['Seller_Prospect'].mean():.0%}")
    c[3].metric("Typical price for an AED 300 item", f"AED {df.loc[df[rc.WTP_TARGET] > 0, rc.WTP_TARGET].median():.0f}")
    c[4].metric("Break-even (recommended plan)", f"Month {int(REC['Breakeven_month'])}")

    st.subheader("What the data says")
    best = CLS["results"].iloc[0]
    prof = CLU["profile"]
    top_seg = prof["Positive_Intent"].idxmax(); rich_seg = prof["Avg_Q19_WTP_AED"].idxmax()
    sup_seg = prof["Seller_Prospect"].idxmax(); low_seg = prof["Positive_Intent"].idxmin()
    st.markdown(f"""
- **Demand exists but is conditional.** {intent:.0%} of people say they would use ReWear, but only
  {(df['Q20_Use_ReWear_5Level'] == 'Definitely yes').mean():.0%} say "definitely". Stated intent always overstates real orders.
- **Experience is the switch.** The best classifier (*{best['Model']}*, ROC-AUC {best['ROC_AUC']:.2f}) ranks past second-hand buying,
  openness to pre-owned, preferred channel and price expectations as the strongest drivers.
- **Buyers, money and supply sit in different groups.** **{top_seg}** are the keenest buyers, **{rich_seg}** pay the most
  (AED {prof.loc[rich_seg, 'Avg_Q19_WTP_AED']:.0f} for an AED 300 item) and **{sup_seg}** are the most willing sellers. **{low_seg}** reject pre-owned.
- **Trust sells together.** Counterfeit worries go with demand for authenticity checks and certificates; "no time to sell" goes with doorstep pickup.
""")
    st.subheader("Verdict")
    st.info(f"**Sustainable — but only as a focused, trust-first marketplace, not a mass-market app.** At the typical accepted price of "
            f"AED {q19.median():.0f}, a 15% commission earns about AED 17 per order, less than a UAE courier, so a mass-market model that pays "
            f"delivery loses AED {abs(SC.iloc[0].contribution):.2f} on every order. The data points to three moves: (1) win the premium, "
            f"authenticated segment that lifts order value, (2) make buyers pay delivery plus a small buyer-protection fee, and (3) solve supply with "
            f"a pickup-and-list concierge for people who keep unused clothes. On that plan ReWear breaks even in **month {int(REC['Breakeven_month'])}** "
            f"and needs about **AED {REC['Extra_funding_needed_AED']/1000:,.0f}k** on top of the AED 220k start-up budget. See tab ⑨.")
    st.caption("This dataset is synthetic. It was built to mirror plausible market behaviour for coursework; it is not evidence of real demand. "
               "Validate with a real waitlist and a small Dubai pilot before investing.")

# ===================================================================== ② DATA QUALITY
with tabs[1]:
    st.subheader("Raw vs cleaned data")
    c = st.columns(4)
    c[0].metric("Raw rows", f"{len(raw):,}")
    c[1].metric("Clean rows", f"{len(df_all):,}", delta=f"{len(df_all) - len(raw)}")
    c[2].metric("Missing cells (raw)", f"{int(raw.isna().sum().sum()):,}")
    c[3].metric("Missing cells (clean)", f"{int(df_all.isna().sum().sum()):,}")
    show(rc.fig_missing(raw))
    st.subheader("Cleaning & transformation log")
    st.dataframe(cleaning_log, width="stretch", hide_index=True)
    show(rc.fig_outlier_box(raw, df_all))
    a, b = st.columns(2)
    a.write("**Raw Emirate labels** — one emirate split into many spellings"); a.dataframe(raw["Emirate"].value_counts(dropna=False), width="stretch")
    b.write("**Clean** — 7 valid emirates"); b.dataframe(df_all["Emirate"].value_counts(), width="stretch")
    insight("Without standardising, 'Dubai', ' dubai', 'DXB' and 'DUBAI' would be four separate groups and every emirate chart and model would be wrong. "
            "Duplicates would double-count people, and speeders who ticked every box would create fake product associations.")
    with st.expander("Column-by-column guide (which algorithm uses which column)"):
        st.dataframe(pd.read_csv(P("data_dictionary.csv")), width="stretch", hide_index=True)
    with st.expander("Preview cleaned data"):
        st.dataframe(df_all.head(50), width="stretch")

# ===================================================================== ③ DESCRIPTIVE
with tabs[2]:
    st.caption("Descriptive analytics: *who* answered and *what* they do today. Filters in the sidebar apply, and every explanation recalculates.")
    show(rc.fig_q20(df))
    c1, c2 = st.columns(2)
    with c1: show(rc.fig_profile(df, "Age_Group", "D1. Respondents by age group"))
    with c2: show(rc.fig_profile(df, "Emirate", "D2. Respondents by emirate"))
    c1, c2 = st.columns(2)
    with c1: show(rc.fig_profile(df, "Monthly_Fashion_Spend_Band", "D3. Monthly fashion spend (AED)"))
    with c2: show(rc.fig_profile(df, "Previous_Secondhand_Purchase", "D4. Previous second-hand purchase"))
    c1, c2 = st.columns(2)
    with c1: show(rc.fig_profile(df, "Preferred_Shopping_Channel", "D5. Preferred shopping channel"))
    with c2: show(rc.fig_profile(df, "Unused_Clothing_Behaviour", "D6. What happens to unused clothes (supply)"))
    c1, c2 = st.columns(2)
    with c1: show(rc.fig_multiselect(df, "Challenge_", "D7. Shopping challenges (Q9)"))
    with c2: show(rc.fig_multiselect(df, "Barrier_", "D8. Barriers to selling (Q11)"))
    c1, c2 = st.columns(2)
    with c1: show(rc.fig_multiselect(df, "Category_", "D9. Preferred categories (Q12)"))
    with c2: show(rc.fig_multiselect(df, "Trust_", "D10. What builds trust (Q14)"))
    show(rc.fig_multiselect(df, "Feature_", "D11. Most wanted platform features (Q13)"))
    c1, c2 = st.columns(2)
    with c1: show(rc.fig_wtp(df))
    with c2: show(rc.fig_wtp_summary(df))

# ===================================================================== ④ DIAGNOSTIC
with tabs[3]:
    st.caption("Diagnostic analytics: *why* some people would use ReWear and others would not. Chi-square tests check whether a gap could be chance; "
               "Cramér's V (0–1) measures its strength. These are links, not proof of cause.")
    show(rc.fig_corr(df))
    show(rc.fig_drivers(df))
    c1, c2 = st.columns(2)
    with c1: show(rc.fig_rate_by(df, "Previous_Secondhand_Purchase", "G3. Intent by previous second-hand buying"))
    with c2: show(rc.fig_rate_by(df, "Motivating_Discount", "G4. Intent by discount needed"))
    c1, c2 = st.columns(2)
    with c1: show(rc.fig_rate_by(df, "Preferred_Shopping_Channel", "G5. Intent by preferred channel"))
    with c2: show(rc.fig_rate_by(df, "Age_Group", "G6. Intent by age group"))
    c1, c2 = st.columns(2)
    with c1: show(rc.fig_wtp_by_spend(df))
    with c2: show(rc.fig_wtp_by_group(df))
    show(rc.fig_challenge_feature(df))
    c1, c2 = st.columns(2)
    with c1: show(rc.fig_supply(df))
    with c2: show(rc.fig_barriers_sellers(df))
    var = st.selectbox("Explore intent by any other question", rc.CATEGORICAL, index=rc.CATEGORICAL.index("Interest_in_Selling"))
    show(rc.fig_rate_by(df, var), key="explore")

# ===================================================================== ⑤ CLASSIFICATION
with tabs[4]:
    st.subheader("Who will use ReWear? — predicting Q20 'would use ReWear' (Yes/No)")
    st.caption("Features: demographics, spend, shopping habits, second-hand experience, discount attitude, channel, challenges, barriers, categories, "
               "features and trust drivers. Q20 itself is held back. 75/25 stratified split + 5-fold cross-validation.")
    r = CLS["results"]
    st.dataframe(r.style.format({c: "{:.3f}" for c in r.columns if c != "Model"})
                 .highlight_max(subset=["Accuracy", "Precision", "Recall", "F1", "ROC_AUC", "CV_AUC_mean"], color="#cde2fb"), width="stretch", hide_index=True)
    c1, c2 = st.columns([3, 2])
    with c1: st.plotly_chart(rc.fig_roc(CLS), width="stretch")
    with c2:
        m = st.selectbox("Confusion matrix for", list(CLS["cms"]), index=list(CLS["cms"]).index("Logistic Regression") if "Logistic Regression" in CLS["cms"] else 0)
        st.plotly_chart(rc.fig_cm(CLS, m), width="stretch")
    b = r.iloc[0]; lr = r.set_index("Model").loc["Logistic Regression"]
    insight(f"**{b['Model']}** has the best test ROC-AUC ({b['ROC_AUC']:.3f}); **Logistic Regression** is within {b['ROC_AUC'] - lr['ROC_AUC']:.3f} and has the best "
            f"cross-validated AUC ({lr['CV_AUC_mean']:.3f}), so it is the practical choice: it is simple to explain. Accuracy of about {lr['Accuracy']:.0%} "
            f"against a {CLS['base_rate']:.0%} base rate means the survey answers carry real signal. AUC ≈ {lr['ROC_AUC']:.2f} means the model ranks a "
            "random adopter above a random non-adopter about 8–9 times in 10 — good enough to score a waitlist and spend ad money on the top band first.")
    st.plotly_chart(rc.fig_importance(CLS), width="stretch")
    insight("Never having bought second-hand is by far the strongest (negative) signal, followed by rejecting pre-owned outright, preferring malls, resale "
            "experience and price expectations. These are the screening questions for a real sign-up form.")

# ===================================================================== ⑥ REGRESSION
with tabs[5]:
    st.subheader("How much will they pay? — predicting Q19 price for an AED 300 item in excellent condition (AED)")
    r = REG["results"]
    st.dataframe(r.style.format({"MAE_AED": "{:.2f}", "RMSE_AED": "{:.2f}", "R2": "{:.3f}", "Alpha": "{:.3f}"}), width="stretch", hide_index=True)
    c1, c2 = st.columns(2)
    with c1: st.plotly_chart(rc.fig_reg_scatter(REG), width="stretch")
    with c2: st.plotly_chart(rc.fig_coefs(REG), width="stretch")
    insight(f"The linear models explain about **{r['R2'].max():.0%}** of the differences in what people would pay, with a typical error of about "
            f"AED {r['MAE_AED'].min():.0f}. Ridge and Lasso barely beat plain linear regression, so the {REG['n_features']} features are not over-fitted; "
            f"Lasso sets {REG['lasso_zeroed']} weak coefficients to zero for a simpler model with equal accuracy. Refusing pre-owned and needing a 40%+ "
            "discount cut the price most; higher fashion spend and luxury interest raise it. Use the model as a pricing assistant for sellers.")

# ===================================================================== ⑦ CLUSTERING
with tabs[6]:
    st.subheader("Who are the customer segments? — K-means on demographics, spend, habits and attitudes")
    c1, c2 = st.columns(2)
    with c1: st.plotly_chart(rc.fig_elbow(CLU), width="stretch")
    with c2: st.plotly_chart(rc.fig_k(CLU), width="stretch")
    insight(f"The silhouette score peaks at **k = {CLU['k']}** ({CLU['silhouette']:.3f}) and the elbow bends around 4–5. A silhouette near 0.1 is normal "
            "for survey data, where people sit on a range rather than in tight clumps, so the segments are real tendencies, not hard boxes.")
    st.plotly_chart(rc.fig_cluster_scatter(CLU, df_all), width="stretch")
    st.plotly_chart(rc.fig_cluster_heat(CLU), width="stretch")
    p = CLU["profile"]
    st.dataframe(p.style.format({c: "{:.0%}" for c in p.columns if c.startswith(("Pct", "Share", "Positive", "Seller"))} |
                                {c: "{:.0f}" for c in p.columns if "AED" in c}), width="stretch")
    insight(" ".join(f"**{n}** — {p.loc[n, 'Share']:.0%} of people, {p.loc[n, 'Positive_Intent']:.0%} would use ReWear, "
                     f"{p.loc[n, 'Seller_Prospect']:.0%} would sell, AED {p.loc[n, 'Avg_Q19_WTP_AED']:.0f} for an AED 300 item." for n in p.index)
            + " Buyers, profit and supply come from different segments, so each needs its own offer.")

# ===================================================================== ⑧ ASSOCIATION RULES
with tabs[7]:
    st.subheader("What goes together? — Apriori on the multi-select answers (challenges, barriers, categories, features, trust)")
    c = st.columns(3)
    c[0].metric("Binary items", ARM["n_items"]); c[1].metric("Frequent itemsets", f"{ARM['n_itemsets']:,}"); c[2].metric("Rules kept", f"{len(ARM['rules']):,}")
    st.plotly_chart(rc.fig_rules(ARM), width="stretch")
    c = st.columns(3)
    minlift = c[0].slider("Min lift", 1.3, 3.0, 1.5, 0.1)
    minsup = c[1].slider("Min support", 0.05, 0.25, 0.05, 0.01)
    block = c[2].selectbox("Consequent block", ["All", "Feature", "Trust", "Category", "Barrier", "Challenge"])
    r = ARM["rules"]; r = r[(r.Lift >= minlift) & (r.Support >= minsup)]
    if block != "All": r = r[r.Consequent_Block == block]
    st.caption(f"**{len(r)} rules** pass the thresholds.")
    st.dataframe(r.head(25).style.format({"Support": "{:.1%}", "Confidence": "{:.1%}", "Lift": "{:.2f}", "Leverage": "{:.3f}", "Conviction": "{:.2f}"}),
                 width="stretch", hide_index=True)
    insight("Three clear patterns appear. **Trust bundle:** counterfeit worry + designer interest → authenticity check and certificate (lift ≈ 2.5, confidence ≈ 90%). "
            "**Luxury entry:** bags/accessories + authenticity → designer/luxury (lift ≈ 2.6). **Seller convenience:** no time + complicated process → doorstep pickup; "
            "don't know the price → AI price suggestions (lift ≈ 2). Lift > 1 means the items go together more often than chance.")
    st.subheader("Cross-sell playbook by segment: if they pick X, offer Y")
    st.dataframe(STR["crosssell"].style.format({"Support_in_segment": "{:.0%}", "Confidence": "{:.0%}", "Lift": "{:.2f}"}), hide_index=True, width="stretch")

# ===================================================================== ⑨ FEASIBILITY
with tabs[8]:
    st.subheader("Industry context (secondary research, 2024–26)")
    st.dataframe(pd.DataFrame(rc.MARKET_EVIDENCE, columns=["Evidence", "Figure", "Source"]), width="stretch", hide_index=True)
    st.caption("UAE resale estimates differ by more than 15× between sources. Treat them as direction, not precision.")

    st.subheader("Unit economics per order (edit the assumptions)")
    c = st.columns(4)
    aov = c[0].number_input("Average order value (AED)", 60, 600, int(round(q19.median())), 5)
    take = c[1].slider("Take rate (commission + buyer fee)", 0.05, 0.30, 0.15, 0.01)
    courier = c[2].slider("Courier cost per parcel (AED)", 8, 35, 18)
    fee = c[3].slider("Delivery fee paid by buyer (AED)", 0, 30, int(round(df_all["Delivery_WTP_AED_Midpoint"].mean())))
    c = st.columns(4)
    ops = c[0].slider("Ops / support / authentication per order (AED)", 0, 25, 4)
    fixed = c[1].number_input("Fixed costs per month (AED)", 10000, 60000, 22500, 500)
    cac = c[2].slider("Customer acquisition cost (AED)", 10, 200, 60, 5)
    opy = c[3].slider("Orders per buyer per year", 1.0, 8.0, 3.0, 0.5)
    u = rc.unit_economics(aov=aov, take_rate=take, courier_cost=courier, buyer_delivery_fee=fee, ops_per_order=ops,
                          fixed_monthly=fixed, cac=cac, orders_per_buyer_year=opy)
    k = st.columns(4)
    k[0].metric("Profit per order", f"AED {u['contribution']:.2f}", "loss-making" if u["contribution"] <= 0 else "positive",
                delta_color="inverse" if u["contribution"] <= 0 else "normal")
    k[1].metric("Break-even orders / month", "never" if u["contribution"] <= 0 else f"{u['breakeven_orders_month']:,.0f}")
    k[2].metric("Active buyers needed", "—" if u["contribution"] <= 0 else f"{u['breakeven_active_buyers']:,.0f}")
    k[3].metric("Customer value ÷ acquisition cost", f"{u['ltv_cac']:.2f}x", "healthy ≥ 3x", delta_color="off")
    wf = go.Figure(go.Waterfall(x=["Revenue", "Payment", "Ops", "Delivery subsidy", "Profit per order"],
                                y=[u["revenue_per_order"], -u["payment_cost"], -u["ops_cost"], -u["delivery_subsidy"], 0],
                                measure=["absolute", "relative", "relative", "relative", "total"],
                                increasing=dict(marker_color=rc.BLUE), decreasing=dict(marker_color=rc.ORANGE), totals=dict(marker_color=rc.AQUA),
                                text=[f"{v:.2f}" for v in [u["revenue_per_order"], -u["payment_cost"], -u["ops_cost"], -u["delivery_subsidy"], u["contribution"]]]))
    st.plotly_chart(rc.style(wf, "F1. Where the commission goes (AED per order)", h=360), width="stretch")
    insight(f"With these inputs each order earns **AED {u['contribution']:.2f}**. At the survey's typical price, if ReWear pays the courier, "
            "more orders mean more losses. Making the buyer pay delivery, adding a small buyer-protection fee and raising the order value with premium stock "
            "are what turn the number positive. All cost inputs are assumptions — replace them with real courier and payment-gateway quotes.")

    st.subheader("Whom to target: segment (clustering) × conversion tier (classification)")
    show(rc.fig_target_matrix(STR))
    show(rc.fig_price_points(STR))
    st.subheader("When do we break even? 36-month cash projection for three strategies")
    show(rc.fig_projection(STR))
    st.dataframe(STR["summary"].drop(columns=["Orders_per_buyer_per_month"], errors="ignore").style.format(precision=2), hide_index=True, width="stretch")
    st.dataframe(STR["sensitivity"].style.format({"Lowest_cash_AED": "AED {:,.0f}"}), hide_index=True, width="stretch")
    insight(f"Only the recommended plan (target likely buyers, put 40% of ad spend on premium buyers, buyer pays delivery plus a 5% + AED 2 protection fee) "
            f"breaks even — in **month {int(REC['Breakeven_month'])}**, with about {REC['Year1_new_customers']:,.0f} new customers in year 1 and roughly "
            f"AED {REC['Extra_funding_needed_AED']/1000:,.0f}k needed beyond the AED 220k budget. Advertising to everyone while paying delivery never breaks even.")
    with st.expander("Projection assumptions"):
        st.table(pd.DataFrame([{"Assumption": k_, "Value": str(v)} for k_, v in rc.STRATEGY_ASSUMPTIONS.items()]))
