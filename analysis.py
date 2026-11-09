"""
analysis.py - shared analysis engine for ReWear UAE
==================================================
Every chart, statistic and model lives here ONCE: descriptive and diagnostic
charts, classification, K-means clustering, Apriori association rules,
regression, unit economics and the 36-month break-even strategy.
It is imported by app.py (the Streamlit dashboard).

Each EDA function returns (plotly_figure, insight_dict). The insight is
computed from the data passed in, so when the dashboard is filtered the
explanation updates with the chart. insight_dict keys:
  what  - what the chart shows (with the actual numbers)
  why   - why it matters for ReWear
  so    - the business action / implication
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

# --------------------------------------------------------------------------- style
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
CAT = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED]
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIV = [[0, "#1c5cab"], [0.25, "#86b6ef"], [0.5, "#f0efec"], [0.75, "#f0a3a2"], [1, "#b42d2d"]]
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
FONT = 'system-ui, -apple-system, "Segoe UI", Roboto, Arial, sans-serif'


def style(fig, title=None, h=420, legend=False):
    fig.update_layout(
        template="none", height=h, title=dict(text=title, x=0, xanchor="left", y=0.985, yanchor="top", font=dict(size=16, color=INK)) if title else None,
        font=dict(family=FONT, size=12, color=INK2), paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        margin=dict(l=10, r=20, t=(95 if legend else 55) if title else 20, b=10), showlegend=legend,
        legend=dict(orientation="h", y=1.01, yanchor="bottom", x=0, title=None),
        hoverlabel=dict(bgcolor="white", font=dict(family=FONT, color=INK)), bargap=0.25)
    fig.update_xaxes(showgrid=False, linecolor=AXIS, tickfont=dict(color=MUTED), title_font=dict(color=INK2), automargin=True)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=AXIS, tickfont=dict(color=MUTED), title_font=dict(color=INK2), automargin=True)
    fig.update_traces(selector=dict(type="bar"), marker_cornerradius=4, marker_line_width=0)
    return fig


def pct(x, d=0):
    return f"{x * 100:.{d}f}%"


# --------------------------------------------------------------------------- columns
ORDERS = {
    "Age_Group": ["18-24", "25-30", "31-40", "41-50", "Above 50"],
    "Monthly_Fashion_Spend_Band": ["Below 100", "100-250", "251-500", "501-1000", "1001-2000", "Above 2000"],
    "Shopping_Frequency": ["Less often", "Every 6 months", "Every 2-3 months", "Monthly", "Several times a month"],
    "Previous_Secondhand_Purchase": ["Never", "Once or twice", "Occasionally", "Frequently"],
    "Motivating_Discount": ["Below 20%", "20-30%", "31-40%", "41-50%", "Above 50%", "Would not buy pre-owned"],
    "Interest_in_Selling": ["Definitely not", "Probably not", "Not sure", "Probably", "Definitely"],
    "Q20_Use_ReWear_5Level": ["Definitely not", "Probably not", "Not sure", "Probably yes", "Definitely yes"],
    "AED300_Item_WTP_Band": ["Would not buy", "Below 75", "75-100", "101-125", "126-150", "151-200", "Above 200"],
}
CATEGORICAL = ["Age_Group", "Emirate", "Occupation", "Monthly_Fashion_Spend_Band", "Shopping_Frequency",
               "Previous_Secondhand_Purchase", "Preferred_Shopping_Channel", "Motivating_Discount",
               "Unused_Clothing_Behaviour", "Interest_in_Selling"]
PREFIXES = ("Challenge_", "Barrier_", "Category_", "Feature_", "Trust_")
TARGET = "Conversion_Intent_Binary"
WTP_TARGET = "AED300_Item_WTP_AED_Midpoint"


def multi_cols(df, prefix=None):
    cols = [c for c in df.columns if c.startswith(PREFIXES) and not c.endswith("_Count")]
    return [c for c in cols if prefix is None or c.startswith(prefix)]


def nice(c):
    for p in PREFIXES:
        if c.startswith(p):
            c = c[len(p):]
    return c.replace("_", " ").replace("Dont", "Don't").replace("AI ", "AI ")


def load(path="data/cleaned_survey_data.csv"):
    return pd.read_csv(path, low_memory=False)


def ordered(df, col):
    return [v for v in ORDERS.get(col, sorted(df[col].dropna().unique())) if v in set(df[col])]


# --------------------------------------------------------------------------- statistics
def cramers_v(x, y):
    t = pd.crosstab(x, y)
    if t.shape[0] < 2 or t.shape[1] < 2:
        return np.nan, np.nan
    chi2, p, _, _ = stats.chi2_contingency(t)
    n = t.values.sum()
    return float(np.sqrt(chi2 / (n * (min(t.shape) - 1)))), float(p)


def driver_table(df):
    """Chi-square test + Cramér's V for every survey answer vs conversion intent."""
    rows = []
    for c in CATEGORICAL + multi_cols(df):
        v, p = cramers_v(df[c], df[TARGET])
        if np.isnan(v):
            continue
        grp = df.groupby(c)[TARGET].mean()
        rows.append({"Driver": c, "Question_Block": c.split("_")[0] if c in multi_cols(df) else "Profile",
                     "Cramers_V": round(v, 3), "p_value": p, "Significant_5pct": p < .05,
                     "Highest_Intent_Group": str(grp.idxmax()), "Highest_Rate": round(grp.max(), 3),
                     "Lowest_Intent_Group": str(grp.idxmin()), "Lowest_Rate": round(grp.min(), 3)})
    return pd.DataFrame(rows).sort_values("Cramers_V", ascending=False).reset_index(drop=True)


CORR_COLS = {
    "Age_Group_Ord": "Age", "Monthly_Fashion_Spend_Band_Ord": "Fashion spend", "Shopping_Frequency_Ord": "Shopping frequency",
    "Previous_Secondhand_Purchase_Ord": "Prior resale buying", "Discount_Needed_Pct": "Discount needed",
    "Interest_in_Selling_Ord": "Interest in selling", "Digital_Channel": "Prefers digital channel",
    "Challenge_Count": "# challenges", "Barrier_Count": "# selling barriers", "Category_Count": "# categories",
    "Feature_Count": "# features wanted", "Trust_Count": "# trust drivers",
    "Maximum_Item_Spend_AED_Midpoint": "Max item spend (Q16)", "Subscription_WTP_AED_Midpoint": "Membership WTP (Q17)",
    "Delivery_WTP_AED_Midpoint": "Delivery WTP (Q18)", "AED300_Item_WTP_AED_Midpoint": "Item WTP (Q19)",
    "Q20_Use_ReWear_5Level_Ord": "Intent to use (Q20)"}


def corr_matrix(df):
    return df[list(CORR_COLS)].rename(columns=CORR_COLS).corr(method="spearman")


# --------------------------------------------------------------------------- EDA charts
def fig_q20(df):
    order = ORDERS["Q20_Use_ReWear_5Level"]
    s = df.Q20_Use_ReWear_5Level.value_counts().reindex(order).fillna(0)
    colors = [RED, "#f0a3a2", "#c3c2b7", "#86b6ef", BLUE]
    fig = go.Figure(go.Bar(x=order, y=s.values, marker_color=colors, text=[pct(v / s.sum()) for v in s.values],
                           textposition="outside", hovertemplate="%{x}: %{y:,} respondents<extra></extra>"))
    fig.update_yaxes(title="Respondents")
    pos = df[TARGET].mean(); strong = (df.Q20_Use_ReWear_5Level == "Definitely yes").mean()
    neg = df.Q20_Use_ReWear_5Level.isin(["Definitely not", "Probably not"]).mean()
    return style(fig, "Q20 - Would you use ReWear?"), dict(
        what=f"{pct(pos, 1)} said 'Definitely' or 'Probably yes' (the positive class), but only {pct(strong, 1)} said 'Definitely yes'. "
             f"{pct(neg, 1)} lean against it and few sit in the middle ({pct((df.Q20_Use_ReWear_5Level=='Not sure').mean(), 1)} 'Not sure').",
        why="The target is close to balanced, which suits classification (no heavy resampling needed). The polarised shape means people have "
            "fairly firm views on pre-owned fashion - it is a 'fits me / does not fit me' product, not a mild preference.",
        so=f"Stated intent overstates real behaviour. Treat the {pct(strong, 0)} 'Definitely yes' group as the realistic early-adopter pool "
           f"and the 'Probably yes' group as people who still need a reason (price, trust) to act.")


def fig_profile(df, col, title=None):
    order = ordered(df, col)
    s = df[col].value_counts().reindex(order)
    if col not in ORDERS:
        s = s.sort_values(ascending=True)
        fig = go.Figure(go.Bar(y=s.index, x=s.values, orientation="h", marker_color=BLUE,
                               text=[pct(v / len(df)) for v in s.values], textposition="outside",
                               hovertemplate="%{y}: %{x:,}<extra></extra>"))
        fig.update_xaxes(title="Respondents")
    else:
        fig = go.Figure(go.Bar(x=s.index, y=s.values, marker_color=BLUE, text=[pct(v / len(df)) for v in s.values],
                               textposition="outside", hovertemplate="%{x}: %{y:,}<extra></extra>"))
        fig.update_yaxes(title="Respondents")
    top = s.idxmax()
    texts = {
        "Age_Group": ("Age is the main lens for adoption: younger shoppers buy more second-hand globally.",
                      "Weight launch content and channels to the largest age groups, but check later charts before assuming the young are the best buyers."),
        "Emirate": ("Dubai is deliberately the largest group (Dubai-first pilot design). The sample is NOT population-weighted.",
                    "Read emirate results as comparisons between groups, not as UAE market shares. Small emirates (n<150) give unstable rates."),
        "Occupation": ("Occupation proxies time and budget: students have time but little money; employees and homemakers hold more wardrobe supply.",
                       "Students are a demand pool; homemakers/employees are likely seller supply - the marketplace needs both."),
        "Monthly_Fashion_Spend_Band": ("Spend is the main driver of how much someone will pay (see regression).",
                                       "Most respondents spend AED 100-1,000 a month. Pricing must work for this middle, not for the small top band."),
        "Shopping_Frequency": ("Frequency drives repeat purchase, the key to marketplace economics.",
                               "Monthly-or-more shoppers are the repeat-order base; target them for retention offers."),
        "Previous_Secondhand_Purchase": ("Prior resale buying is the strongest single behavioural predictor of intent (see diagnostics).",
                                         "People who already buy second-hand are the cheapest to acquire; 'Never' buyers need trust-building first."),
        "Preferred_Shopping_Channel": ("Channel tells us where to acquire customers.",
                                       "Digital-first shoppers (marketplaces, social, resale apps) can be reached online; mall shoppers need a different hook."),
        "Unused_Clothing_Behaviour": ("This is the SUPPLY question: what happens to clothes people no longer wear.",
                                      "'Keep in wardrobe' and 'Discard' are untapped supply ReWear can unlock; 'Donate' competes with charities such as Thrift for Good."),
        "Interest_in_Selling": ("Seller interest measures whether the supply side will show up.",
                                "Marketplaces fail on supply first. A strong 'Probably/Definitely' share is necessary but must be proven with real listings."),
        "Motivating_Discount": ("Discount needed vs. the new price - a direct read of price sensitivity.",
                                "The modal answer sets the pricing guidance for sellers; the 'Would not buy' group should be excluded from targeting."),
    }
    why, so = texts.get(col, ("", ""))
    return style(fig, title or col.replace("_", " "), h=380), dict(
        what=f"Largest group: '{top}' with {s.max():,} respondents ({pct(s.max() / len(df), 1)}). "
             f"Smallest: '{s.idxmin()}' ({pct(s.min() / len(df), 1)}).", why=why, so=so)


def fig_multiselect(df, prefix, title):
    cols = multi_cols(df, prefix)
    if prefix == "Challenge_":
        cols = [c for c in cols if "No_Significant" not in c]
    s = df[cols].mean().sort_values()
    fig = go.Figure(go.Bar(y=[nice(c) for c in s.index], x=s.values, orientation="h", marker_color=BLUE,
                           text=[pct(v) for v in s.values], textposition="outside",
                           hovertemplate="%{y}: %{x:.1%} selected<extra></extra>"))
    fig.update_xaxes(title="Share of respondents who selected it", tickformat=".0%", range=[0, max(.8, s.max() + .1)])
    top3 = ", ".join(f"{nice(c)} ({pct(v)})" for c, v in s.sort_values(ascending=False).head(3).items())
    bottom = nice(s.idxmin())
    meaning = {
        "Challenge_": ("These are the pains ReWear's value proposition must solve.",
                       "Lead marketing with the top pains. Price pains favour a value message; trust/counterfeit pains require verification built into the product, not added later."),
        "Barrier_": ("These block the SUPPLY side - why people do not already sell clothes.",
                     "Each top barrier maps to a service: no time -> pickup/concierge listing; don't know price -> AI price suggestions; payment security -> escrow."),
        "Category_": ("Category demand decides which inventory to recruit first.",
                      "Recruit sellers in the top categories at launch so buyers find relevant stock on the first visit (marketplace 'liquidity')."),
        "Feature_": ("Feature priorities decide what the MVP must contain.",
                     "Build the top 3-4 features into the MVP; defer low-ranked ones. This protects the AED 65k app budget."),
        "Trust_": ("Trust drivers show what makes a stranger's item believable.",
                   "Trust is the core product in resale. Make the top drivers visible on every listing page."),
    }[prefix]
    return style(fig, title, h=60 + 34 * len(cols)), dict(
        what=f"Top three: {top3}. Least selected: {bottom} ({pct(s.min())}). Average respondent ticked "
             f"{df[cols].sum(axis=1).mean():.1f} of {len(cols)} options.", why=meaning[0], so=meaning[1])


def fig_wtp(df):
    order = ORDERS["AED300_Item_WTP_Band"]
    s = df.AED300_Item_WTP_Band.value_counts().reindex(order).fillna(0)
    fig = go.Figure(go.Bar(x=order, y=s.values, marker_color=BLUE, text=[pct(v / s.sum()) for v in s.values],
                           textposition="outside", hovertemplate="AED %{x}: %{y:,}<extra></extra>"))
    fig.update_xaxes(title="AED willing to pay for an item that costs AED 300 new (excellent condition)")
    fig.update_yaxes(title="Respondents")
    buyers = df[df[WTP_TARGET] > 0][WTP_TARGET]
    return style(fig, "Q19 - Price acceptance for a AED 300 item"), dict(
        what=f"Among those who would buy, the median price is AED {buyers.median():.0f} ({pct(buyers.median() / 300)} of the new price); "
             f"mean AED {buyers.mean():.0f}. {pct((df[WTP_TARGET] == 0).mean(), 1)} would not buy at any price.",
        why="This is the regression target and the basis for average order value (AOV) in the financial model.",
        so=f"Pre-owned items need to be priced at about {pct(buyers.median() / 300)} of retail to clear. A typical AED 150-300 high-street "
           f"item therefore sells for roughly AED 60-150, which means low commission per order. That is the central economic challenge.")


def fig_wtp_summary(df):
    rows = [("Max spend on one item (Q16)", "Maximum_Item_Spend_AED_Midpoint"), ("Item worth AED 300 new (Q19)", WTP_TARGET),
            ("Monthly membership (Q17)", "Subscription_WTP_AED_Midpoint"), ("Delivery fee per order (Q18)", "Delivery_WTP_AED_Midpoint")]
    t = pd.DataFrame([{"Measure": a, "Median AED": df[c].median(), "Mean AED": df[c].mean(),
                       "Share at AED 0": (df[c] == 0).mean()} for a, c in rows])
    fig = go.Figure()
    fig.add_bar(y=t.Measure, x=t["Median AED"], orientation="h", name="Median", marker_color=BLUE,
                text=[f"AED {v:.0f}" for v in t["Median AED"]], textposition="outside")
    fig.update_xaxes(title="AED (median of band midpoints)", type="log")
    deliv = df.Delivery_WTP_AED_Midpoint
    return style(fig, "Willingness to pay across the four pricing questions", h=300), dict(
        what=f"Median delivery fee accepted: AED {deliv.median():.0f} ({pct((deliv == 0).mean())} want free delivery only). "
             f"Median membership: AED {df.Subscription_WTP_AED_Midpoint.median():.1f}/month.",
        why="Local courier cost per parcel in the UAE is well above what buyers say they will pay for delivery. The gap becomes a cost ReWear absorbs.",
        so="Do not launch a paid membership. Treat delivery as a cost to engineer down (batching, drop-off points) and test a buyer-protection fee instead.")


def fig_outlier_box(raw, clean):
    a = pd.to_numeric(raw.Monthly_Fashion_Spend_AED_Midpoint, errors="coerce").dropna()
    b = clean.Monthly_Fashion_Spend_AED_Midpoint
    fig = go.Figure()
    fig.add_box(x=a, name="Raw", marker_color=ORANGE, boxpoints="outliers", line_width=1.5)
    fig.add_box(x=b, name="Cleaned", marker_color=BLUE, boxpoints="outliers", line_width=1.5)
    fig.update_xaxes(title="Monthly fashion spend, AED (band midpoint)")
    return style(fig, "Fashion spend before and after cleaning", h=300), dict(
        what=f"Raw data ranges from AED {a.min():,.0f} to AED {a.max():,.0f}, including negative entries and values up to AED {a.max():,.0f} "
             f"that match no answer band. After cleaning the range is AED {b.min():,.0f}-{b.max():,.0f}.",
        why="A handful of extreme values would drag the mean spend and distort regression coefficients.",
        so="Outliers were flagged (Outlier_Flag) and capped, not deleted; the original values are kept in *_Raw columns for sensitivity checks.")


def fig_missing(raw):
    m = raw.isna().sum()
    m = m[m > 0].sort_values()
    fig = go.Figure(go.Bar(y=[c.replace("_", " ") for c in m.index], x=m.values, orientation="h", marker_color=ORANGE,
                           text=[f"{v} ({v / len(raw):.1%})" for v in m.values], textposition="outside"))
    fig.update_xaxes(title="Missing cells in raw data")
    return style(fig, "Missing values in the raw survey export", h=320), dict(
        what=f"{int(m.sum())} missing cells across {len(m)} columns; the worst column is {m.idxmax().replace('_', ' ')} ({m.max() / len(raw):.1%}).",
        why="Every missing rate is under 3%, so dropping respondents would waste good answers in the other 60+ columns.",
        so="Mode imputation for categories; spend midpoints re-derived from bands; Q19 WTP filled with the median of people in the same spend band.")


# --------------------------------------------------------------------------- diagnostic charts
def fig_rate_by(df, col, title=None):
    order = ordered(df, col)
    g = df.groupby(col)[TARGET].agg(["mean", "size"]).reindex(order)
    if col not in ORDERS:
        g = g.sort_values("mean")
    overall = df[TARGET].mean()
    colors = [BLUE if m >= overall else "#86b6ef" for m in g["mean"]]
    fig = go.Figure(go.Bar(x=g.index, y=g["mean"], marker_color=colors, customdata=g["size"],
                           text=[pct(v) for v in g["mean"]], textposition="outside",
                           hovertemplate="%{x}<br>Positive intent: %{y:.1%}<br>n = %{customdata:,}<extra></extra>"))
    fig.add_hline(y=overall, line_dash="dash", line_color=MUTED, annotation_text=f"All respondents {pct(overall)}",
                  annotation_position="top left", annotation_font_color=INK2)
    fig.update_yaxes(title="Positive intent rate", tickformat=".0%", range=[0, min(1, g["mean"].max() + .15)])
    v, p = cramers_v(df[col], df[TARGET])
    hi, lo = g["mean"].idxmax(), g["mean"].idxmin()
    gap = g["mean"].max() - g["mean"].min()
    strength = "strong" if v >= .3 else "moderate" if v >= .15 else "weak" if v >= .07 else "negligible"
    texts = {
        "Previous_Secondhand_Purchase": "Experience removes the stigma and uncertainty of buying used. This is the clearest adoption ladder in the data.",
        "Motivating_Discount": "People who need extreme discounts, or reject pre-owned outright, will not convert at any realistic price.",
        "Preferred_Shopping_Channel": "Shoppers already buying through digital marketplaces or resale apps face no new behaviour, so the switch to ReWear is small.",
        "Age_Group": "Younger groups are more open, but the gap is smaller than behaviour-based gaps, so age alone is a weak targeting rule.",
        "Emirate": "Differences between emirates are small once behaviour is considered; location matters for logistics more than for demand.",
        "Interest_in_Selling": "People willing to sell are also more willing to buy, a useful sign that one user can play both roles.",
        "Occupation": "Occupation differences reflect the hidden mix of budget-driven and premium shoppers inside each group.",
        "Monthly_Fashion_Spend_Band": "Spend level changes HOW MUCH people pay more than WHETHER they use ReWear.",
        "Shopping_Frequency": "Frequent shoppers carry more repeat potential, which matters more for lifetime value than for first conversion.",
        "Unused_Clothing_Behaviour": "People who already resell clothes are the most natural ReWear users on both sides.",
    }
    return style(fig, title or f"Positive intent by {col.replace('_', ' ').lower()}", h=380), dict(
        what=f"Highest intent: '{hi}' ({pct(g['mean'].max())}, n={g.loc[hi, 'size']:,}); lowest: '{lo}' ({pct(g['mean'].min())}). "
             f"Gap = {gap * 100:.0f} percentage points. Chi-square p = {p:.1e}; Cramér's V = {v:.2f} ({strength} association).",
        why=texts.get(col, ""),
        so=("Prioritise the high-intent group for launch campaigns." if v >= .15 else
            "Do not use this variable alone for targeting; behavioural variables separate users better.")
           + " This is an association in survey data, not proof of cause.")


def fig_corr(df):
    c = corr_matrix(df)
    mask = np.triu(np.ones_like(c, dtype=bool), k=1)
    z = c.where(~mask)
    fig = go.Figure(go.Heatmap(z=z.values, x=c.columns, y=c.index, zmin=-1, zmax=1, colorscale=DIV,
                               text=np.where(mask, "", c.round(2).astype(str).values), texttemplate="%{text}", textfont=dict(size=9),
                               hovertemplate="%{y} vs %{x}: ρ = %{z:.2f}<extra></extra>", colorbar=dict(title="ρ", thickness=12)))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(tickangle=-45)
    t = c["Intent to use (Q20)"].drop("Intent to use (Q20)").sort_values()
    w = c["Item WTP (Q19)"].drop(["Item WTP (Q19)", "Intent to use (Q20)"]).sort_values()
    return style(fig, "Spearman correlation matrix (ordinal + numeric variables)", h=640), dict(
        what=f"Intent (Q20) correlates most positively with {t.index[-1]} (ρ={t.iloc[-1]:.2f}) and {t.index[-2]} (ρ={t.iloc[-2]:.2f}), "
             f"and most negatively with {t.index[0]} (ρ={t.iloc[0]:.2f}). Item WTP (Q19) is driven by {w.index[-1]} (ρ={w.iloc[-1]:.2f}) "
             f"and falls with {w.index[0]} (ρ={w.iloc[0]:.2f}).",
        why="Spearman is used because most survey answers are ordered bands, not true numbers. Two separate stories appear: "
            "WHETHER people adopt depends on experience and price attitude; HOW MUCH they pay depends on budget.",
        so="Use behaviour (experience, discount attitude) to target who to acquire, and spend level to set prices. "
           "No pair is so highly correlated (|ρ|>0.8) that it causes serious multicollinearity in the models.")


def fig_drivers(df, top=15):
    d = driver_table(df).head(top).iloc[::-1]
    fig = go.Figure(go.Bar(y=[nice(x).replace("_", " ") for x in d.Driver], x=d.Cramers_V, orientation="h",
                           marker_color=[BLUE if s else "#c3c2b7" for s in d.Significant_5pct],
                           text=d.Cramers_V.map(lambda v: f"{v:.2f}"), textposition="outside",
                           hovertemplate="%{y}<br>Cramér's V %{x:.3f}<extra></extra>"))
    fig.update_xaxes(title="Cramér's V with Q20 intent (0 = none, 1 = perfect)")
    full = driver_table(df)
    return style(fig, f"Top {top} drivers of conversion intent (chi-square test)", h=520), dict(
        what=f"{int(full.Significant_5pct.sum())} of {len(full)} survey variables are significantly related to intent (p<0.05). "
             f"Strongest: {nice(full.Driver[0])} (V={full.Cramers_V[0]:.2f}), {nice(full.Driver[1])} (V={full.Cramers_V[1]:.2f}), "
             f"{nice(full.Driver[2])} (V={full.Cramers_V[2]:.2f}).",
        why="This ranks every question by how much it tells us about adoption, using one consistent test for categorical data.",
        so="The top drivers become the screening questions for a real pilot sign-up form and the core features of the classification model.")


def fig_wtp_by_spend(df):
    order = ordered(df, "Monthly_Fashion_Spend_Band")
    d = df[df[WTP_TARGET] > 0]
    fig = go.Figure()
    for b in order:
        fig.add_box(y=d.loc[d.Monthly_Fashion_Spend_Band == b, WTP_TARGET], name=b, marker_color=BLUE, boxmean=True, line_width=1.5)
    fig.update_yaxes(title="Q19 WTP for AED 300 item (AED)")
    fig.update_xaxes(title="Monthly fashion spend band (AED)")
    med = d.groupby("Monthly_Fashion_Spend_Band")[WTP_TARGET].median().reindex(order)
    rho = stats.spearmanr(d.Monthly_Fashion_Spend_Band_Ord, d[WTP_TARGET])[0]
    return style(fig, "Price acceptance rises with fashion budget", h=400), dict(
        what=f"Median WTP climbs from AED {med.iloc[0]:.0f} (spend {order[0]}) to AED {med.iloc[-1]:.0f} (spend {order[-1]}). "
             f"Spearman ρ = {rho:.2f}. Boxes overlap heavily, so spend is a driver but not the whole story.",
        why="This confirms Q4 as the main regression predictor and shows why one price rule cannot fit everyone.",
        so="Use spend band for price guidance (e.g. suggested listing price) and to decide which buyers see premium stock first.")


def fig_wtp_by_group(df):
    d = df[df[WTP_TARGET] > 0].copy()
    d["Group"] = np.where(d.Category_Designer_Luxury == 1, "Wants designer/luxury", "Does not")
    d["Experience"] = d.Previous_Secondhand_Purchase
    g = d.groupby(["Experience", "Group"])[WTP_TARGET].mean().reset_index()
    fig = px.bar(g, x="Experience", y=WTP_TARGET, color="Group", barmode="group",
                 category_orders={"Experience": ORDERS["Previous_Secondhand_Purchase"]}, color_discrete_sequence=[BLUE, ORANGE],
                 labels={WTP_TARGET: "Mean Q19 WTP (AED)"}, text_auto=".0f")
    lux = d.groupby("Group")[WTP_TARGET].mean()
    return style(fig, "WTP by resale experience and luxury interest", h=380, legend=True), dict(
        what=f"Respondents who want designer/luxury pay AED {lux['Wants designer/luxury']:.0f} on average vs AED {lux['Does not']:.0f}. "
             f"WTP also rises with second-hand experience in both groups.",
        why="Two independent levers raise price acceptance: premium taste and familiarity with resale.",
        so="A premium/authenticated section lifts AOV, the single biggest lever on commission per order (see Financials).")


def fig_challenge_feature(df):
    ch = [c for c in multi_cols(df, "Challenge_") if "No_Significant" not in c]
    fe = multi_cols(df, "Feature_")
    lift = pd.DataFrame(index=[nice(c) for c in ch], columns=[nice(f) for f in fe], dtype=float)
    for c in ch:
        for f in fe:
            pc = df.loc[df[c] == 1, f].mean(); pf = df[f].mean()
            lift.loc[nice(c), nice(f)] = pc / pf if pf else np.nan
    fig = go.Figure(go.Heatmap(z=lift.values, x=lift.columns, y=lift.index, colorscale=DIV, zmid=1, zmin=0.5, zmax=1.5,
                               text=lift.round(2).values, texttemplate="%{text}", textfont=dict(size=9),
                               hovertemplate="Has '%{y}' -> wants '%{x}': lift %{z:.2f}<extra></extra>", colorbar=dict(title="lift", thickness=12)))
    fig.update_xaxes(tickangle=-40)
    fig.update_yaxes(autorange="reversed")
    st = lift.stack().sort_values(ascending=False)
    return style(fig, "Which pain leads to which feature request? (lift)", h=460), dict(
        what=f"Strongest links: {st.index[0][0]} -> {st.index[0][1]} (lift {st.iloc[0]:.2f}), {st.index[1][0]} -> {st.index[1][1]} "
             f"(lift {st.iloc[1]:.2f}), {st.index[2][0]} -> {st.index[2][1]} (lift {st.iloc[2]:.2f}). Lift > 1 = more likely than average.",
        why="This connects the problem (Q9) to the solution (Q13): it shows which feature fixes which pain.",
        so="Message each feature to the people who have the matching pain, e.g. show the authenticity check to shoppers worried about counterfeits.")


def fig_supply(df):
    order = ORDERS["Interest_in_Selling"][::-1]
    t = pd.crosstab(df.Unused_Clothing_Behaviour, df.Interest_in_Selling, normalize="index")[order]
    t = t.loc[t["Definitely"].add(t["Probably"]).sort_values().index]
    colors = [BLUE, "#86b6ef", "#c3c2b7", "#f0a3a2", RED]
    fig = go.Figure()
    for c, col in zip(order, colors):
        fig.add_bar(y=t.index, x=t[c], name=c, orientation="h", marker_color=col,
                    hovertemplate=f"%{{y}}<br>{c}: %{{x:.1%}}<extra></extra>")
    fig.update_layout(barmode="stack")
    fig.update_xaxes(tickformat=".0%", title="Share of group")
    fig.update_traces(marker_line_color=SURFACE, marker_line_width=1)
    keep = df[df.Unused_Clothing_Behaviour == "Keep in wardrobe"]
    k_rate = keep.Seller_Prospect.mean()
    return style(fig, "Seller supply: interest in selling by what people do with unused clothes", h=380, legend=True), dict(
        what=f"{pct(len(keep) / len(df))} of respondents keep unused clothes in the wardrobe; {pct(k_rate)} of them would probably or "
             f"definitely sell on ReWear. Existing resellers show the highest seller interest.",
        why="Marketplaces usually fail on supply before demand. Idle wardrobes are the supply ReWear needs to unlock.",
        so=f"Target 'wardrobe keepers' with a low-effort concierge listing (pickup + photos). That is about "
           f"{int(len(keep) * k_rate):,} of {len(df):,} respondents ({pct(len(keep) * k_rate / len(df))}).")


def fig_barriers_sellers(df):
    ba = multi_cols(df, "Barrier_")
    g = df.groupby("Seller_Prospect")[ba].mean().T
    g.columns = ["Not a seller prospect", "Seller prospect"]
    g = g.sort_values("Seller prospect")
    fig = go.Figure()
    fig.add_bar(y=[nice(i) for i in g.index], x=g["Seller prospect"], name="Seller prospect", orientation="h", marker_color=BLUE)
    fig.add_bar(y=[nice(i) for i in g.index], x=g["Not a seller prospect"], name="Not a seller prospect", orientation="h", marker_color=ORANGE)
    fig.update_layout(barmode="group")
    fig.update_xaxes(tickformat=".0%", title="Share citing the barrier")
    top = g["Seller prospect"].idxmax()
    return style(fig, "Selling barriers: prospects vs non-prospects", h=420, legend=True), dict(
        what=f"Among willing sellers the top barrier is '{nice(top)}' ({pct(g.loc[top, 'Seller prospect'])}). "
             f"Non-prospects are far more likely to say 'Not interested' ({pct(g.loc['Barrier_Not_Interested', 'Not a seller prospect'])}).",
        why="Willing sellers are held back by fixable friction (time, process, pricing), whereas non-prospects are simply not interested.",
        so="Spend on removing friction for willing sellers; do not spend on persuading the uninterested.")


# --------------------------------------------------------------------------- models
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve,
                             confusion_matrix, mean_absolute_error, mean_squared_error, r2_score, silhouette_score)


def classification_X(df):
    num = ["Maximum_Item_Spend_AED_Midpoint", "Subscription_WTP_AED_Midpoint", "Delivery_WTP_AED_Midpoint", WTP_TARGET]
    X = pd.get_dummies(df[CATEGORICAL + multi_cols(df) + num], drop_first=True).astype(float)
    return X, df[TARGET].astype(int)


def run_classification(df, seed=42):
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.svm import SVC
    X, y = classification_X(df)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, random_state=seed, stratify=y)
    models = {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=4000, C=0.3)),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, min_samples_leaf=25, random_state=seed),
        "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=seed),
        "Gradient Boosting": GradientBoostingClassifier(random_state=seed),
        "K-Nearest Neighbours": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=35)),
        "Naive Bayes": GaussianNB(),
        "Support Vector Machine": make_pipeline(StandardScaler(), SVC(C=0.5, probability=True, random_state=seed)),
    }
    rows, rocs, cms, fitted = [], {}, {}, {}
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    for name, m in models.items():
        m.fit(Xtr, ytr)
        pr = m.predict(Xte); sc = m.predict_proba(Xte)[:, 1]
        cv_auc = cross_val_score(m, Xtr, ytr, cv=cv, scoring="roc_auc", n_jobs=-1)
        rows.append({"Model": name, "Accuracy": accuracy_score(yte, pr), "Precision": precision_score(yte, pr),
                     "Recall": recall_score(yte, pr), "F1": f1_score(yte, pr), "ROC_AUC": roc_auc_score(yte, sc),
                     "CV_AUC_mean": cv_auc.mean(), "CV_AUC_sd": cv_auc.std()})
        fpr, tpr, _ = roc_curve(yte, sc); rocs[name] = (fpr, tpr)
        cms[name] = confusion_matrix(yte, pr); fitted[name] = m
    res = pd.DataFrame(rows).sort_values("ROC_AUC", ascending=False).reset_index(drop=True)
    best = res.Model[0]
    rf = fitted["Random Forest"]
    imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    lr = fitted["Logistic Regression"][-1]
    coef = pd.Series(lr.coef_[0], index=X.columns).sort_values()
    return dict(results=res, rocs=rocs, cms=cms, best=best, importance=imp, logit_coef=coef,
                n_train=len(Xtr), n_test=len(Xte), base_rate=float(y.mean()))


def fig_roc(cls):
    fig = go.Figure()
    for i, (name, (f, t)) in enumerate(cls["rocs"].items()):
        auc = cls["results"].set_index("Model").loc[name, "ROC_AUC"]
        fig.add_scatter(x=f, y=t, mode="lines", name=f"{name} ({auc:.3f})", line=dict(width=2, color=CAT[i % 8]))
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color=MUTED, width=1), name="Random guess", showlegend=False)
    fig.update_xaxes(title="False positive rate", range=[0, 1]); fig.update_yaxes(title="True positive rate", range=[0, 1.02])
    fig = style(fig, "ROC curves - all classifiers (test set)", h=460, legend=True)
    fig.update_layout(legend=dict(orientation="v", x=.55, y=.05, yanchor="bottom"))
    return fig


def fig_cm(cls, name=None):
    name = name or cls["best"]
    cm = cls["cms"][name]
    labels = ["No intent (0)", "Intent (1)"]
    txt = [[f"{v:,}<br>{v / cm.sum():.1%}" for v in row] for row in cm]
    fig = go.Figure(go.Heatmap(z=cm, x=labels, y=labels, colorscale=[[0, "#cde2fb"], [1, "#184f95"]], showscale=False,
                               text=txt, texttemplate="%{text}", textfont=dict(size=14)))
    fig.update_xaxes(title="Predicted", side="bottom"); fig.update_yaxes(title="Actual", autorange="reversed")
    return style(fig, f"Confusion matrix - {name}", h=360)


def fig_importance(cls, top=15):
    s = cls["importance"].head(top).iloc[::-1]
    fig = go.Figure(go.Bar(y=[nice(i).replace("_", " ") for i in s.index], x=s.values, orientation="h", marker_color=BLUE))
    fig.update_xaxes(title="Random Forest importance (mean decrease in impurity)")
    return style(fig, f"Top {top} predictors of intent", h=480)


CLUSTER_FEATURES = ["Age_Group_Ord", "Monthly_Fashion_Spend_Band_Ord", "Shopping_Frequency_Ord", "Previous_Secondhand_Purchase_Ord",
                    "Interest_in_Selling_Ord", "Discount_Needed_Pct", "Digital_Channel", "Rejects_Preowned",
                    "Category_Designer_Luxury", "Category_Childrenswear", "Category_Shoes_Sneakers", "Category_Sportswear",
                    "Feature_Authenticity_Check", "Feature_Home_Delivery", "Feature_Wardrobe_Valuation", "Feature_Clothing_Exchange",
                    "Feature_AI_Recommendations", "Challenge_Affordability", "Challenge_Online_Seller_Trust",
                    "Challenge_Counterfeit_Concerns", "Barrier_No_Time", "Barrier_Not_Interested", "Barrier_Payment_Security",
                    WTP_TARGET]

SEGMENT_SIGNATURES = {   # used only to NAME clusters after they are found
    "Young Value Seekers": {"Challenge_Affordability": 1, "Age_Group_Ord": -1, "Feature_Clothing_Exchange": 1, "Category_Shoes_Sneakers": 1},
    "Premium Brand Hunters": {"Category_Designer_Luxury": 1, "Feature_Authenticity_Check": 1, WTP_TARGET: 1, "Monthly_Fashion_Spend_Band_Ord": 1},
    "Busy Declutterers": {"Category_Childrenswear": 1, "Barrier_No_Time": 1, "Feature_Wardrobe_Valuation": 1, "Interest_in_Selling_Ord": 1},
    "Cautious Mainstream": {"Challenge_Online_Seller_Trust": 1, "Barrier_Payment_Security": 1, "Digital_Channel": -1},
    "Resistant Traditionalists": {"Rejects_Preowned": 1, "Barrier_Not_Interested": 1, "Previous_Secondhand_Purchase_Ord": -1},
}


def run_clustering(df, k=None, seed=42):
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from scipy.optimize import linear_sum_assignment
    Z = StandardScaler().fit_transform(df[CLUSTER_FEATURES])
    ktab = []
    for kk in range(2, 9):
        km = KMeans(kk, n_init=10, random_state=seed).fit(Z)
        ktab.append({"k": kk, "Inertia": km.inertia_, "Silhouette": silhouette_score(Z, km.labels_, sample_size=2500, random_state=seed)})
    ktab = pd.DataFrame(ktab)
    k = k or int(ktab.loc[ktab.k.between(3, 7)].sort_values("Silhouette", ascending=False).k.iloc[0])
    km = KMeans(k, n_init=20, random_state=seed).fit(Z)
    lab = km.labels_
    zdf = pd.DataFrame(Z, columns=CLUSTER_FEATURES); zdf["c"] = lab
    zc = zdf.groupby("c").mean()
    names = list(SEGMENT_SIGNATURES)
    score = np.array([[sum(w * zc.loc[c, f] for f, w in SEGMENT_SIGNATURES[n].items()) for n in names] for c in zc.index])
    r, cidx = linear_sum_assignment(-score)
    name_map = {c: names[j] for c, j in zip(r, cidx)}
    for c in zc.index:
        name_map.setdefault(c, f"Segment {c}")
    seg = pd.Series(lab).map(name_map).values
    d = df.assign(Segment=seg)
    prof = d.groupby("Segment").agg(
        Respondents=("Respondent_ID", "count"), Share=("Respondent_ID", lambda s: len(s) / len(d)),
        Positive_Intent=(TARGET, "mean"), Seller_Prospect=("Seller_Prospect", "mean"),
        Avg_Fashion_Spend_AED=("Monthly_Fashion_Spend_AED_Midpoint", "mean"), Avg_Q19_WTP_AED=(WTP_TARGET, "mean"),
        Avg_Max_Item_Spend_AED=("Maximum_Item_Spend_AED_Midpoint", "mean"),
        Pct_Age_18_30=("Age_Group", lambda s: s.isin(["18-24", "25-30"]).mean()),
        Pct_Bought_Secondhand=("Previous_Secondhand_Purchase", lambda s: (s != "Never").mean()),
        Pct_Want_Designer=("Category_Designer_Luxury", "mean"), Pct_Want_Childrenswear=("Category_Childrenswear", "mean"),
        Pct_Affordability_Pain=("Challenge_Affordability", "mean"), Pct_Trust_Pain=("Challenge_Online_Seller_Trust", "mean"),
        Pct_No_Time_To_Sell=("Barrier_No_Time", "mean"), Pct_Reject_Preowned=("Rejects_Preowned", "mean"),
        Top_Emirate=("Emirate", lambda s: s.mode()[0]), Top_Channel=("Preferred_Shopping_Channel", lambda s: s.mode()[0]),
    ).sort_values("Positive_Intent", ascending=False)
    pca = PCA(2, random_state=seed).fit(Z)
    P = pca.transform(Z)
    sil = float(ktab.set_index("k").loc[k, "Silhouette"])
    return dict(k=k, ktab=ktab, labels=lab, segment=seg, profile=prof, zprofile=zc.rename(index=name_map),
                pca=P, pca_var=pca.explained_variance_ratio_, silhouette=sil)


def fig_k(clu):
    t = clu["ktab"]
    fig = go.Figure()
    fig.add_scatter(x=t.k, y=t.Silhouette, mode="lines+markers", line=dict(color=BLUE, width=2), marker=dict(size=9), name="Silhouette")
    fig.add_vline(x=clu["k"], line_dash="dash", line_color=MUTED, annotation_text=f"chosen k = {clu['k']}")
    fig.update_xaxes(title="Number of clusters (k)", dtick=1); fig.update_yaxes(title="Silhouette score")
    return style(fig, "Choosing k: silhouette score", h=320)


def fig_elbow(clu):
    t = clu["ktab"]
    fig = go.Figure(go.Scatter(x=t.k, y=t.Inertia, mode="lines+markers", line=dict(color=BLUE, width=2), marker=dict(size=9)))
    fig.update_xaxes(title="Number of clusters (k)", dtick=1); fig.update_yaxes(title="Within-cluster sum of squares")
    return style(fig, "Elbow curve", h=320)


def fig_cluster_scatter(clu, df):
    P = clu["pca"]
    d = pd.DataFrame({"PC1": P[:, 0], "PC2": P[:, 1], "Segment": clu["segment"]})
    order = list(clu["profile"].index)
    fig = px.scatter(d.sample(min(2500, len(d)), random_state=1), x="PC1", y="PC2", color="Segment", opacity=.6,
                     category_orders={"Segment": order}, color_discrete_sequence=CAT)
    fig.update_traces(marker=dict(size=6, line=dict(width=0)))
    v = clu["pca_var"]
    fig.update_xaxes(title=f"Principal component 1 ({v[0]:.0%} of variance)"); fig.update_yaxes(title=f"Principal component 2 ({v[1]:.0%})")
    return style(fig, "Segments projected onto two principal components", h=480, legend=True)


def fig_cluster_heat(clu):
    z = clu["zprofile"].loc[list(clu["profile"].index)]
    labels = {"Age_Group_Ord": "Age", "Monthly_Fashion_Spend_Band_Ord": "Fashion spend", "Shopping_Frequency_Ord": "Shop frequency",
              "Previous_Secondhand_Purchase_Ord": "Resale experience", "Interest_in_Selling_Ord": "Seller interest",
              "Discount_Needed_Pct": "Discount needed", "Digital_Channel": "Digital channel", "Rejects_Preowned": "Rejects pre-owned",
              WTP_TARGET: "Q19 WTP"}
    cols = [labels.get(c, nice(c)) for c in z.columns]
    fig = go.Figure(go.Heatmap(z=z.values, x=cols, y=z.index, colorscale=DIV, zmid=0, zmin=-1.5, zmax=1.5,
                               text=z.round(1).values, texttemplate="%{text}", textfont=dict(size=9),
                               hovertemplate="%{y}<br>%{x}: %{z:.2f} SD from average<extra></extra>",
                               colorbar=dict(title="SD vs avg", thickness=12)))
    fig.update_xaxes(tickangle=-45); fig.update_yaxes(autorange="reversed")
    return style(fig, "Segment profiles (standardised: red = above average, blue = below)", h=430)


def run_association(df, min_support=.05, min_conf=.5, min_lift=1.3):
    from mlxtend.frequent_patterns import apriori, association_rules
    cols = [c for c in multi_cols(df) if "No_Significant" not in c]
    fi = apriori(df[cols].astype(bool), min_support=min_support, use_colnames=True, max_len=3)
    r = association_rules(fi, metric="lift", min_threshold=min_lift)
    r = r[r.confidence >= min_conf].copy()
    r["Antecedent"] = r.antecedents.map(lambda s: " + ".join(sorted(nice(x) for x in s)))
    r["Consequent"] = r.consequents.map(lambda s: " + ".join(sorted(nice(x) for x in s)))
    r["Antecedent_Blocks"] = r.antecedents.map(lambda s: "/".join(sorted({x.split("_")[0] for x in s})))
    r["Consequent_Block"] = r.consequents.map(lambda s: "/".join(sorted({x.split("_")[0] for x in s})))
    r["Rule_Length"] = r.antecedents.map(len) + r.consequents.map(len)
    r = r[["Antecedent", "Consequent", "Antecedent_Blocks", "Consequent_Block", "Rule_Length", "support", "confidence", "lift",
           "leverage", "conviction"]].rename(columns=str.title)
    r = r.sort_values(["Lift", "Support"], ascending=False).reset_index(drop=True)
    return dict(rules=r, n_itemsets=len(fi), n_items=len(cols))


def fig_rules(ar):
    r = ar["rules"]
    fig = px.scatter(r, x="Support", y="Confidence", color="Lift", size="Lift", size_max=14,
                     color_continuous_scale=[[0, "#cde2fb"], [1, "#0d366b"]], hover_data=["Antecedent", "Consequent"])
    fig.update_traces(marker=dict(line=dict(width=0)))
    fig.update_xaxes(tickformat=".0%"); fig.update_yaxes(tickformat=".0%")
    fig = style(fig, f"{len(r):,} association rules: support vs confidence (colour = lift)", h=420)
    fig.update_layout(coloraxis_colorbar=dict(thickness=12))
    return fig


def run_regression(df, seed=42):
    from sklearn.linear_model import LinearRegression, Ridge, Lasso, RidgeCV, LassoCV
    from sklearn.ensemble import RandomForestRegressor
    X = pd.get_dummies(df[CATEGORICAL + multi_cols(df) + ["Log_Fashion_Spend"]].drop(columns=["Monthly_Fashion_Spend_Band"]),
                       drop_first=True).astype(float)
    y = df[WTP_TARGET]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, random_state=seed)
    ridge_a = RidgeCV(alphas=np.logspace(-2, 3, 30)).fit(StandardScaler().fit_transform(Xtr), ytr).alpha_
    lasso_a = LassoCV(cv=5, random_state=seed, alphas=40).fit(StandardScaler().fit_transform(Xtr), ytr).alpha_
    models = {"Linear Regression": make_pipeline(StandardScaler(), LinearRegression()),
              "Ridge Regression": make_pipeline(StandardScaler(), Ridge(alpha=ridge_a)),
              "Lasso Regression": make_pipeline(StandardScaler(), Lasso(alpha=lasso_a, max_iter=20000)),
              "Random Forest Regressor": RandomForestRegressor(n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=seed)}
    rows, preds, fitted = [], {}, {}
    for n, m in models.items():
        m.fit(Xtr, ytr); p = m.predict(Xte)
        rows.append({"Model": n, "MAE_AED": mean_absolute_error(yte, p), "RMSE_AED": mean_squared_error(yte, p) ** .5,
                     "R2": r2_score(yte, p), "Alpha": ridge_a if "Ridge" in n else lasso_a if "Lasso" in n else np.nan})
        preds[n] = p; fitted[n] = m
    res = pd.DataFrame(rows).sort_values("MAE_AED").reset_index(drop=True)
    coefs = pd.DataFrame({n: pd.Series(fitted[n][-1].coef_, index=X.columns) for n in ["Linear Regression", "Ridge Regression", "Lasso Regression"]})
    coefs["abs"] = coefs["Lasso Regression"].abs()
    coefs = coefs.sort_values("abs", ascending=False).drop(columns="abs")
    return dict(results=res, coefs=coefs, y_test=yte.values, preds=preds, lasso_zeroed=int((coefs["Lasso Regression"] == 0).sum()),
                n_features=X.shape[1], best=res.Model[0])


def fig_reg_scatter(reg, name=None):
    name = name or reg["best"]
    y, p = reg["y_test"], reg["preds"][name]
    rng = np.random.default_rng(0); jitter = rng.normal(0, 2.5, len(y))
    fig = go.Figure(go.Scatter(x=y + jitter, y=p, mode="markers", marker=dict(size=6, color=BLUE, opacity=.35, line=dict(width=0)),
                               hovertemplate="Actual %{x:.0f}<br>Predicted %{y:.0f}<extra></extra>"))
    fig.add_scatter(x=[0, 230], y=[0, 230], mode="lines", line=dict(dash="dash", color=MUTED, width=1))
    fig.update_xaxes(title="Actual Q19 WTP (AED, band midpoint, jittered)"); fig.update_yaxes(title="Predicted WTP (AED)")
    return style(fig, f"Actual vs predicted - {name}", h=420)


def fig_coefs(reg, top=15):
    c = reg["coefs"].head(top).iloc[::-1]
    fig = go.Figure()
    for i, n in enumerate(["Linear Regression", "Ridge Regression", "Lasso Regression"]):
        fig.add_bar(y=[nice(x).replace("_", " ") for x in c.index], x=c[n], name=n, orientation="h", marker_color=CAT[i])
    fig.update_layout(barmode="group")
    fig.update_xaxes(title="Standardised coefficient (AED per 1 SD change)")
    return style(fig, f"Top {top} WTP drivers: Linear vs Ridge vs Lasso", h=620, legend=True)


# --------------------------------------------------------------------------- finance
def unit_economics(aov=150, take_rate=.15, payment_pct=.029, payment_fixed=1.0, ops_per_order=4.0,
                   courier_cost=18.0, buyer_delivery_fee=5.0, fixed_monthly=22500, cac=60, orders_per_buyer_year=3.0,
                   buyer_retention_years=1.5):
    revenue = aov * take_rate
    payment = aov * payment_pct + payment_fixed
    delivery_subsidy = max(courier_cost - buyer_delivery_fee, 0)
    contribution = revenue - payment - ops_per_order - delivery_subsidy
    be_orders_m = fixed_monthly / contribution if contribution > 0 else np.inf
    ltv = contribution * orders_per_buyer_year * buyer_retention_years
    return dict(revenue_per_order=revenue, payment_cost=payment, ops_cost=ops_per_order, delivery_subsidy=delivery_subsidy,
                contribution=contribution, contribution_margin_gmv=contribution / aov, breakeven_orders_month=be_orders_m,
                breakeven_gmv_year=be_orders_m * 12 * aov, breakeven_active_buyers=be_orders_m * 12 / orders_per_buyer_year,
                ltv=ltv, ltv_cac=ltv / cac if cac else np.nan, cac=cac)


def scenarios(df):
    q18 = float(df.Delivery_WTP_AED_Midpoint.mean())
    q19 = float(df.loc[df[WTP_TARGET] > 0, WTP_TARGET].median())   # typical accepted price for a mid-range item
    base = dict(aov=q19, take_rate=.15, buyer_delivery_fee=q18)
    sc = {
        "A. Mass-market, ReWear subsidises delivery": base,
        "B. Mass-market, buyer pays full delivery": dict(base, buyer_delivery_fee=18.0),
        "C. Premium mix (AOV 350, 20% take, buyer pays delivery)": dict(aov=350, take_rate=.20, buyer_delivery_fee=18.0, ops_per_order=9.0, orders_per_buyer_year=2.0),
        "D. Blended: 70% mass (B) + 30% premium (C)": None,
    }
    rows = {}
    for k, v in sc.items():
        if v is None:
            b, c = rows["B. Mass-market, buyer pays full delivery"], rows["C. Premium mix (AOV 350, 20% take, buyer pays delivery)"]
            aov = .7 * q19 + .3 * 350
            contrib = .7 * b["contribution"] + .3 * c["contribution"]
            be = 22500 / contrib
            rows[k] = dict(aov=aov, revenue_per_order=.7 * b["revenue_per_order"] + .3 * c["revenue_per_order"], contribution=contrib,
                           contribution_margin_gmv=contrib / aov, breakeven_orders_month=be, breakeven_gmv_year=be * 12 * aov,
                           breakeven_active_buyers=be * 12 / (.7 * 3 + .3 * 2), ltv=.7 * b["ltv"] + .3 * c["ltv"],
                           ltv_cac=(.7 * b["ltv"] + .3 * c["ltv"]) / 60)
        else:
            u = unit_economics(**v); u["aov"] = v["aov"]; rows[k] = u
    t = pd.DataFrame(rows).T
    return t[["aov", "revenue_per_order", "contribution", "contribution_margin_gmv", "breakeven_orders_month",
              "breakeven_gmv_year", "breakeven_active_buyers", "ltv", "ltv_cac"]]


MARKET_EVIDENCE = [
    ("UAE e-commerce market, 2025", "AED 42.2bn; 19% CAGR 2020-25; forecast 9.9% CAGR to AED 67.2bn by 2030",
     "EZDubai / Euromonitor via WAM & Zawya (2026)"),
    ("UAE apparel & footwear e-commerce, 2025", "AED 11.4bn - the largest UAE e-commerce vertical", "WAM (2026)"),
    ("Global second-hand apparel, 2025", "≈ USD 257bn, ~10% of global apparel spend; ~9% CAGR to USD 393bn by 2030",
     "ThredUp / GlobalData Resale Report 2026"),
    ("Consumers buying second-hand", "59% of global respondents shopped second-hand apparel in 2025", "ThredUp Resale Report 2026"),
    ("UAE resale market size", "No official figure. Estimates range from ≈ USD 341m (luxury-led, conservative) to USD 5.2bn (thrift-inclusive)",
     "Khaleej Times (2025) citing KD Market Insights; Mobility Foresights (2025)"),
    ("UAE textile waste", "≈ 220,000 tonnes a year; Naseej national circularity initiative launched June 2026", "The National / WAM (2026)"),
    ("Benchmark marketplace economics", "Vinted 2025: GMV €10.8bn, revenue €1.1bn (≈10% take rate), net profit €62m - profitable only from 2023",
     "Vinted 2025 results via Retail Insight Network / Sharecast (2026)"),
    ("Local competition", "Yalla BB (2026 app launch), Bazaara, Luved, Taggy, REVYN, The Luxury Closet, Gemaee, Dubizzle, Thrift for Good",
     "Gulf News, Images RetailME, MENA Startup Digest, Entrepreneur ME (2025-26)"),
]


# =========================================================================== STRATEGY
# Brings the four algorithms together into one decision:
#   classification -> conversion probability -> High / Medium / Low tiers (who is likely to buy)
#   clustering     -> segment / persona (who they are, how to reach them)
#   regression     -> predicted willingness to pay (what price they accept)
#   association    -> cross-sell rules inside each segment (what else to offer)
# and feeds them into a month-by-month break-even projection.
STRATEGY_ASSUMPTIONS = {
    "Base CAC (AED per new buyer, untargeted)": 60, "ReWear share of a buyer's fashion purchases": 0.25,
    "Monthly buyer churn": 0.05, "Courier cost per parcel (AED)": 18, "Mass take rate": 0.15, "Premium take rate": 0.20,
    "Premium AOV (AED)": 350, "Ops cost per mass order (AED)": 4, "Ops cost per premium order (AED, incl. authentication)": 9,
    "Buyer-protection fee": "5% of item + AED 2 (recommended strategy only)", "Payment cost": "2.9% + AED 1",
    "Starting cash after one-off setup (AED)": 105000, "Fixed costs per month (AED)": "Y1 15,000 · Y2 22,500 · Y3 30,000",
    "Acquisition budget per month (AED)": "M1-6 10,000 · M7-12 15,000 · Y2 20,000 · Y3 25,000",
    "Premium Brand Hunters share of new buyers (recommended)": 0.40,
}
FREQ_PER_MONTH = {"Several times a month": 3, "Monthly": 1, "Every 2-3 months": 0.4, "Every 6 months": 0.17, "Less often": 0.08}


def _contrib(aov, take, ops, bpf=False, courier=18.0, buyer_fee=18.0):
    rev = aov * take + ((aov * 0.05 + 2) if bpf else 0)
    return rev - (aov * 0.029 + 1) - ops - max(courier - buyer_fee, 0)


def budget_m(m):
    return 10000 if m <= 6 else 15000 if m <= 12 else 20000 if m <= 24 else 25000


def fixed_m(m):
    return 15000 if m <= 12 else 22500 if m <= 24 else 30000


def project(cac, opm, cm, months=36, churn=0.05, cash0=105000, budget=budget_m, fixed=fixed_m):
    rows, A, cash = [], 0.0, cash0
    for m in range(1, months + 1):
        B = budget(m); new = B / cac; A = A * (1 - churn) + new; o = A * opm
        contrib = o * cm; profit = contrib - fixed(m) - B; cash += profit
        rows.append({"month": m, "new_buyers": new, "active_buyers": A, "orders": o, "contribution": contrib,
                     "marketing": B, "fixed": fixed(m), "profit": profit, "cash": cash})
    t = pd.DataFrame(rows)
    be = t.loc[t.profit >= 0, "month"]
    return t, (int(be.iloc[0]) if len(be) else None)


def run_strategy(df, clu, ar, seed=42):
    from sklearn.model_selection import cross_val_predict, StratifiedKFold, KFold
    from sklearn.linear_model import LogisticRegression, Lasso
    X, y = classification_X(df)
    p = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(max_iter=4000, C=0.3)), X, y,
                          cv=StratifiedKFold(5, shuffle=True, random_state=seed), method="predict_proba")[:, 1]
    RX = pd.get_dummies(df[CATEGORICAL + multi_cols(df) + ["Log_Fashion_Spend"]].drop(columns=["Monthly_Fashion_Spend_Band"]), drop_first=True).astype(float)
    w = np.clip(cross_val_predict(make_pipeline(StandardScaler(), Lasso(alpha=0.33, max_iter=20000)), RX, df[WTP_TARGET],
                                  cv=KFold(5, shuffle=True, random_state=seed)), 0, 300)
    d = df.assign(Conversion_Probability=p.round(3), Predicted_WTP_AED=w.round(1), Segment=clu["segment"],
                  Conversion_Tier=np.select([p >= 0.7, p >= 0.4], ["High", "Medium"], "Low"),
                  Fashion_Buys_Per_Month=df.Shopping_Frequency.map(FREQ_PER_MONTH))
    base = float(d[TARGET].mean())

    def action(seg, tier):
        if seg == "Resistant Traditionalists" or tier == "Low": return "Do not target"
        if tier == "High": return "Target now"
        if seg == "Cautious Mainstream": return "Later (after trust is proven)"
        return "Nurture (retarget)"
    tm = d.groupby(["Segment", "Conversion_Tier"]).agg(
        Respondents=("Respondent_ID", "count"), Avg_Probability=("Conversion_Probability", "mean"),
        Actual_Intent=(TARGET, "mean"), Avg_Predicted_WTP=("Predicted_WTP_AED", "mean"), Seller_Prospect=("Seller_Prospect", "mean")).reset_index()
    tm["Share"] = tm.Respondents / len(d)
    tm["Action"] = [action(s, t) for s, t in zip(tm.Segment, tm.Conversion_Tier)]
    d["Action"] = [action(s, t) for s, t in zip(d.Segment, d.Conversion_Tier)]

    # price points by segment and by demographic, from regression predictions
    ps = d.groupby("Segment").agg(Respondents=("Respondent_ID", "count"), Median_Predicted_WTP=("Predicted_WTP_AED", "median"),
                                  P25=("Predicted_WTP_AED", lambda s: s.quantile(.25)), P75=("Predicted_WTP_AED", lambda s: s.quantile(.75)),
                                  Avg_Conversion_Prob=("Conversion_Probability", "mean"), Delivery_WTP=("Delivery_WTP_AED_Midpoint", "mean"),
                                  Membership_WTP=("Subscription_WTP_AED_Midpoint", "mean"), Max_Item_Spend=("Maximum_Item_Spend_AED_Midpoint", "median")).reset_index()
    ps["Price_as_pct_of_new"] = ps.Median_Predicted_WTP / 300
    pdemo = []
    for col in ["Age_Group", "Emirate", "Occupation", "Monthly_Fashion_Spend_Band"]:
        for v in ordered(d, col):
            s = d[d[col] == v]
            pdemo.append({"Dimension": col.replace("_", " "), "Group": v, "Respondents": len(s), "Median_Predicted_WTP": s.Predicted_WTP_AED.median(),
                          "Avg_Conversion_Prob": s.Conversion_Probability.mean(), "Target_Now_Share": (s.Action == "Target now").mean()})
    pdemo = pd.DataFrame(pdemo)

    # cross-sell playbook: strongest rules inside each segment (consequent = category or feature we can offer)
    from mlxtend.frequent_patterns import apriori, association_rules
    cols = [c for c in multi_cols(df) if "No_Significant" not in c]
    cs = []
    for seg in [s for s in d.Segment.unique() if s != "Resistant Traditionalists"]:
        sd = d[d.Segment == seg]
        fi = apriori(sd[cols].astype(bool), min_support=0.06, use_colnames=True, max_len=2)
        r = association_rules(fi, metric="lift", min_threshold=1.1)
        first = lambda col: col.map(lambda z: next(iter(z)))
        a, c = first(r.antecedents), first(r.consequents)
        svc = r[a.str.startswith(("Challenge_", "Barrier_")) & c.str.startswith("Feature_") & (r.confidence >= 0.40)]
        prod = r[a.str.startswith("Category_") & c.str.startswith("Category_") & (r.confidence >= 0.30)]
        picks = []
        for frame, k in [(svc, 2), (prod, 1)]:
            frame = frame.assign(score=frame.confidence * frame.lift).sort_values("score", ascending=False)
            seen = set()
            for _, x in frame.iterrows():
                cons = next(iter(x.consequents))
                if cons in seen: continue
                seen.add(cons); picks.append(x)
                if len(seen) == k: break
        for x in picks:
            cons = next(iter(x.consequents))
            cs.append({"Segment": seg, "If_they_pick": nice(next(iter(x.antecedents))), "Offer_them": nice(cons),
                       "Offer_type": "Product category" if cons.startswith("Category_") else "Service / feature",
                       "Support_in_segment": x.support, "Confidence": x.confidence, "Lift": x.lift})
    cs = pd.DataFrame(cs)

    # break-even projection: three strategies
    def mixstats(pool, premium_share=None):
        segs = pool.Segment.value_counts(normalize=True)
        if premium_share is not None:
            other = segs.drop("Premium Brand Hunters", errors="ignore"); other = other / other.sum() * (1 - premium_share)
            segs = pd.concat([pd.Series({"Premium Brand Hunters": premium_share}), other])
        fpm = pool.groupby("Segment").Fashion_Buys_Per_Month.mean()
        aov = pool[pool[WTP_TARGET] > 0].groupby("Segment")[WTP_TARGET].median()
        return segs, fpm, aov
    out = {}
    share = STRATEGY_ASSUMPTIONS["ReWear share of a buyer's fashion purchases"]
    # S1 unfocused mass market: everyone with positive intent, ReWear pays delivery beyond the buyer's stated fee
    pos = d[d[TARGET] == 1]
    segs, fpm, aov = mixstats(pos)
    q18 = float(d.Delivery_WTP_AED_Midpoint.mean()); q19 = float(d.loc[d[WTP_TARGET] > 0, WTP_TARGET].median())
    cm1 = _contrib(q19, .15, 4, buyer_fee=q18); opm1 = float((segs * fpm).sum()) * share
    out["S1"] = dict(name="Unfocused mass-market launch", cac=60.0, cm=cm1, opm=opm1, premium=float(segs.get("Premium Brand Hunters", 0)),
                     how="Advertise to everyone; typical AED 113 items; ReWear absorbs delivery above the AED 3-4 buyers will pay")
    # S2 model-targeted: buy only High-tier prospects (cheaper CAC), buyer pays delivery, no extra fee
    tgt = d[d.Action == "Target now"]
    segs, fpm, aov = mixstats(tgt)
    prec = float(tgt[TARGET].mean())
    cac_t = 60 * base / prec
    def seg_cm(s, bpf):
        return _contrib(350, .20, 9, bpf) if s == "Premium Brand Hunters" else _contrib(float(aov.get(s, q19)), .15, 4, bpf)
    cm2 = float(sum(segs[s] * seg_cm(s, False) for s in segs.index)); opm2 = float((segs * fpm.reindex(segs.index)).sum()) * share
    out["S2"] = dict(name="Model-targeted, buyer pays delivery", cac=cac_t, cm=cm2, opm=opm2, premium=float(segs.get("Premium Brand Hunters", 0)),
                     how="Spend only on High-probability prospects (classification); buyers pay full delivery")
    # S3 recommended: model-targeted + premium-led mix (40% premium hunters) + buyer-protection fee
    ps3 = STRATEGY_ASSUMPTIONS["Premium Brand Hunters share of new buyers (recommended)"]
    pool3 = d[(d.Action == "Target now") | ((d.Segment == "Premium Brand Hunters") & (d.Conversion_Tier == "Medium"))]
    segs, fpm, aov = mixstats(pool3, premium_share=ps3)
    prec3 = float(pool3[TARGET].mean()); cac3 = 60 * base / prec3
    cm3 = float(sum(segs[s] * seg_cm(s, True) for s in segs.index)); opm3 = float((segs * fpm.reindex(segs.index)).sum()) * share
    out["S3"] = dict(name="Recommended: premium-led, targeted, buyer-protection fee", cac=cac3, cm=cm3, opm=opm3, premium=ps3,
                     how="Target High-tier + premium Medium-tier; 40% of acquisition spend on Premium Brand Hunters; 5% + AED 2 buyer-protection fee; buyer pays delivery")
    proj, summ = [], []
    for k, s in out.items():
        t, be = project(s["cac"], s["opm"], s["cm"])
        t["strategy"] = k; proj.append(t)
        y1 = t[t.month <= 12]
        summ.append({"Strategy": k, "Name": s["name"], "How": s["how"], "CAC_AED": s["cac"], "Contribution_per_order_AED": s["cm"],
                     "Orders_per_buyer_per_month": s["opm"], "Premium_share": s["premium"], "Breakeven_month": be,
                     "Year1_new_customers": y1.new_buyers.sum(), "Year1_orders": y1.orders.sum(),
                     "Active_buyers_m36": t.active_buyers.iloc[-1], "Lowest_cash_AED": t.cash.min(), "Cash_m36_AED": t.cash.iloc[-1],
                     "Extra_funding_needed_AED": max(0.0, -t.cash.min()),
                     "LTV_AED": s["cm"] * s["opm"] / 0.05, "LTV_to_CAC": s["cm"] * s["opm"] / 0.05 / s["cac"]})
    proj = pd.concat(proj, ignore_index=True); summ = pd.DataFrame(summ)
    # sensitivity of the recommended strategy
    sens = []
    s3 = out["S3"]
    for lab, kw in [("Base case", {}), ("Churn 4%/month", {"churn": .04}), ("Churn 7%/month", {"churn": .07}),
                    ("CAC +25%", {"cac": s3["cac"] * 1.25}), ("CAC -25%", {"cac": s3["cac"] * .75}),
                    ("Orders per buyer -20%", {"opm": s3["opm"] * .8}), ("No buyer-protection fee", {"cm": float(sum(segs[s] * seg_cm(s, False) for s in segs.index))})]:
        a = dict(cac=s3["cac"], opm=s3["opm"], cm=s3["cm"]); churn = kw.pop("churn", .05); a.update(kw)
        t, be = project(a["cac"], a["opm"], a["cm"], months=60, churn=churn)
        sens.append({"Case": lab, "Breakeven_month": be if be else "beyond 60", "Lowest_cash_AED": t.cash.min()})
    sens = pd.DataFrame(sens)
    return dict(scores=d, target=tm, price_seg=ps, price_demo=pdemo, crosssell=cs, projection=proj, summary=summ,
                sensitivity=sens, base_rate=base, strategies=out)


ACTION_COLORS = {"Target now": "#1c5cab", "Nurture (retarget)": "#86b6ef", "Later (after trust is proven)": "#eda100", "Do not target": "#c3c2b7"}
SEG_ORDER = ["Young Value Seekers", "Premium Brand Hunters", "Busy Declutterers", "Cautious Mainstream", "Resistant Traditionalists"]


def fig_target_matrix(st):
    t = st["target"]
    fig = go.Figure()
    for a, col in ACTION_COLORS.items():
        s = t[t.Action == a]
        fig.add_bar(y=s.Segment, x=s.Respondents, name=a, orientation="h", marker_color=col,
                    customdata=np.c_[s.Conversion_Tier, s.Avg_Probability, s.Avg_Predicted_WTP],
                    text=[f"{tr}: {n:,}" for tr, n in zip(s.Conversion_Tier, s.Respondents)], textposition="inside", insidetextanchor="middle",
                    hovertemplate="%{y}<br>%{customdata[0]} tier: %{x:,} people<br>avg probability %{customdata[1]:.0%}<br>predicted WTP AED %{customdata[2]:.0f}<extra>" + a + "</extra>")
    fig.update_layout(barmode="stack"); fig.update_yaxes(categoryorder="array", categoryarray=SEG_ORDER[::-1])
    fig.update_traces(marker_line_color=SURFACE, marker_line_width=1, textfont=dict(color="white", size=10))
    fig.update_xaxes(title="Respondents")
    tn = t[t.Action == "Target now"]
    return style(fig, "Who to target: segment (clustering) × conversion tier (classification)", h=380, legend=True), dict(
        what=f"{int(tn.Respondents.sum()):,} respondents ({tn.Respondents.sum() / t.Respondents.sum():.0%}) are 'Target now': High-probability people outside the Resistant segment. "
             f"Their actual intent is {(tn.Actual_Intent * tn.Respondents).sum() / tn.Respondents.sum():.0%} vs {st['base_rate']:.0%} for everyone. "
             f"{int(t[t.Action == 'Do not target'].Respondents.sum()):,} ({t[t.Action == 'Do not target'].Respondents.sum() / t.Respondents.sum():.0%}) should not be targeted.",
        why="Classification says how likely each person is to buy; clustering says who they are and how to reach them. Together they turn a "
            "'spray and pray' campaign into a list of named groups with a clear action each.",
        so="Spend acquisition money only on 'Target now'; retarget 'Nurture' with trust content; leave Cautious Mainstream until reviews and guarantees exist; never pay to reach Resistant Traditionalists.")


def fig_price_points(st):
    p = st["price_seg"].set_index("Segment").reindex(SEG_ORDER).reset_index()
    fig = go.Figure()
    fig.add_bar(x=p.Segment, y=p.Median_Predicted_WTP, marker_color=[ACTION_COLORS["Do not target"] if s == "Resistant Traditionalists" else BLUE for s in p.Segment],
                error_y=dict(type="data", symmetric=False, array=p.P75 - p.Median_Predicted_WTP, arrayminus=p.Median_Predicted_WTP - p.P25, color=MUTED, thickness=1.2),
                text=[f"AED {v:.0f}" for v in p.Median_Predicted_WTP], textposition="inside", insidetextanchor="start", textfont=dict(color="white", size=13),
                hovertemplate="%{x}<br>median AED %{y:.0f}<extra></extra>")
    fig.update_yaxes(title="Predicted price for an AED 300 item (AED)", range=[0, 260])
    top, low = p.iloc[p.Median_Predicted_WTP.idxmax()], p[p.Segment != "Resistant Traditionalists"].sort_values("Median_Predicted_WTP").iloc[0]
    return style(fig, "Price points by segment (Lasso regression predictions; bars = middle 50%)", h=380), dict(
        what=f"{top.Segment} accept about AED {top.Median_Predicted_WTP:.0f} for an AED 300 item ({top.Price_as_pct_of_new:.0%} of new); "
             f"{low.Segment} only about AED {low.Median_Predicted_WTP:.0f} ({low.Price_as_pct_of_new:.0%}). Delivery tolerance is AED {p.Delivery_WTP.min():.0f}–{p.Delivery_WTP.max():.0f} across segments.",
        why="One price list cannot fit every customer. Regression turns each person's answers into a predicted price, so pricing can be set per segment instead of guessed.",
        so="Guide sellers to list at about 30–40% of retail for value segments and about 55% for premium items; show premium stock to high-WTP buyers; do not plan revenue around a paid membership (most accept only AED 5–14 a month).")


def fig_projection(st):
    t = st["projection"]; s = st["summary"].set_index("Strategy")
    fig = go.Figure()
    for i, k in enumerate(["S1", "S2", "S3"]):
        x = t[t.strategy == k]
        fig.add_scatter(x=x.month, y=x.cash / 1000, mode="lines", name=s.loc[k, "Name"], line=dict(width=3 if k == "S3" else 2, color=[RED, MUTED, BLUE][i]),
                        hovertemplate="Month %{x}<br>Cash AED %{y:,.0f}k<extra>" + s.loc[k, "Name"] + "</extra>")
    fig.add_hline(y=0, line_color=INK2, line_width=1)
    be = s.loc["S3", "Breakeven_month"]
    if be:
        fig.add_vline(x=be, line_dash="dash", line_color=BLUE, annotation_text=f"Recommended strategy: monthly break-even in month {int(be)}", annotation_position="bottom right")
    fig.update_xaxes(title="Month after launch", dtick=6); fig.update_yaxes(title="Cash balance (AED thousand)")
    r = s.loc["S3"]
    return style(fig, "Cash over 36 months under three strategies (start: AED 105k left after setup)", h=440, legend=True), dict(
        what=f"The recommended strategy reaches monthly break-even in month {int(r.Breakeven_month)} and cash bottoms at AED {r.Lowest_cash_AED / 1000:,.0f}k, so about "
             f"AED {r.Extra_funding_needed_AED / 1000:,.0f}k of extra funding is needed. Model-targeting alone ({s.loc['S2', 'Name']}) "
             f"{'never breaks even in 36 months' if pd.isna(s.loc['S2', 'Breakeven_month']) else 'breaks even in month ' + str(int(s.loc['S2', 'Breakeven_month']))}; "
             f"the unfocused launch never does and burns AED {-s.loc['S1', 'Cash_m36_AED'] / 1000:,.0f}k by month 36.",
        why="This is the 'burn your hands' test: a business can grow customers every month and still run out of cash if each order earns too little.",
        so=f"Launch only with all three levers: targeting, premium-led mix and a buyer-protection fee. Reserve about AED {r.Extra_funding_needed_AED / 1000:,.0f}k on top of the "
           f"AED 220k plan (more if acquisition costs run high; see sensitivity), and stop if month-6 metrics miss the gates.")


def column_guide(df):
    Q = {"Age_Group": "Q1", "Emirate": "Q2", "Occupation": "Q3", "Monthly_Fashion_Spend_Band": "Q4", "Monthly_Fashion_Spend_AED_Midpoint": "Q4",
         "Shopping_Frequency": "Q5", "Previous_Secondhand_Purchase": "Q6", "Preferred_Shopping_Channel": "Q7", "Motivating_Discount": "Q8",
         "Unused_Clothing_Behaviour": "Q10", "Interest_in_Selling": "Q15", "Maximum_Item_Spend_Band": "Q16", "Maximum_Item_Spend_AED_Midpoint": "Q16",
         "Subscription_WTP_Band": "Q17", "Subscription_WTP_AED_Midpoint": "Q17", "Delivery_WTP_Band": "Q18", "Delivery_WTP_AED_Midpoint": "Q18",
         "AED300_Item_WTP_Band": "Q19", "AED300_Item_WTP_AED_Midpoint": "Q19", "Q20_Use_ReWear_5Level": "Q20", "Conversion_Intent_Binary": "Q20"}
    for p, q in [("Challenge_", "Q9"), ("Barrier_", "Q11"), ("Category_", "Q12"), ("Feature_", "Q13"), ("Trust_", "Q14")]:
        for c in multi_cols(df, p): Q[c] = q
    desc = {
        "Respondent_ID": ("Unique respondent key", "None (identifier)", "Excluded from every model so IDs cannot leak into predictions."),
        "Age_Group": ("Age band", "Classification, Clustering (ordinal), Regression", "Younger groups expected to show higher intent; age shapes segments."),
        "Emirate": ("Emirate of residence", "Classification, Regression, Profiling", "Mainly for launch geography and logistics; weak demand effect expected."),
        "Occupation": ("Occupation", "Classification, Regression, Profiling", "Students vs employees vs homemakers differ in budget and supply."),
        "Monthly_Fashion_Spend_Band": ("Monthly fashion spend band (answer)", "Classification, Clustering (ordinal)", "Budget level; separates premium from value shoppers."),
        "Monthly_Fashion_Spend_AED_Midpoint": ("AED midpoint of the spend band (capped)", "Regression (as log spend), EDA", "Strongest driver of how much people will pay."),
        "Shopping_Frequency": ("How often they shop for fashion", "Classification, Clustering, Break-even projection", "Sets orders per buyer per month in the projection."),
        "Previous_Secondhand_Purchase": ("Bought second-hand before?", "Classification, Clustering, Regression", "Expected strongest adoption predictor (experience ladder)."),
        "Preferred_Shopping_Channel": ("Preferred channel", "Classification, Clustering (digital flag)", "Where to acquire customers; digital shoppers convert more."),
        "Motivating_Discount": ("Discount needed vs new price", "Classification, Clustering, Regression", "Price sensitivity; 'would not buy' identifies the resistant group."),
        "Unused_Clothing_Behaviour": ("What happens to unused clothes", "Classification, Supply analysis", "'Keep in wardrobe' = untapped seller supply."),
        "Interest_in_Selling": ("Would sell on ReWear?", "Classification, Clustering (ordinal), Supply", "Seller-side conversion; identifies the supply segment."),
        "Q20_Use_ReWear_5Level": ("Would use ReWear (5 levels)", "Profiling only", "Kept for EDA; never an input to any model."),
        "Conversion_Intent_Binary": ("TARGET: Definitely/Probably yes = 1", "Classification target", "What the classifiers predict; also profiles clusters after they are formed."),
        "AED300_Item_WTP_AED_Midpoint": ("TARGET: price for an AED 300 item", "Regression target, Clustering input", "What Linear/Ridge/Lasso predict; basis for price points and AOV."),
        "Synthetic_Data_Flag": ("Marks every row as simulated", "None", "Honesty flag; excluded."), "Outlier_Flag": ("1 = extreme value was capped", "Data quality", "Lets results be re-run without outliers."),
    }
    rows = []
    for c in df.columns:
        if c in desc:
            m, u, e = desc[c]
        elif c.startswith("Challenge_"):
            m, u, e = (f"Q9 pain selected: {nice(c)} (1/0)", "Association rules, Classification, some Clustering", "Pains that should co-occur with matching feature requests.")
        elif c.startswith("Barrier_"):
            m, u, e = (f"Q11 selling barrier: {nice(c)} (1/0)", "Association rules, Classification, some Clustering", "Supply friction -> which seller service to build.")
        elif c.startswith("Category_"):
            m, u, e = (f"Q12 category wanted: {nice(c)} (1/0)", "Association rules (cross-sell), Classification, Regression", "Category bundles to cross-sell; luxury raises WTP.")
        elif c.startswith("Feature_"):
            m, u, e = (f"Q13 feature wanted: {nice(c)} (1/0)", "Association rules, Classification, some Clustering", "MVP scope and features to offer as add-ons.")
        elif c.startswith("Trust_"):
            m, u, e = (f"Q14 trust driver: {nice(c)} (1/0)", "Association rules, Classification", "Trust package shown on listings.")
        elif c.endswith("_Band"):
            m, u, e = ("Answer band for a pricing question", "EDA, Classification", "Human-readable answer; its midpoint column is the number.")
        elif c.endswith("_AED_Midpoint"):
            m, u, e = ("AED midpoint of a pricing answer", "Classification input, EDA, Financial model", "Q16-Q18 inform AOV, membership and delivery pricing.")
        elif c.endswith("_Ord"):
            m, u, e = ("Ordered number version of an answer (0 = lowest)", "Correlation, Clustering", "Lets ordered answers be correlated and clustered.")
        elif c.endswith("_Raw"):
            m, u, e = ("Value before outlier capping", "Data quality only", "Kept for sensitivity checks; not used in models.")
        elif c.endswith("_Count"):
            m, u, e = ("Number of options ticked in a multi-select block", "EDA, Correlation", "Engagement with the problem/solution space.")
        elif c == "Segment":
            m, u, e = ("Customer segment found by K-Means", "Output of Clustering", "Persona used for targeting, pricing and cross-sell.")
        else:
            m, u, e = ({"Discount_Needed_Pct": "Discount needed as a number", "Rejects_Preowned": "1 = would not buy pre-owned", "Log_Fashion_Spend": "log of fashion spend",
                        "Digital_Channel": "1 = prefers a digital channel", "Seller_Prospect": "1 = probably/definitely would sell",
                        "Has_Idle_Supply": "1 = keeps or discards unused clothes", "WTP_Pct_of_New_Price": "Q19 as % of the AED 300 price",
                        "Premium_Interest": "1 = wants designer or bags"}.get(c, "Engineered feature"),
                       "Clustering / Regression / EDA (engineered)", "Business-meaningful driver created in cleaning.")
        rows.append({"Column": c, "Question": Q.get(c, "Derived"), "What it means": m, "Used by": u, "What we expect / why": e})
    return pd.DataFrame(rows)
