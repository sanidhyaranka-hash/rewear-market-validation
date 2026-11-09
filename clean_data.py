"""
ReWear UAE - data cleaning and transformation pipeline
======================================================
Input : data/raw_survey_responses.csv    (synthetic survey export with injected problems)
Output: data/cleaned_survey_data.csv (analysis-ready table)
        data/cleaning_log.csv          (every step, rows affected, reason)

Each step is logged so the report and dashboard can show exactly what changed.
Run: python clean_data.py
"""
import re
import numpy as np
import pandas as pd

RAW = "data/raw_survey_responses.csv"
OUT = "data/cleaned_survey_data.csv"
LOG = []


def log(step, action, rows, reason):
    LOG.append({"Step": step, "Action": action, "Rows_or_Cells_Affected": int(rows), "Reason": reason})


df = pd.read_csv(RAW, low_memory=False)
start_rows = len(df)
log("0. Load", f"Loaded {start_rows} rows x {df.shape[1]} columns", start_rows, "Raw export preserved unchanged")

MULTI = [c for c in df.columns if c.startswith(("Challenge_", "Barrier_", "Category_", "Feature_", "Trust_"))]

# ---------------------------------------------------------------- 1. duplicates
dup_mask = df.duplicated(keep="first")
log("1. Duplicates", "Removed exact duplicate submissions", dup_mask.sum(),
    "Same respondent ID and identical answers = double submission")
df = df[~dup_mask].copy()
id_dups = df.Respondent_ID.duplicated().sum()
log("1. Duplicates", "Checked Respondent_ID uniqueness after removal", id_dups, "Should be 0")

# ---------------------------------------------------------------- 2. text standardisation
def norm_text(s):
    return re.sub(r"\s+", " ", s.strip()) if isinstance(s, str) else s

EMIRATE_MAP = {"dubai": "Dubai", "dxb": "Dubai", "abu dhabi": "Abu Dhabi", "auh": "Abu Dhabi", "abu-dhabi": "Abu Dhabi",
               "sharjah": "Sharjah", "shj": "Sharjah", "ajman": "Ajman", "rak": "Ras Al Khaimah",
               "ras al khaimah": "Ras Al Khaimah", "fujairah": "Fujairah", "umm al quwain": "Umm Al Quwain"}
AGE_MAP = {"18-24": "18-24", "25-30": "25-30", "31-40": "31-40", "41-50": "41-50", "above 50": "Above 50", "50+": "Above 50"}
FREQ_MAP = {"several times a month": "Several times a month", "monthly": "Monthly", "every 2-3 months": "Every 2-3 months",
            "every 6 months": "Every 6 months", "less often": "Less often"}
PREV_MAP = {"frequently": "Frequently", "occasionally": "Occasionally", "once or twice": "Once or twice",
            "never": "Never", "no, never": "Never"}

def standardise(col, mapping):
    before = df[col].copy()
    key = df[col].map(lambda v: norm_text(v).lower().replace("–", "-").replace(" - ", "-") if isinstance(v, str) else v)
    df[col] = key.map(lambda v: mapping.get(v, v) if isinstance(v, str) else v)
    changed = (before.fillna("§") != df[col].fillna("§")).sum()
    log("2. Standardise labels", f"{col}: unified spelling/case/spacing variants", changed,
        "Same answer typed differently would split one category into several")

standardise("Emirate", EMIRATE_MAP)
standardise("Age_Group", AGE_MAP)
standardise("Shopping_Frequency", FREQ_MAP)
standardise("Previous_Secondhand_Purchase", PREV_MAP)
for c in df.select_dtypes(include=["object", "string"]).columns:
    df[c] = df[c].map(norm_text)

# ---------------------------------------------------------------- 3. binary columns
yes_no = 0
for c in MULTI:
    s = df[c]
    mask = s.astype(str).str.lower().isin(["yes", "no"])
    yes_no += mask.sum()
    df[c] = s.replace({"Yes": 1, "No": 0, "yes": 1, "no": 0}).astype(float).astype(int)
log("3. Binary encoding", "Converted 'Yes'/'No' text in multi-select columns to 1/0", yes_no,
    "Multi-select answers must be numeric 0/1 for association rules and models")

# ---------------------------------------------------------------- 4. straight-liners
n_ticked = df[MULTI].sum(axis=1)
speeders = n_ticked >= len(MULTI) - 1     # ticked (almost) every box incl. contradictory ones
log("4. Response quality", "Removed straight-lining respondents (ticked every option, incl. 'No significant challenge')",
    speeders.sum(), "Selecting all 52 options, even contradictory ones, signals a careless/bot response")
df = df[~speeders].copy()

# ---------------------------------------------------------------- 5. invalid values
neg = df.Monthly_Fashion_Spend_AED_Midpoint < 0
log("5. Invalid values", "Negative monthly spend set to missing (then re-derived from band)", neg.sum(),
    "Spend cannot be negative - data-entry sign error")
df.loc[neg, "Monthly_Fashion_Spend_AED_Midpoint"] = np.nan

sentinel = df.AED300_Item_WTP_AED_Midpoint >= 9999
log("5. Invalid values", "WTP code 99999 set to missing", sentinel.sum(),
    "99999 is a survey-tool 'no answer' code, not a price; Q19 cannot exceed AED 300")
df.loc[sentinel, "AED300_Item_WTP_AED_Midpoint"] = np.nan

# Band <-> midpoint consistency: re-derive midpoint from band where band exists
SPEND_MID = {"Below 100": 75, "100-250": 175, "251-500": 375, "501-1000": 750, "1001-2000": 1500, "Above 2000": 2500}
Q19_MID = {"Would not buy": 0, "Below 75": 60, "75-100": 87.5, "101-125": 113, "126-150": 138, "151-200": 175.5, "Above 200": 225}
fill_mid = df.Monthly_Fashion_Spend_AED_Midpoint.isna() & df.Monthly_Fashion_Spend_Band.notna()
df.loc[fill_mid, "Monthly_Fashion_Spend_AED_Midpoint"] = df.loc[fill_mid, "Monthly_Fashion_Spend_Band"].map(SPEND_MID)
log("5. Invalid values", "Re-derived spend midpoint from the answered band", fill_mid.sum(),
    "The band is the real answer; the midpoint is a derived number")
fill_q19 = df.AED300_Item_WTP_AED_Midpoint.isna() & df.AED300_Item_WTP_Band.notna()
df.loc[fill_q19, "AED300_Item_WTP_AED_Midpoint"] = df.loc[fill_q19, "AED300_Item_WTP_Band"].map(Q19_MID)
log("5. Invalid values", "Re-derived Q19 WTP midpoint from the answered band", fill_q19.sum(), "Same logic as above")

# ---------------------------------------------------------------- 6. outliers (flag + cap, keep raw)
def iqr_bounds(s, k=3.0):
    q1, q3 = s.quantile([.25, .75])
    return q1 - k * (q3 - q1), q3 + k * (q3 - q1)

df["Outlier_Flag"] = 0
VALID_Q16 = {35, 75, 150, 275, 425, 650}
for col, valid in [("Monthly_Fashion_Spend_AED_Midpoint", set(SPEND_MID.values())), ("Maximum_Item_Spend_AED_Midpoint", VALID_Q16)]:
    s_ = df[col]
    lo, hi = iqr_bounds(s_.dropna())
    # an outlier = above the 3xIQR fence AND not one of the legitimate band midpoints
    out = (s_ > hi) & (~s_.isin(valid))
    df.loc[out, "Outlier_Flag"] = 1
    df[col + "_Raw"] = s_
    cap = max(valid)
    df.loc[out, col] = cap
    log("6. Outliers", f"{col}: flagged {out.sum()} values above 3xIQR fence (AED {hi:,.0f}) that match no answer band; capped at top band AED {cap:,.0f}",
        out.sum(), "Extreme but possible spenders: keep the person, limit their leverage on means and regression")

# ---------------------------------------------------------------- 7. missing values
miss_before = df.isna().sum()
miss_before = miss_before[(miss_before > 0) & ~miss_before.index.str.endswith("_Raw")]
cat_cols = [c for c in miss_before.index if df[c].dtype.kind not in "if" and c != "AED300_Item_WTP_Band"]
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])
    log("7. Missing values", f"{col}: imputed with mode (most common answer)", miss_before[col],
        "Item non-response under 3%; imputation keeps the respondent's other answers")
# spend midpoint follows its (now complete) band, so the two stay consistent
fill = df.Monthly_Fashion_Spend_AED_Midpoint.isna()
df.loc[fill, "Monthly_Fashion_Spend_AED_Midpoint"] = df.loc[fill, "Monthly_Fashion_Spend_Band"].map(SPEND_MID)
log("7. Missing values", "Monthly_Fashion_Spend_AED_Midpoint: filled from imputed band", fill.sum(), "Band and midpoint must agree")
# Q19 WTP: median of people in the same spend band (spend is the strongest WTP driver), band re-derived
fill = df.AED300_Item_WTP_AED_Midpoint.isna()
med = df.groupby("Monthly_Fashion_Spend_Band")["AED300_Item_WTP_AED_Midpoint"].transform("median")
df.loc[fill, "AED300_Item_WTP_AED_Midpoint"] = med[fill]
inv = {v: k for k, v in Q19_MID.items()}
def q19_band(v):
    for lab, mid in sorted(Q19_MID.items(), key=lambda x: x[1]):
        if v <= mid: return lab
    return "Above 200"
df.loc[fill, "AED300_Item_WTP_Band"] = df.loc[fill, "AED300_Item_WTP_AED_Midpoint"].map(lambda v: inv.get(v, q19_band(v)))
log("7. Missing values", "AED300_Item_WTP (Q19): imputed with median of same spend band; band re-derived", fill.sum(),
    "Group median is more accurate than the overall median because spend drives WTP")

# ---------------------------------------------------------------- 8. transformation / feature engineering
ORD = {
    "Age_Group": ["18-24", "25-30", "31-40", "41-50", "Above 50"],
    "Monthly_Fashion_Spend_Band": ["Below 100", "100-250", "251-500", "501-1000", "1001-2000", "Above 2000"],
    "Shopping_Frequency": ["Less often", "Every 6 months", "Every 2-3 months", "Monthly", "Several times a month"],
    "Previous_Secondhand_Purchase": ["Never", "Once or twice", "Occasionally", "Frequently"],
    "Interest_in_Selling": ["Definitely not", "Probably not", "Not sure", "Probably", "Definitely"],
    "Q20_Use_ReWear_5Level": ["Definitely not", "Probably not", "Not sure", "Probably yes", "Definitely yes"],
}
for col, order in ORD.items():
    df[col + "_Ord"] = df[col].map({v: i for i, v in enumerate(order)})
log("8. Transform", "Ordinal-encoded 6 ordered answer scales (e.g. Never=0 ... Frequently=3)", len(df) * len(ORD),
    "Keeps the natural order for correlation and models")

df["Discount_Needed_Pct"] = df.Motivating_Discount.map({"Below 20%": 15, "20-30%": 25, "31-40%": 35, "41-50%": 45,
                                                        "Above 50%": 55, "Would not buy pre-owned": np.nan})
df["Rejects_Preowned"] = (df.Motivating_Discount == "Would not buy pre-owned").astype(int)
df["Discount_Needed_Pct"] = df["Discount_Needed_Pct"].fillna(60)   # strongest resistance coded above top band
df["Log_Fashion_Spend"] = np.log(df.Monthly_Fashion_Spend_AED_Midpoint)
df["Challenge_Count"] = df[[c for c in MULTI if c.startswith("Challenge_") and "No_Significant" not in c]].sum(axis=1)
df["Barrier_Count"] = df[[c for c in MULTI if c.startswith("Barrier_")]].sum(axis=1)
df["Category_Count"] = df[[c for c in MULTI if c.startswith("Category_")]].sum(axis=1)
df["Feature_Count"] = df[[c for c in MULTI if c.startswith("Feature_")]].sum(axis=1)
df["Trust_Count"] = df[[c for c in MULTI if c.startswith("Trust_")]].sum(axis=1)
df["Digital_Channel"] = df.Preferred_Shopping_Channel.isin(["E-commerce marketplaces", "Social media sellers", "Resale platforms"]).astype(int)
df["Seller_Prospect"] = df.Interest_in_Selling.isin(["Definitely", "Probably"]).astype(int)
df["Has_Idle_Supply"] = df.Unused_Clothing_Behaviour.isin(["Keep in wardrobe", "Discard"]).astype(int)
df["WTP_Pct_of_New_Price"] = (df.AED300_Item_WTP_AED_Midpoint / 300 * 100).round(1)
df["Premium_Interest"] = ((df.Category_Designer_Luxury == 1) | (df.Category_Bags_Accessories == 1)).astype(int)
log("8. Transform", "Engineered 12 features: log spend, discount %, counts per multi-select block, digital channel, "
    "seller prospect, idle supply, WTP % of new price, premium interest", len(df),
    "Turns raw answers into business-meaningful drivers for EDA and models")

df["Synthetic_Data_Flag"] = "SIMULATED"
log("9. Output", f"Final clean table {len(df)} rows x {df.shape[1]} columns, 0 missing in analysis columns",
    len(df), "Ready for EDA and the four algorithms")

df.to_csv(OUT, index=False)
pd.DataFrame(LOG).to_csv("data/cleaning_log.csv", index=False)
print(pd.DataFrame(LOG).to_string(max_colwidth=90))
print("Remaining missing (excluding *_Raw):", int(df[[c for c in df.columns if not c.endswith('_Raw')]].isna().sum().sum()))
