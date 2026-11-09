"""
ReWear UAE - synthetic survey data generator (v2)
=================================================
Simulates 5,000 responses to the 20-question ReWear questionnaire.

Why the data looks the way it does
----------------------------------
Real survey answers are not independent coin flips: a 22-year-old student on a
small budget answers differently from a 38-year-old who buys designer bags.
To make the dataset useful for EDA and for the four algorithms, every
respondent is first assigned to one of five HIDDEN customer types (latent
personas). Their answers are then drawn from probabilities that depend on that
type plus individual-level drivers, followed by random noise. The persona label
itself is NOT written to the survey file - a real survey would never contain
it - so clustering has to rediscover the structure.

Algorithm hooks built into the simulation
-----------------------------------------
* Classification  - Q20 intent is driven by prior resale experience, discount
                    sensitivity, channel, trust needs and persona, plus noise.
* Clustering      - the five latent personas create real (but overlapping)
                    groups in demographics, spend, categories and features.
* Association     - linked multi-select items (e.g. counterfeit concern ->
                    authenticity check -> authenticity certificate).
* Regression      - Q19 WTP depends on fashion spend, persona, designer
                    interest, discount sensitivity and resale experience.

Data-quality problems injected on purpose (for the cleaning step)
-----------------------------------------------------------------
missing values, inconsistent labels (" dubai", "DXB", "18 - 24"), mixed
Yes/No text in 0/1 columns, duplicate submissions, straight-lining "speeders",
impossible values (negative spend, 99999 sentinel) and genuine extreme outliers.

Run:  python generate_data.py            -> data/raw_survey_responses.csv
All records are SIMULATED. They are not real UAE consumer evidence.
"""
import numpy as np
import pandas as pd

SEED = 20261109
N = 5000
rng = np.random.default_rng(SEED)

# ----------------------------------------------------------------------------
# 1. Latent personas (hidden)
# ----------------------------------------------------------------------------
PERSONAS = ["Value_Seeker", "Premium_Hunter", "Busy_Declutterer",
            "Cautious_Mainstream", "Resistant"]
P_SHARE = [0.26, 0.17, 0.20, 0.24, 0.13]
persona = rng.choice(len(PERSONAS), size=N, p=P_SHARE)


def pick(options, probs_by_persona):
    """Draw one categorical answer per respondent using persona-specific probabilities."""
    out = np.empty(N, dtype=object)
    for k in range(len(PERSONAS)):
        idx = np.where(persona == k)[0]
        p = np.array(probs_by_persona[k], dtype=float)
        out[idx] = rng.choice(options, size=len(idx), p=p / p.sum())
    return out


# ----------------------------------------------------------------------------
# 2. Q1-Q8 demographics and shopping behaviour
# ----------------------------------------------------------------------------
AGE = ["18-24", "25-30", "31-40", "41-50", "Above 50"]
age = pick(AGE, [[45, 33, 14, 6, 2], [10, 32, 38, 15, 5], [6, 20, 40, 25, 9],
                 [18, 24, 28, 19, 11], [6, 12, 25, 30, 27]])

EMIRATE = ["Dubai", "Abu Dhabi", "Sharjah", "Ajman", "Ras Al Khaimah", "Fujairah", "Umm Al Quwain"]
emirate = pick(EMIRATE, [[38, 18, 26, 9, 5, 2, 2], [55, 30, 9, 3, 2, 0.5, 0.5],
                         [40, 25, 21, 7, 4, 2, 1], [36, 26, 22, 8, 5, 2, 1],
                         [30, 28, 22, 8, 7, 3, 2]])

OCC = ["University student", "Private employee", "Government employee",
       "Self-employed", "Homemaker", "Other"]
occ = pick(OCC, [[42, 35, 7, 6, 4, 6], [4, 50, 18, 22, 4, 2], [3, 40, 15, 8, 30, 4],
                 [12, 45, 22, 8, 7, 6], [3, 38, 32, 8, 10, 9]])

SPEND_BANDS = ["Below 100", "100-250", "251-500", "501-1000", "1001-2000", "Above 2000"]
SPEND_MID = {"Below 100": 75, "100-250": 175, "251-500": 375, "501-1000": 750,
             "1001-2000": 1500, "Above 2000": 2500}
spend_band = pick(SPEND_BANDS, [[22, 38, 27, 10, 2, 1], [1, 4, 15, 32, 30, 18],
                                [4, 15, 35, 30, 12, 4], [8, 24, 36, 23, 7, 2],
                                [10, 22, 33, 23, 9, 3]])

FREQ = ["Several times a month", "Monthly", "Every 2-3 months", "Every 6 months", "Less often"]
freq = pick(FREQ, [[18, 34, 30, 12, 6], [30, 38, 22, 7, 3], [8, 25, 37, 20, 10],
                   [9, 27, 34, 20, 10], [4, 14, 30, 27, 25]])

PREV = ["Frequently", "Occasionally", "Once or twice", "Never"]
prev = pick(PREV, [[22, 38, 25, 15], [18, 34, 28, 20], [10, 30, 32, 28],
                   [3, 15, 30, 52], [1, 5, 16, 78]])

CHANNEL = ["E-commerce marketplaces", "Social media sellers", "Resale platforms",
           "Brand websites", "Mall/physical stores"]
channel = pick(CHANNEL, [[30, 32, 22, 6, 10], [18, 10, 20, 34, 18], [40, 15, 15, 12, 18],
                         [28, 12, 6, 18, 36], [14, 6, 2, 16, 62]])

DISC = ["Below 20%", "20-30%", "31-40%", "41-50%", "Above 50%", "Would not buy pre-owned"]
disc = pick(DISC, [[2, 10, 28, 32, 27, 1], [22, 34, 26, 12, 5, 1], [8, 25, 34, 22, 9, 2],
                   [3, 14, 30, 28, 17, 8], [1, 4, 10, 15, 15, 55]])

# ----------------------------------------------------------------------------
# 3. Multi-select blocks (Q9, Q11-Q14) with built-in item links
# ----------------------------------------------------------------------------
def bern(p):
    return (rng.random(N) < np.clip(p, 0.01, 0.97)).astype(int)


def by_persona(vals):
    return np.array(vals, dtype=float)[persona]


# Q9 challenges
ch = {}
ch["Challenge_High_Branded_Prices"] = bern(by_persona([.62, .45, .30, .35, .25]))
ch["Challenge_Affordability"] = bern(by_persona([.66, .10, .25, .30, .25]) + .15 * (ch["Challenge_High_Branded_Prices"]))
ch["Challenge_Limited_Variety"] = bern(by_persona([.30, .40, .25, .22, .15]))
ch["Challenge_Sizing"] = bern(by_persona([.28, .25, .35, .30, .20]))
ch["Challenge_Quality_Concerns"] = bern(by_persona([.20, .35, .25, .52, .40]))
ch["Challenge_Counterfeit_Concerns"] = bern(by_persona([.12, .62, .15, .35, .25]))
ch["Challenge_Online_Seller_Trust"] = bern(by_persona([.18, .35, .25, .55, .45]) + .12 * ch["Challenge_Counterfeit_Concerns"])
ch["Challenge_Return_Exchange"] = bern(by_persona([.22, .25, .30, .45, .30]) + .10 * ch["Challenge_Quality_Concerns"])
any_ch = sum(ch.values()) > 0
ch["Challenge_No_Significant_Challenge"] = ((~any_ch) | (rng.random(N) < .01)).astype(int)
for k in list(ch)[:-1]:   # 'no challenge' should not co-exist with a challenge
    ch[k] = np.where(ch["Challenge_No_Significant_Challenge"] == 1, 0, ch[k])

# Q10 unused clothing behaviour
UNUSED = ["Keep in wardrobe", "Donate", "Give to friends/family", "Sell through social media",
          "Sell through resale platform", "Discard", "Other"]
unused = pick(UNUSED, [[20, 15, 20, 25, 14, 3, 3], [22, 12, 10, 22, 28, 2, 4],
                       [38, 28, 14, 8, 6, 3, 3], [36, 26, 20, 6, 3, 4, 5],
                       [34, 32, 20, 3, 1, 6, 4]])

# Q11 barriers to selling
ba = {}
ba["Barrier_No_Time"] = bern(by_persona([.20, .35, .62, .25, .20]))
ba["Barrier_Complicated_Process"] = bern(by_persona([.22, .20, .40, .30, .25]) + .22 * ba["Barrier_No_Time"])
ba["Barrier_Dont_Know_Where"] = bern(by_persona([.22, .10, .35, .40, .20]))
ba["Barrier_Dont_Know_Price"] = bern(by_persona([.30, .18, .35, .35, .15]) + .15 * ba["Barrier_Dont_Know_Where"])
ba["Barrier_Payment_Security"] = bern(by_persona([.15, .25, .18, .45, .30]) + .10 * ch["Challenge_Online_Seller_Trust"])
ba["Barrier_Selling_To_Strangers"] = bern(by_persona([.12, .20, .22, .42, .35]) + .15 * ba["Barrier_Payment_Security"])
ba["Barrier_Low_Resale_Value"] = bern(by_persona([.35, .20, .25, .22, .30]))
ba["Barrier_Not_Interested"] = bern(by_persona([.08, .12, .06, .22, .62]))

# Q12 categories
ca = {}
ca["Category_Designer_Luxury"] = bern(by_persona([.12, .78, .18, .15, .06]))
ca["Category_Bags_Accessories"] = bern(by_persona([.25, .55, .25, .25, .12]) + .18 * ca["Category_Designer_Luxury"])
ca["Category_Shoes_Sneakers"] = bern(by_persona([.62, .45, .25, .30, .15]))
ca["Category_Sportswear"] = bern(by_persona([.48, .18, .25, .25, .12]) + .12 * ca["Category_Shoes_Sneakers"])
ca["Category_Branded_Casual"] = bern(by_persona([.60, .35, .35, .40, .20]))
ca["Category_Formalwear"] = bern(by_persona([.15, .40, .28, .30, .15]))
ca["Category_Childrenswear"] = bern(by_persona([.04, .10, .70, .20, .10]))
ca["Category_Traditional_Ethnic"] = bern(by_persona([.15, .25, .35, .30, .20]))
ca["Category_Menswear"] = bern(by_persona([.35, .30, .25, .30, .25]))

# Q13 features (linked to challenges and barriers)
fe = {}
fe["Feature_Authenticity_Check"] = bern(by_persona([.15, .62, .15, .25, .12]) + .35 * ch["Challenge_Counterfeit_Concerns"] + .10 * ca["Category_Designer_Luxury"])
fe["Feature_Verified_Sellers"] = bern(by_persona([.20, .40, .25, .40, .20]) + .32 * ch["Challenge_Online_Seller_Trust"])
fe["Feature_Seller_Ratings"] = bern(by_persona([.30, .30, .20, .30, .15]) + .20 * fe["Feature_Verified_Sellers"])
fe["Feature_Quality_Grading"] = bern(by_persona([.20, .40, .25, .35, .15]) + .30 * ch["Challenge_Quality_Concerns"])
fe["Feature_Secure_Payments"] = bern(by_persona([.20, .35, .25, .40, .20]) + .32 * ba["Barrier_Payment_Security"])
fe["Feature_Returns_Refunds"] = bern(by_persona([.20, .25, .25, .35, .15]) + .35 * ch["Challenge_Return_Exchange"])
fe["Feature_Home_Delivery"] = bern(by_persona([.30, .35, .55, .35, .20]) + .20 * ba["Barrier_No_Time"])
fe["Feature_AI_Price_Suggestions"] = bern(by_persona([.20, .15, .30, .15, .05]) + .35 * ba["Barrier_Dont_Know_Price"])
fe["Feature_Wardrobe_Valuation"] = bern(by_persona([.15, .25, .45, .12, .05]) + .15 * ba["Barrier_No_Time"])
fe["Feature_AI_Recommendations"] = bern(by_persona([.42, .25, .15, .15, .08]))
fe["Feature_Clothing_Exchange"] = bern(by_persona([.45, .08, .25, .12, .08]) + .10 * ch["Challenge_Affordability"])

# Q14 trust drivers (linked to features)
tr = {}
tr["Trust_Authenticity_Certificate"] = bern(by_persona([.08, .45, .10, .15, .08]) + .38 * fe["Feature_Authenticity_Check"])
tr["Trust_Verified_Identity"] = bern(by_persona([.12, .25, .18, .30, .15]) + .30 * fe["Feature_Verified_Sellers"])
tr["Trust_Reviews"] = bern(by_persona([.30, .20, .25, .25, .12]) + .28 * fe["Feature_Seller_Ratings"])
tr["Trust_Condition_Ratings"] = bern(by_persona([.15, .25, .20, .25, .10]) + .35 * fe["Feature_Quality_Grading"])
tr["Trust_Detailed_Photos"] = bern(by_persona([.35, .30, .30, .35, .20]) + .12 * ch["Challenge_Quality_Concerns"])
tr["Trust_Secure_Gateway"] = bern(by_persona([.12, .20, .15, .25, .12]) + .38 * fe["Feature_Secure_Payments"])
tr["Trust_Return_Guarantee"] = bern(by_persona([.12, .15, .18, .25, .10]) + .38 * fe["Feature_Returns_Refunds"])
tr["Trust_Customer_Support"] = bern(by_persona([.15, .25, .25, .30, .15]))

# Q15 interest in selling
SELL = ["Definitely", "Probably", "Not sure", "Probably not", "Definitely not"]
sell = pick(SELL, [[22, 34, 22, 14, 8], [24, 32, 20, 14, 10], [32, 38, 16, 9, 5],
                   [8, 24, 32, 22, 14], [3, 8, 19, 30, 40]])

# ----------------------------------------------------------------------------
# 4. Willingness to pay (Q16-Q19) - numeric latent value, then banded
# ----------------------------------------------------------------------------
spend_mid = np.array([SPEND_MID[b] for b in spend_band], dtype=float)
log_spend = np.log(spend_mid)
prev_score = np.select([prev == "Frequently", prev == "Occasionally", prev == "Once or twice"], [3, 2, 1], 0)
disc_need = pd.Series(disc).map({"Below 20%": 15, "20-30%": 25, "31-40%": 35, "41-50%": 45,
                                 "Above 50%": 55, "Would not buy pre-owned": 70}).to_numpy()

# Q19: what would you pay for an item that cost AED 300 new, in excellent condition?
q19 = (8 + 16.0 * log_spend + by_persona([-8, 30, 4, -4, -14])
       + 14 * ca["Category_Designer_Luxury"] + 5 * prev_score - 1.15 * (disc_need - 35)
       + 6 * fe["Feature_Authenticity_Check"] + rng.normal(0, 24, N))
q19 = np.clip(q19, 20, 280)
would_not_buy = (disc == "Would not buy pre-owned") & (rng.random(N) < .70) | (rng.random(N) < .015)
q19 = np.where(would_not_buy, 0, q19)

Q19_BANDS = [(0, "Would not buy", 0), (75, "Below 75", 60), (100, "75-100", 87.5), (125, "101-125", 113),
             (150, "126-150", 138), (200, "151-200", 175.5), (9e9, "Above 200", 225)]
def band_q19(v):
    if v == 0:
        return "Would not buy", 0.0
    for upper, lab, mid in Q19_BANDS[1:]:
        if v < upper:
            return lab, float(mid)
q19_band, q19_mid = zip(*[band_q19(v) for v in q19])

# Q16 maximum spend on one pre-owned branded item
q16 = np.exp(np.log(spend_mid) * .55 + by_persona([1.9, 2.9, 2.2, 2.1, 1.7]) + .25 * ca["Category_Designer_Luxury"] + rng.normal(0, .35, N))
Q16 = [(50, "Below 50", 35), (100, "50-100", 75), (200, "101-200", 150), (350, "201-350", 275),
       (500, "351-500", 425), (9e9, "Above 500", 650)]
q16_band, q16_mid = zip(*[next((l, float(m)) for u, l, m in Q16 if v < u) for v in q16])

# Q17 membership fee per month
q17 = np.clip(by_persona([4, 14, 9, 3, 0]) + 1.6 * np.log(spend_mid / 375) + rng.normal(0, 5, N), 0, 60)
Q17 = [(1, "Free only", 0), (10.5, "5-10", 7.5), (20.5, "11-20", 15.5), (30.5, "21-30", 25.5),
       (50.5, "31-50", 40.5), (9e9, "Above 50", 60)]
q17_band, q17_mid = zip(*[next((l, float(m)) for u, l, m in Q17 if v < u) for v in q17])

# Q18 delivery fee per order
q18 = np.clip(by_persona([2, 9, 8, 4, 1]) + 1.2 * np.log(spend_mid / 375) + 2 * fe["Feature_Home_Delivery"] + rng.normal(0, 3.5, N), 0, 30)
Q18 = [(.5, "Free only", 0), (5.5, "1-5", 3), (10.5, "6-10", 8), (15.5, "11-15", 13),
       (20.5, "16-20", 18), (9e9, "Above 20", 25)]
q18_band, q18_mid = zip(*[next((l, float(m)) for u, l, m in Q18 if v < u) for v in q18])

# ----------------------------------------------------------------------------
# 5. Q20 conversion intent (target) - logistic model + noise
# ----------------------------------------------------------------------------
age_young = np.isin(age, ["18-24", "25-30"]).astype(int)
digital = np.isin(channel, ["E-commerce marketplaces", "Social media sellers", "Resale platforms"]).astype(int)
resale_channel = (channel == "Resale platforms").astype(int)
trust_feature_count = (fe["Feature_Verified_Sellers"] + fe["Feature_Secure_Payments"]
                       + fe["Feature_Authenticity_Check"] + fe["Feature_Returns_Refunds"])
sell_score = pd.Series(sell).map({"Definitely": 2, "Probably": 1, "Not sure": 0,
                                  "Probably not": -1, "Definitely not": -2}).to_numpy()
logit = (-0.95 + by_persona([0.55, 0.45, 0.25, -0.35, -1.6])
         + 0.42 * prev_score + 0.35 * digital + 0.45 * resale_channel + 0.22 * age_young
         + 0.10 * trust_feature_count + 0.18 * sell_score
         - 1.30 * (disc == "Would not buy pre-owned")
         - 0.30 * ba["Barrier_Not_Interested"]
         + 0.25 * ch["Challenge_Affordability"]
         + rng.logistic(0, 0.55, N))        # individual noise
p_intent = 1 / (1 + np.exp(-logit))
# Map probability to the 5-level answer (people near the threshold say "Not sure")
u = p_intent + rng.normal(0, .06, N)
q20 = np.select([u > .78, u > .52, u > .44, u > .22], ["Definitely yes", "Probably yes", "Not sure", "Probably not"], "Definitely not")

# ----------------------------------------------------------------------------
# 6. Assemble the clean "truth" table in the questionnaire order
# ----------------------------------------------------------------------------
df = pd.DataFrame({
    "Respondent_ID": [f"RW{i:05d}" for i in range(1, N + 1)],
    "Age_Group": age, "Emirate": emirate, "Occupation": occ,
    "Monthly_Fashion_Spend_Band": spend_band, "Monthly_Fashion_Spend_AED_Midpoint": spend_mid,
    "Shopping_Frequency": freq, "Previous_Secondhand_Purchase": prev,
    "Preferred_Shopping_Channel": channel, "Motivating_Discount": disc,
})
for block in (ch,):
    for k, v in block.items():
        df[k] = v
df["Unused_Clothing_Behaviour"] = unused
for block in (ba, ca, fe, tr):
    for k, v in block.items():
        df[k] = v
df["Interest_in_Selling"] = sell
df["Maximum_Item_Spend_Band"] = q16_band
df["Maximum_Item_Spend_AED_Midpoint"] = q16_mid
df["Subscription_WTP_Band"] = q17_band
df["Subscription_WTP_AED_Midpoint"] = q17_mid
df["Delivery_WTP_Band"] = q18_band
df["Delivery_WTP_AED_Midpoint"] = q18_mid
df["AED300_Item_WTP_Band"] = q19_band
df["AED300_Item_WTP_AED_Midpoint"] = q19_mid
df["Q20_Use_ReWear_5Level"] = q20
df["Conversion_Intent_Binary"] = np.isin(q20, ["Definitely yes", "Probably yes"]).astype(int)

# Persona is kept only in a separate validation file (never in the survey data)
pd.DataFrame({"Respondent_ID": df.Respondent_ID, "Hidden_Persona": np.array(PERSONAS)[persona]}) \
  .to_csv("data/hidden_truth_segments.csv", index=False)

# ----------------------------------------------------------------------------
# 7. Inject realistic data-quality problems
# ----------------------------------------------------------------------------
raw = df.copy()
for c in raw.columns:
    if raw[c].dtype.kind in "iuf":
        raw[c] = raw[c].astype(object)

# 7a. Missing values (item non-response)
miss_plan = {"Occupation": .020, "Preferred_Shopping_Channel": .015, "Motivating_Discount": .012,
             "Interest_in_Selling": .025, "Monthly_Fashion_Spend_Band": .010,
             "AED300_Item_WTP_Band": .018, "Shopping_Frequency": .008}
for col, rate in miss_plan.items():
    idx = rng.choice(N, int(N * rate), replace=False)
    raw.loc[idx, col] = np.nan
    if col == "Monthly_Fashion_Spend_Band":
        raw.loc[idx, "Monthly_Fashion_Spend_AED_Midpoint"] = np.nan
    if col == "AED300_Item_WTP_Band":
        raw.loc[idx, "AED300_Item_WTP_AED_Midpoint"] = np.nan

# 7b. Inconsistent labels (typing / export problems)
def scramble(col, mapping, rate):
    idx = rng.choice(N, int(N * rate), replace=False)
    for i in idx:
        v = raw.at[i, col]
        if isinstance(v, str) and v in mapping:
            raw.at[i, col] = rng.choice(mapping[v])
scramble("Emirate", {"Dubai": [" dubai", "DXB", "DUBAI", "Dubai "], "Abu Dhabi": ["abu dhabi", "AUH", "Abu-Dhabi"],
                     "Sharjah": ["sharjah", "SHJ"], "Ajman": ["ajman "], "Ras Al Khaimah": ["RAK", "Ras al Khaimah"]}, .06)
scramble("Age_Group", {"18-24": ["18 - 24", "18–24"], "25-30": ["25 - 30", "25–30"], "31-40": ["31 - 40"],
                       "41-50": ["41 - 50"], "Above 50": ["50+", "above 50"]}, .04)
scramble("Shopping_Frequency", {"Monthly": ["monthly", "MONTHLY"], "Every 2-3 months": ["every 2-3 months"]}, .03)
scramble("Previous_Secondhand_Purchase", {"Never": ["never", "No, never"], "Frequently": ["frequently"]}, .02)

# 7c. Mixed Yes/No text in binary columns
for col in ["Challenge_Sizing", "Feature_Home_Delivery", "Trust_Reviews", "Category_Sportswear"]:
    idx = rng.choice(N, int(N * .025), replace=False)
    raw.loc[idx, col] = raw.loc[idx, col].map({1: "Yes", 0: "No"})

# 7d. Impossible values and genuine extreme spenders
neg = rng.choice(N, 6, replace=False)
raw.loc[neg, "Monthly_Fashion_Spend_AED_Midpoint"] = -375.0                 # sign entry error
sentinel = rng.choice(N, 8, replace=False)
raw.loc[sentinel, "AED300_Item_WTP_AED_Midpoint"] = 99999.0                  # survey-tool "no answer" code
extreme = rng.choice(N, 25, replace=False)
raw.loc[extreme, "Monthly_Fashion_Spend_AED_Midpoint"] = rng.uniform(6000, 25000, 25).round(0)  # real but extreme
ext2 = rng.choice(N, 15, replace=False)
raw.loc[ext2, "Maximum_Item_Spend_AED_Midpoint"] = rng.uniform(1500, 6000, 15).round(0)

# 7e. Straight-liners / speeders: ticked every option in every multi-select
multi = [c for c in raw.columns if c.startswith(("Challenge_", "Barrier_", "Category_", "Feature_", "Trust_"))]
speed = rng.choice(N, 20, replace=False)
raw.loc[speed, multi] = 1

# 7f. Duplicate submissions (same person submitted twice)
dups = raw.sample(45, random_state=SEED)
raw = pd.concat([raw, dups], ignore_index=True)
raw = raw.sample(frac=1, random_state=SEED).reset_index(drop=True)
raw["Synthetic_Data_Flag"] = "SIMULATED"

raw.to_csv("data/raw_survey_responses.csv", index=False)
print("Raw rows:", len(raw), "| columns:", raw.shape[1])
print("Positive intent (truth):", round(df.Conversion_Intent_Binary.mean(), 3))
