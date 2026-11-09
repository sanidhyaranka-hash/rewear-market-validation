# ReWear UAE — Pre-Owned Fashion Marketplace — Market Validation

Data Analytics (MGB) individual submission: synthetic survey → cleaning → EDA → 4 algorithms → Streamlit dashboard.

## Files
| File | What it is |
|---|---|
| `data/raw_survey_responses.csv` | Raw synthetic survey export: 5,045 rows, with missing values, outliers, messy labels, straight-liners and duplicates |
| `data/cleaned_survey_data.csv` | Cleaned and transformed dataset: 4,980 rows × 90 columns, 0 missing cells in analysis columns |
| `data/cleaning_log.csv` | Every cleaning step and how many rows or cells it changed |
| `data/data_dictionary.csv` | Each column: question, meaning, the algorithm(s) it feeds and what we expect |
| `data/hidden_truth_segments.csv` | True customer type per respondent, used only to validate clustering |
| `generate_data.py` | Creates the raw synthetic data (seeded, so it can be reproduced exactly) |
| `clean_data.py` | Cleaning and transformation pipeline; writes the cleaned data and the log |
| `analysis.py` | Charts with auto-written insights, classification, regression, K-means clustering, Apriori association rules, unit economics and the 36-month break-even strategy |
| `app.py` | Streamlit dashboard: 9 tabs, 40+ charts, each with a written insight |

## Run it
```bash
pip install -r requirements.txt
python generate_data.py      # optional – data already included
python clean_data.py         # optional – cleaned data already included
streamlit run app.py
```
The first load trains the models, which takes about 30–60 seconds.

## Dashboard tabs
1. Overview & verdict
2. Data quality: raw vs clean, cleaning log, column guide
3. Descriptive EDA (Q20, D1–D11, Q19)
4. Diagnostic & correlation (G1–G6 plus explore-any-question)
5. Classification: who will use ReWear (7 models)
6. Regression: what they will pay (Linear, Ridge, Lasso, Random Forest)
7. Clustering: five customer segments
8. Association rules: trust, luxury and seller-convenience bundles, with adjustable thresholds and a cross-sell playbook
9. Feasibility & break-even: market data, an editable unit-economics model, target matrix, price points and a 36-month projection

The data is synthetic and was built for coursework. It is not evidence of real demand.
