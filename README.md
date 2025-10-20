# Churn Prediction & Retention Playbook (Online Retail)
[![Open in Colab](https://img.shields.io/badge/Colab-Open%20Notebook-orange?logo=googlecolab)]([<YOUR_COLAB_LINK>](https://colab.research.google.com/drive/1oLuJUnUtsHdYQle4Ax4-zX7KavDLr-qu?usp=sharing))
[![Kaggle](https://img.shields.io/badge/Kaggle-View%20Notebook-blue?logo=kaggle)](<YOUR_KAGGLE_LINK>)

## Goal
Predict 30-day churn and design **LTV × Risk** interventions that maximize **net profit**.

## Highlights
- **Time-based split**: Observation **90d**, Prediction **30d**; label leakage guarded  
- Features: **RFM + behavior + seasonality + country + returns + wholesaler heuristic**  
- Models: **Calibrated Logistic / Random Forest** (`class_weight='balanced'`), **recall-first thresholding**  
- Evaluation: **Recall, Precision, F1, ROC-AUC**, Probability Histogram, **Lift Chart**  
- Actions: **2×2 LTV × Risk** segments with **costed ROI**; A/B funnel with **Wilson 95% CI**

## Results
- **Recall**: **75.89%** · **Precision**: **68.01%** · **F1**: **71.73%**  
- **ROC-AUC**: **0.704**  · PR-AUC: **TODO** (fill with `average_precision_score`)  
- **Segment uplift (estimated, GBP)**  
  - **A (High-Value × High-Risk)** — Count **11**, Avg LTV **£2,528.52**, **Net uplift £275.00**  
  - **B (High-Value × Low-Risk)** — Count **173**, Avg LTV **£4,817.27**, **Net uplift £865.00**  
  - **C (Low-Value × High-Risk)** — Count **313**, Avg LTV **£380.50**, **Net uplift £0.00**  
  - **D (Low-Value × Low-Risk)** — Count **239**, Avg LTV **£783.40**, **Net uplift −£119.50**  
  - **Total projected net uplift**: **£1,020.50** (under current cost & margin assumptions)
- **A/B (Experiment)**  
  - Delivered→Purchased **95% Wilson CIs** computed per strategy（see Experiment visuals）  
  - **TODO**: Summarize numeric deltas (e.g., “VIP50 vs Control +K pp, CI [lo, hi]”, net profit差异)

## Live Demos
- Plotly HTML (open locally):  
  `outputs/figs/fig_cohort_retention.html` · `fig_prob_hist.html` · `fig_lift.html` · `fig_ltv_risk_scatter.html` · `experiment_dashboard.html`  
- Kaggle Notebook (one-click run): <YOUR_KAGGLE_LINK>  
- Open in Colab: <YOUR_COLAB_LINK>

## Data
- **Original source**: UCI Machine Learning Repository — *Online Retail* (coverage **2010-12-01 → 2011-12-09**, GBP)  
  https://archive.ics.uci.edu/ml/datasets/online+retail  
- **Mirror used**: Kaggle — *Online Retail Dataset* (Ulrik Thyge Pedersen)  
  https://www.kaggle.com/datasets/ulrikthygepedersen/online-retail-dataset/data

**Schema**  
`InvoiceNo` (6-digit; **starts with ‘C’ = cancellation**), `StockCode`, `Description`, `Quantity`, `InvoiceDate`, `UnitPrice`, `CustomerID`, `Country`.

**Preprocessing (this project)**  
- Remove cancellations (`InvoiceNo` startswith **'C'**)  
- `Sales = Quantity × UnitPrice` (GBP)  
- Windows: **Obs 90d**, **Pred 30d**; **churn = no purchase in Pred window**  
- **Wholesaler** heuristic: `AOV ≥ P90` **or** `avg_lines_per_invoice ≥ 10`

**Download via Kaggle API**
```bash
pip install kaggle
kaggle datasets download -d ulrikthygepedersen/online-retail-dataset -p data/raw -f OnlineRetail.csv --unzip

Note: Please do not commit the raw CSV to the repo. Provide commands/scripts to fetch instead.
Accessed: 2025-10-19 (UTC-7).
```
Note: Please do not commit the raw CSV to the repo. Provide commands/scripts to fetch instead.
Accessed: 2025-10-19 (UTC-7)

## Repro
```bash
# 1) Env
python -V                     # 3.10+ recommended
pip install -r requirements.txt

# 2) Run end-to-end (CSV path & output dir)
python scripts/run_all.py --csv data/raw/OnlineRetail.csv --out outputs/
```
requirements.txt
```bash
pandas
numpy
scikit-learn
plotly
```
## Business Actions
- A (High-Value × High-Risk) — High-touch offers (VIP coupon, concierge) with margin guardrails
  - Rationale: high churn risk + high LTV; current sim shows +£275 uplift on small base（11 users）
- B (High-Value × Low-Risk) — Value preservation（membership perks, double points, surprise-&-delight）
  - Rationale: largest net uplift £865 due to volume; avoid over-incentivizing
- C (Low-Value × High-Risk) — Low-cost automation（email/SMS nudges, small voucher）
  - Rationale: risk high but LTV low; current settings yield £0 uplift → test smaller incentives / creative
- D (Low-Value × Low-Risk) — Light touch / holdout
  - Rationale: −£119.50 under assumptions; avoid spend unless creative improves unit economics
- Guardrails: cool-down windows, one-time codes, min margin threshold; weekly ROI review

## Methods
- Labeling: time-based split; Obs 90d to build features, Pred 30d to mark churn (no purchase)
- Features: RFM; activity (7/14/30d); seasonality; country; returns rate; wholesaler flag
- Models: Logistic / RandomForest (class_weight='balanced'); probability calibration（可选）
- Thresholding: tune for target Recall; report Recall / Precision / F1 / ROC-AUC（plus PR-AUC）
- Evaluation: Probability histogram, Lift (deciles), ROC/PR curves; segmentation metrics & costed ROI
- Experiment: A/B funnel（Delivered→Opened→Clicked→Purchased）, Wilson 95% CI for conversion, net profit = margin − offer cost

## Limitations & Next
- Dataset span & seasonality; wholesale mix may bias behavior
-	Window choices (90/30d) affect label density; consider rolling cuts & multi-slice training
-	Next: BG/NBD / Pareto-NBD for LTV, uplift modeling, cost-aware optimization（budget constraints & targeting）

## Repo Map
```bash
churn-retention-playbook/
├─ README.md
├─ LICENSE
├─ requirements.txt
├─ scripts/
│  └─ run_all.py                 # one-click: data→features→train→plots
├─ src/
│  ├─ etl.py                     # cleaning; cancellations; Sales
│  ├─ features.py                # RFM/behavior/seasonality/country/returns/wholesale
│  ├─ labeling.py                # 90d+30d windows; churn rule
│  ├─ modeling.py                # LR/RF; calibration; thresholding
│  ├─ policy.py                  # LTV×Risk segmentation; ROI
│  └─ plotly_helpers.py          # Plotly figures (overview/segments/experiment)
├─ notebooks/
│  ├─ 01_EDA.ipynb
│  ├─ 02_Features_and_Labeling.ipynb
│  ├─ 03_Modeling_and_Evaluation.ipynb
│  └─ 04_Segments_and_Experiment.ipynb
└─ outputs/
   ├─ figs/
   │  ├─ fig_cohort_retention.html
   │  ├─ fig_prob_hist.html
   │  ├─ fig_lift.html
   │  ├─ fig_ltv_risk_scatter.html
   │  └─ experiment_dashboard.html
   ├─ model_metrics.json
   └─ latest_scores_and_segments.csv
```
## License & Citation
-	License: MIT (see LICENSE)
-	Cite:
  -	Kaggle mirror — Online Retail Dataset (Ulrik Thyge Pedersen)
