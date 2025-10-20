# Churn Prediction & Retention Playbook (Online Retail)

**Goal**  
Predict next-30-day churn and design **LTV × Risk** interventions that maximize **net profit**.

**Highlights**  
- Time-based split: Observation **90d**, Prediction **30d**; leakage guarded.  
- Features: **RFM + behavior + seasonality + country + returns + wholesaler heuristic**.  
- Models: Calibrated Logistic / Random Forest (`class_weight='balanced'`), **recall-first thresholding**.  
- Evaluation: **PR-AUC, F1, Recall**, Probability Histogram, **Lift Chart**.  
- Actions: **2×2 LTV × Risk** segments with **costed ROI**; A/B funnel with **Wilson CI**.

**Live Demos**  
- Plotly HTML (open locally): see `outputs/figs/`  
- Kaggle Notebook (one-click run): <YOUR_KAGGLE_LINK>  
- Open in Colab: https://colab.research.google.com/drive/1oLuJUnUtsHdYQle4Ax4-zX7KavDLr-qu?usp=sharing

## Data

- Kaggle — *Online Retail Dataset* (Ulrik Thyge Pedersen). Please respect source terms; do **not** re-upload the raw CSV in this repo.  [oai_citation:1‡Kaggle](https://www.kaggle.com/datasets/ulrikthygepedersen/online-retail-dataset/data?utm_source=chatgpt.com)

**Schema (fields)**  
`InvoiceNo` (6-digit; **starts with ‘C’ = cancellation**), `StockCode`, `Description`, `Quantity`, `InvoiceDate`, `UnitPrice`, `CustomerID`, `Country`.  [oai_citation:2‡Kaggle](https://www.kaggle.com/datasets/tunguz/online-retail?utm_source=chatgpt.com)

**Preprocessing (this project):**
- Remove cancellations (`InvoiceNo` startswith **'C'**).  
- `Sales = Quantity × UnitPrice` (GBP).  
- Define sliding windows: **Observation 90d**, **Prediction 30d**; **churn= no purchase in prediction window**.
- Wholesale heuristic: **AOV ≥ P90** or **avg_lines_per_invoice ≥ 10**.

**Repro (download via Kaggle API):**
```bash
pip install kaggle
kaggle datasets download -d ulrikthygepedersen/online-retail-dataset -p data/raw -f OnlineRetail.csv --unzip
