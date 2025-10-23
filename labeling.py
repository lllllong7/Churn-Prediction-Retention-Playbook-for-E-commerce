
import pandas as pd
from datetime import datetime, timedelta

def make_slice_and_label(df_transactions: pd.DataFrame, df_returns: pd.DataFrame, t_ref: pd.Timestamp | None = None, obs_days: int = 90, pred_days: int = 30):
    if t_ref is None:
        t_ref = pd.to_datetime(df_transactions["InvoiceDate"]).max().normalize()
    obs_start = t_ref - pd.Timedelta(days=obs_days)
    pred_end  = t_ref + pd.Timedelta(days=pred_days)
    df_obs  = df_transactions[(df_transactions["InvoiceDate"] > obs_start) & (df_transactions["InvoiceDate"] <= t_ref)]
    df_pred = df_transactions[(df_transactions["InvoiceDate"] > t_ref) & (df_transactions["InvoiceDate"] <= pred_end)]
    customers = df_obs["CustomerID"].dropna().unique()
    X_base = pd.DataFrame({"CustomerID": customers}).set_index("CustomerID")
    future_buyers = set(df_pred["CustomerID"].dropna().unique())
    y = X_base.index.to_series().map(lambda cid: 0 if cid in future_buyers else 1); y.name = "Churn"
    return df_obs, df_pred, X_base, y, t_ref

def compute_ltv_proxy(df_transactions: pd.DataFrame, t_ref, days: int = 180) -> pd.DataFrame:
    start = t_ref - pd.Timedelta(days=days)
    df_ltv = df_transactions[(df_transactions["InvoiceDate"] > start) & (df_transactions["InvoiceDate"] <= t_ref)]
    ltv = df_ltv.groupby("CustomerID")["Sales"].sum().reset_index().rename(columns={"Sales": "Monetary_LTV_Proxy"})
    return ltv
