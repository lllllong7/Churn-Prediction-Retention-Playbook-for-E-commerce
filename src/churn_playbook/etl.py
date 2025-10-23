import kagglehub
import pandas as pd
import numpy as np

def load_and_clean() -> pd.DataFrame:
    from kagglehub import KaggleDatasetAdapter
    # Load the latest version
    df = kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        "ulrikthygepedersen/online-retail-dataset",
        "online_retail.csv",
        pandas_kwargs={"low_memory": False, "parse_dates": ["InvoiceDate"]}
    )

    if "InvoiceDate" in df.columns:
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    if "CustomerID" in df.columns:
        df = df.dropna(subset=["CustomerID"])
    for c in ["Quantity","UnitPrice"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def split_transactions_and_returns(df: pd.DataFrame):
    df = df.copy()
    df["IsCanceled"] = df["InvoiceNo"].astype(str).str.startswith("C")
    return df[~df["IsCanceled"]].copy(), df[df["IsCanceled"]].copy()

def add_sales(df_transactions: pd.DataFrame) -> pd.DataFrame:
    df_transactions = df_transactions.copy()
    df_transactions["Sales"] = df_transactions["Quantity"] * df_transactions["UnitPrice"]
    return df_transactions

def detect_wholesalers(df_transactions: pd.DataFrame, aov_quantile: float = 0.90, line_items_threshold: int = 10):
    aov = df_transactions.groupby("CustomerID")["Sales"].sum() / df_transactions.groupby("CustomerID")["InvoiceNo"].nunique()
    aov_threshold = aov.quantile(aov_quantile)
    lines_per_customer = df_transactions.groupby("CustomerID")["InvoiceNo"].nunique()
    is_wholesaler = (aov >= aov_threshold) | (lines_per_customer >= line_items_threshold)
    wholesaler_ids = is_wholesaler[is_wholesaler].index
    return set(wholesaler_ids), float(aov_threshold), int(line_items_threshold)

def add_wholesaler_flag(df_transactions: pd.DataFrame, wholesaler_ids: set) -> pd.DataFrame:
    df_transactions = df_transactions.copy()
    df_transactions["IsWholesaler"] = df_transactions["CustomerID"].apply(lambda x: x in wholesaler_ids)
    return df_transactions
