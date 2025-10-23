
import pandas as pd
import numpy as np
from datetime import timedelta

def build_features(df_transactions: pd.DataFrame, df_returns: pd.DataFrame, df_obs: pd.DataFrame, t_ref, X_base: pd.DataFrame, use_country_top_k: int = 5) -> pd.DataFrame:
    X = X_base.reset_index()
    last_purchase = df_obs.groupby("CustomerID")["InvoiceDate"].max().reset_index()
    last_purchase["Recency"] = (t_ref - last_purchase["InvoiceDate"]).dt.days
    X = X.merge(last_purchase[["CustomerID","Recency"]], on="CustomerID", how="left")

    rfm_agg = df_obs.groupby("CustomerID").agg(Frequency=("InvoiceNo","nunique"), Monetary=("Sales","sum")).reset_index()
    X = X.merge(rfm_agg, on="CustomerID", how="left")

    activity_agg = df_obs.groupby("CustomerID").agg(AvgOrderValue=("Sales","mean"), UniqueItems=("StockCode","nunique"), TotalQuantity=("Quantity","sum")).reset_index()
    X = X.merge(activity_agg, on="CustomerID", how="left")

    return_counts = df_returns[(df_returns["InvoiceDate"] > (t_ref - pd.Timedelta(days=90))) & (df_returns["InvoiceDate"] <= t_ref)].groupby("CustomerID")["IsCanceled"].count()
    total_orders_obs = df_transactions[(df_transactions["InvoiceDate"] > (t_ref - pd.Timedelta(days=90))) & (df_transactions["InvoiceDate"] <= t_ref)].groupby("CustomerID")["InvoiceNo"].nunique()
    ret_rate = (return_counts / total_orders_obs).fillna(0).rename("ReturnRate").reset_index()
    X = X.merge(ret_rate, on="CustomerID", how="left")

    top_countries = df_transactions["Country"].value_counts().head(use_country_top_k).index.tolist()
    country_map = df_transactions[df_transactions["Country"].isin(top_countries)].groupby("CustomerID")["Country"].agg(lambda s: s.mode().iat[0]).reset_index()
    country_dummies = pd.get_dummies(country_map, columns=["Country"], prefix="Country")
    X = X.merge(country_dummies, on="CustomerID", how="left")

    if "IsWholesaler" in df_transactions.columns:
        is_wholesaler_df = df_transactions[["CustomerID","IsWholesaler"]].drop_duplicates("CustomerID")
        X = X.merge(is_wholesaler_df, on="CustomerID", how="left")

    X["Recency"] = X["Recency"].fillna(91)
    for col in ["Frequency","Monetary","AvgOrderValue","UniqueItems","TotalQuantity","ReturnRate"]:
        if col in X.columns: X[col] = X[col].fillna(0)
    if "IsWholesaler" in X.columns:
        X["IsWholesaler"] = X["IsWholesaler"].fillna(False)

    return X.set_index("CustomerID")
