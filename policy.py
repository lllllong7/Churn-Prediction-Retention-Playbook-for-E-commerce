
import pandas as pd
import numpy as np

def create_segment(row, ltv_threshold, churn_risk_threshold):
    is_high_value = row["LTV_Proxy"] >= ltv_threshold
    is_high_risk  = row["p_churn"]   >= churn_risk_threshold
    if is_high_value and is_high_risk:
        return "A. High-Value, High-Risk"
    elif is_high_value and not is_high_risk:
        return "B. High-Value, Low-Risk"
    elif not is_high_value and is_high_risk:
        return "C. Low-Value, High-Risk"
    else:
        return "D. Low-Value, Low-Risk"

def calculate_profit(row):
    is_high_value = "High-Value" in row["Segment"]
    base_profit = 500 if is_high_value else 100
    cost_map = {"A":50, "B":5, "C":5, "D":1}
    segment_prefix = row["Segment"].split(".")[0]
    expected_retained_users = row["Count"] * row["Estimated_Uplift_Rate"]
    net_uplift = (expected_retained_users * base_profit) - (row["Count"] * cost_map[segment_prefix])
    return net_uplift

def segment_and_summarize(X: pd.DataFrame, y_test, p_test, ltv_col="Monetary_LTV_Proxy", risk_thr=0.6, ltv_quantile=0.75):
    seg = X.loc[y_test.index].copy()
    seg["p_churn"]   = p_test
    seg["LTV_Proxy"] = seg.get(ltv_col, 0.0)
    LTV_THRESHOLD = seg["LTV_Proxy"].quantile(ltv_quantile)
    CHURN_RISK_THRESHOLD = risk_thr
    seg["Segment"] = seg.apply(lambda r: create_segment(r, LTV_THRESHOLD, CHURN_RISK_THRESHOLD), axis=1)

    segment_summary = seg.groupby("Segment").agg(
        Count=("p_churn","size"), Avg_LTV=("LTV_Proxy","mean"), Avg_Churn_Prob=("p_churn","mean")
    ).reset_index()

    policy_text = {
        'A. High-Value, High-Risk': 'High-cost personalized retention (exclusive coupons, priority service)',
        'B. High-Value, Low-Risk': 'Maintaining satisfaction (membership upgrades, double points)',
        'C. Low-Value, High-Risk': 'Low-cost automated recall (emails/SMS, small-value coupons)',
        'D. Low-Value, Low-Risk': 'Light touch or no intervention (No Treatment)',
    }
    policy_uplift = {
        'A. High-Value, High-Risk': 0.15,
        'B. High-Value, Low-Risk': 0.02,
        'C. Low-Value, High-Risk': 0.05,
        'D. Low-Value, Low-Risk': 0.005,
    }

    segment_summary["Policy"] = segment_summary["Segment"].map(policy_text)
    segment_summary["Estimated_Uplift_Rate"] = segment_summary["Segment"].map(policy_uplift)

    segment_summary["Estimated_Net_Profit_Uplift"] = segment_summary.apply(calculate_profit, axis=1)
    return seg.reset_index(), segment_summary
