import numpy as np
import pandas as pd

def simulate_ab_from_segments(
    segment_summary: pd.DataFrame,
    email_per_user: int = 100,
    seed: int = 42,
    strategies: list[tuple[str, str, float]] | None = None,
    stage_rates: dict | None = None,
    revenue_share_of_ltv: float = 0.6,
) -> pd.DataFrame:
    """
    Simulate aggregated A/B outcomes from a segment summary table.

    Parameters
    ----------
    segment_summary : pd.DataFrame
        Must contain ['Segment', 'Count', 'Avg_LTV'].
    email_per_user : int
        Number of deliveries (emails/messages) per user.
    seed : int
        Random seed (kept for interface consistency; current simulation is deterministic).
    strategies : list[tuple[str, str, float]] | None
        List of tuples: (segment_label, strategy_name, per_user_cost_gbp).
        Defaults to:
            ("A. High-Value, High-Risk", "VIP50",       50.0),
            ("C. Low-Value, High-Risk", "LightVoucher",  5.0)
    stage_rates : dict | None
        Conversion rates for treatment/control with chained denominators:
        opened = delivered * open_rate
        clicked = opened * click_rate
        purchased = clicked * buy_rate
        Defaults:
            treatment: open=0.45, click=0.16, buy=0.07
            control:   open=0.43, click=0.12, buy=0.048
    revenue_share_of_ltv : float
        Revenue per purchase is `revenue_share_of_ltv * Avg_LTV`.

    Returns
    -------
    pd.DataFrame
        Columns:
        ['strategy','group','delivered','opened','clicked','purchased','revenue_gbp','offer_cost_gbp']
        where group is 'treatment' or 'control' (lowercase).
    """
    np.random.seed(seed)

    required = {"Segment", "Count", "Avg_LTV"}
    if not required.issubset(segment_summary.columns):
        raise ValueError(f"segment_summary must contain {required}, got {segment_summary.columns.tolist()}")

    if strategies is None:
        strategies = [
            ("A. High-Value, High-Risk", "VIP50",       50.0),
            ("C. Low-Value, High-Risk", "LightVoucher",  5.0),
        ]

    if stage_rates is None:
        stage_rates = {
            "treatment": {"open": 0.45, "click": 0.16, "buy": 0.07},
            "control":   {"open": 0.43, "click": 0.12, "buy": 0.048},
        }

    rows: list[dict] = []
    for seg_label, strat_name, per_user_cost in strategies:
        row = segment_summary.loc[segment_summary["Segment"] == seg_label]
        if row.empty:
            continue

        n_users = int(row["Count"].values[0])
        delivered_t = int(n_users * email_per_user)
        delivered_c = int(n_users * email_per_user)

        rt = stage_rates["treatment"]
        rc = stage_rates["control"]

        opened_t    = int(delivered_t * rt["open"])
        clicked_t   = int(opened_t   * rt["click"])
        purchased_t = int(clicked_t  * rt["buy"])

        opened_c    = int(delivered_c * rc["open"])
        clicked_c   = int(opened_c   * rc["click"])
        purchased_c = int(clicked_c  * rc["buy"])

        avg_ltv = float(row["Avg_LTV"].values[0])
        rev_t = purchased_t * (revenue_share_of_ltv * avg_ltv)
        rev_c = purchased_c * (revenue_share_of_ltv * avg_ltv)

        offer_cost_t = float(n_users * per_user_cost)
        offer_cost_c = 0.0

        rows += [
            dict(strategy=strat_name, group="treatment",
                 delivered=delivered_t, opened=opened_t, clicked=clicked_t, purchased=purchased_t,
                 revenue_gbp=rev_t, offer_cost_gbp=offer_cost_t),
            dict(strategy=strat_name, group="control",
                 delivered=delivered_c, opened=opened_c, clicked=clicked_c, purchased=purchased_c,
                 revenue_gbp=rev_c, offer_cost_gbp=offer_cost_c),
        ]

    cols = ["strategy","group","delivered","opened","clicked","purchased","revenue_gbp","offer_cost_gbp"]
    return pd.DataFrame(rows, columns=cols)