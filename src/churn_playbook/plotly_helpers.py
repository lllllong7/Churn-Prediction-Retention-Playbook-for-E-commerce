
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_curve, roc_auc_score

def compute_kpis_from_slice(df_obs, y, X, ltv_col="Monetary_LTV_Proxy"):
    churn = float(y.mean()); retention = 1 - churn
    arpu = float((df_obs.groupby("CustomerID")["Sales"].sum().mean()) if "Sales" in df_obs.columns else 0.0)
    aov  = float((df_obs.groupby("InvoiceNo")["Sales"].sum().mean()) if "Sales" in df_obs.columns else 0.0)
    ltv  = float(X.get(ltv_col, pd.Series([0])).mean())
    return {"Churn": churn, "Retention": retention, "ARPU": arpu, "AOV": aov, "LTV": ltv}

def make_kpi_cards(kpis: dict):
    fig = go.Figure()
    items = list(kpis.items())
    for i,(k,v) in enumerate(items):
        if k in ("Churn","Retention"):
            fig.add_trace(go.Indicator(mode="number", value=v*100, number={"valueformat":".2f","font":{"size":28}},
                                       number_suffix="%", title={"text":k}, domain={"row":0,"column":i}))
        else:
            fig.add_trace(go.Indicator(mode="number", value=v, number={"valueformat":".2f","font":{"size":28}},
                                       title={"text":k}, domain={"row":0,"column":i}))
    fig.update_layout(grid={"rows":1,"columns":max(1,len(items))}, height=220, margin=dict(l=20,r=20,t=30,b=10))
    return fig

def make_cohort_heatmap(df_transactions):
    df = df_transactions.copy()
    df["OrderMonth"] = df["InvoiceDate"].dt.to_period("M").dt.to_timestamp()
    first = df.groupby("CustomerID")["OrderMonth"].min()
    df["Cohort"] = df["CustomerID"].map(first)
    df["Period"] = (df["OrderMonth"].dt.to_period("M").astype(int) - df["Cohort"].dt.to_period("M").astype(int))
    pivot = df.pivot_table(index="Cohort", columns="Period", values="CustomerID", aggfunc="nunique").fillna(0)
    for i in range(pivot.shape[0]):
        pivot.iloc[i,:] = pivot.iloc[i,:] / max(pivot.iloc[i,0], 1)
    fig = px.imshow(pivot, aspect="auto", color_continuous_scale="Blues", labels=dict(color="Retention"))
    fig.update_layout(title="Cohort Retention by First Purchase Month")
    return fig

def make_prob_histogram(p_churn):
    return px.histogram(pd.DataFrame({"p":p_churn}), x="p", nbins=30, title="Churn Probability Histogram")

def make_lift_chart(y_true, y_prob, n_bins: int = 10):
    df = pd.DataFrame({"y_true": y_true, "y_prob": y_prob}).sort_values("y_prob", ascending=False).reset_index(drop=True)
    df["bin"] = pd.qcut(df["y_prob"], q=n_bins, labels=False, duplicates="drop")
    bins = (df.groupby("bin").agg(pos=("y_true","sum"), cnt=("y_true","size")).sort_index(ascending=False))
    baseline = df["y_true"].mean()
    bins["rate"] = bins["pos"] / bins["cnt"]
    bins["lift"] = bins["rate"] / baseline
    bins["cum_pos"] = bins["pos"].cumsum()
    bins["cum_cnt"] = bins["cnt"].cumsum()
    bins["cum_gain"] = bins["cum_pos"] / df["y_true"].sum()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_bar(x=bins.index.astype(str), y=bins["lift"], name="Lift (per bin)")
    fig.add_trace(go.Scatter(x=bins.index.astype(str), y=bins["cum_gain"], mode="lines+markers", name="Cumulative Gain"),
                  secondary_y=True)
    fig.update_layout(title="Lift Chart (by predicted-probability bins)", xaxis_title="Probability Bins (High → Low)")
    fig.update_yaxes(title_text="Lift", secondary_y=False)
    fig.update_yaxes(title_text="Cumulative Gain", secondary_y=True, range=[0,1])
    return fig

def make_pr_curve(y_true, y_score):
    p, r, thr = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    fig = go.Figure(go.Scatter(x=r, y=p, mode="lines", name=f"PR (AP={ap:.3f})"))
    fig.update_layout(title="Precision–Recall Curve", xaxis_title="Recall", yaxis_title="Precision")
    return fig, {"ap": float(ap)}

def make_roc_curve(y_true, y_score):
    fpr, tpr, thr = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    fig = go.Figure(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC={auc:.3f})"))
    fig.update_layout(title="ROC Curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
    return fig, {"auc": float(auc)}

# === Segment ===
def make_ltv_risk_scatter(segment_df: pd.DataFrame, risk_th: float, ltv_th: float):
    df = segment_df.copy()
    if "CustomerID" not in df.columns:
        df = df.reset_index().rename(columns={"index":"CustomerID"})
    fig = px.scatter(
        df, x="LTV_Proxy", y="p_churn", color="Segment", hover_data=["CustomerID"],
        size=(df["LTV_Proxy"] - df["LTV_Proxy"].min() + 1e-6),
        labels={"LTV_Proxy":"LTV Proxy (GBP)", "p_churn":"Churn Probability"},
        title="LTV × Risk Segmentation (with thresholds)"
    )
    fig.add_hline(y=risk_th, line_dash="dash", annotation_text=f"Risk {risk_th:.2f}")
    fig.add_vline(x=ltv_th,  line_dash="dash", annotation_text=f"LTV P75 ({ltv_th:.0f})")
    return fig

def make_segment_summary_plot(segment_summary: pd.DataFrame, margin_rate=0.4, high_cost=50.0, low_cost=5.0):
    summary = segment_summary.copy()
    if "Revenue" not in summary.columns:
        summary["Revenue"] = summary["Count"] * summary["Avg_LTV"]
    if "Estimated_Net_Profit_Uplift" not in summary.columns:
        summary["GrossProfit"] = summary["Revenue"] * margin_rate
        def _cost(seg, n):
            if seg.startswith("A"): return n * high_cost
            if seg.startswith("C"): return n * low_cost
            return 0.0
        summary["Estimated_Net_Profit_Uplift"] = summary.apply(lambda r: r["GrossProfit"] - _cost(r["Segment"], r["Count"]), axis=1)
    fig = make_subplots(rows=1, cols=3, subplot_titles=("Users", "Revenue (GBP)", "Est. Net Profit (GBP)"))
    fig.add_bar(x=summary["Segment"], y=summary["Count"], name="Users", row=1, col=1)
    fig.add_bar(x=summary["Segment"], y=summary["Revenue"], name="Revenue", row=1, col=2)
    fig.add_bar(x=summary["Segment"], y=summary["Estimated_Net_Profit_Uplift"], name="Est. Net Profit", row=1, col=3)
    fig.update_layout(title="Segment Size, Revenue, Estimated Net Profit", showlegend=False)
    return fig

def wilson_ci(successes: int, n: int, z: float = 1.96):
    if n == 0: return (0,0,0)
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2/(2*n)) / denom
    half = (z / denom) * np.sqrt(p*(1-p)/n + z**2/(4*n**2))
    return max(0, center - half), min(1, center + half)

def aggregate_ab(ab_df, by=("group",)):
    """
    Aggregate AB results by keys in `by`.
    Requires columns (case-insensitive):
      strategy, group, delivered, opened, clicked, purchased, revenue_gbp, offer_cost_gbp
    Returns an aggregated DataFrame with delivered-based and stage-based rates.
    """
    import numpy as np
    import pandas as pd

    df = ab_df.copy()
    df.columns = [c.lower() for c in df.columns]
    for col in ["delivered","opened","clicked","purchased","revenue_gbp","offer_cost_gbp"]:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")
    if isinstance(by, str):
        by = (by,)
    by = tuple(k.lower() for k in by)

    # Sum counts and money by group keys
    agg = df.groupby(list(by), as_index=False).sum(numeric_only=True)

    # Delivered-based rates (uniform denominator)
    agg["open_rate_overall"]     = agg["opened"]    / agg["delivered"].clip(lower=1)
    agg["click_rate_overall"]    = agg["clicked"]   / agg["delivered"].clip(lower=1)
    agg["purchase_rate_overall"] = agg["purchased"] / agg["delivered"].clip(lower=1)

    # Stage-based rates (per-stage denominators)
    agg["ctr_stage"]      = agg["clicked"]   / agg["opened"].replace({0: np.nan})
    agg["buy_rate_stage"] = agg["purchased"] / agg["clicked"].replace({0: np.nan})

    # Profit
    agg["net_profit_gbp"] = agg["revenue_gbp"] - agg["offer_cost_gbp"]
    return agg

# === AB plots (now accept `by` and `rate_mode`) ===
def make_ab_funnel(ab_df, by=("group",)):
    """
    Line funnel: delivered -> opened -> clicked -> purchased
    One line per unique key in `by` (e.g., ("strategy","group")).
    """
    import plotly.graph_objects as go
    agg = aggregate_ab(ab_df, by=by)
    stages = ["delivered","opened","clicked","purchased"]
    fig = go.Figure()
    for _, row in agg.iterrows():
        label = " / ".join(str(row[k]) for k in agg.columns if k in by)
        vals = [row[s] for s in stages]
        fig.add_trace(go.Scatter(x=stages, y=vals, mode="lines+markers", name=label))
    fig.update_layout(title="Experiment Funnel", yaxis_title="Count")
    return fig

def make_ab_ci(ab_df, by=("group",), target=("purchased","delivered")):
    """
    Wilson CI bar for rate = target[0] / target[1], aggregated by `by`.
    Example: by=("strategy","group"), target=("purchased","delivered")
    """
    import plotly.graph_objects as go

    num_col, den_col = [c.lower() for c in target]
    agg = aggregate_ab(ab_df, by=by)

    labels, rates, upper_err, lower_err = [], [], [], []
    for _, r in agg.iterrows():
        k = int(r[num_col])
        n = int(r[den_col]) if int(r[den_col]) > 0 else 1
        p_hat = k / n
        lo, hi = wilson_ci(k, n, z=1.96)

        label = " / ".join(str(r[k]) for k in by)
        labels.append(label)
        rates.append(p_hat)
        # Plotly wants upper array (array) and lower array (arrayminus)
        upper_err.append(max(0.0, hi - p_hat))
        lower_err.append(max(0.0, p_hat - lo))

    fig = go.Figure(go.Bar(
        x=labels,
        y=rates,
        error_y=dict(
            type="data",
            array=upper_err,     # upper (positive) errors
            arrayminus=lower_err # lower (negative) errors
        )
    ))
    fig.update_layout(
        title=f"{num_col} / {den_col} (95% Wilson CI)",
        yaxis_tickformat=".1%",
        xaxis_title=" / ".join(by),
        yaxis_title="Rate"
    )
    return fig

def make_ab_profit(ab_df, by=("group",)):
    """
    Net profit bar: revenue - offer_cost, aggregated by `by`.
    """
    import plotly.express as px
    agg = aggregate_ab(ab_df, by=by)
    agg["key"] = agg.apply(lambda r: " / ".join(str(r[k]) for k in agg.columns if k in by), axis=1)
    return px.bar(agg, x="key", y="net_profit_gbp", title="Net Profit by Group")

def make_ab_stage(ab_df, by=("group",), rate_mode="delivered"):
    """
    Stage rate comparison. rate_mode:
      - "delivered": uses open/click/purchase rates with delivered denominator
      - "stage":     uses CTR = clicked/opened and BuyRate = purchased/clicked (plus purchase_overall)
    """
    import pandas as pd, plotly.express as px
    agg = aggregate_ab(ab_df, by=by)
    agg["key"] = agg.apply(lambda r: " / ".join(str(r[k]) for k in agg.columns if k in by), axis=1)

    if rate_mode == "stage":
        plot_df = agg[["key","open_rate_overall","ctr_stage","buy_rate_stage","purchase_rate_overall"]].rename(
            columns={"open_rate_overall":"open_rate(d)",
                     "ctr_stage":"ctr(open→click)",
                     "buy_rate_stage":"buy_rate(click→purchase)",
                     "purchase_rate_overall":"purchase_rate(d)"}).melt(
            id_vars=["key"], var_name="stage", value_name="rate"
        )
    else:
        plot_df = agg[["key","open_rate_overall","click_rate_overall","purchase_rate_overall"]].rename(
            columns={"open_rate_overall":"open_rate(d)",
                     "click_rate_overall":"click_rate(d)",
                     "purchase_rate_overall":"purchase_rate(d)"}).melt(
            id_vars=["key"], var_name="stage", value_name="rate"
        )

    fig = px.bar(plot_df, x="stage", y="rate", color="key", barmode="group",
                 title=f"Stage Rates by {', '.join(by)} [{rate_mode}]")
    fig.update_yaxes(tickformat=".1%")
    return fig

from copy import deepcopy
from plotly.subplots import make_subplots
import plotly.graph_objects as go


# ---------------------------
# 1) Overview dashboard
# ---------------------------
def combine_overview(
    *,
    fig_kpi=None, fig_cohort=None, fig_hist=None, fig_lift=None, fig_pr=None, fig_roc=None,
    title="Overview Dashboard", height=1500
):
    kpi_cols = max(1, len(fig_kpi.data) if (fig_kpi and hasattr(fig_kpi, "data")) else 2)
    left_cols  = max(1, kpi_cols // 2)
    right_cols = max(1, kpi_cols - left_cols)

    specs = []
    # Row 1: KPI indicators
    specs.append([{"type": "domain"} for _ in range(kpi_cols)])
    # Row 2: Cohort heatmap (full width)
    specs.append([{"type": "xy", "colspan": kpi_cols}] + [None] * (kpi_cols - 1))
    # Row 3: Histogram (left) + Lift (right, with secondary y)
    specs.append(
        [{"type": "xy", "colspan": left_cols}] + [None] * (left_cols - 1) +
        [{"type": "xy", "secondary_y": True, "colspan": right_cols}] + [None] * (right_cols - 1)
    )
    # Row 4: PR (left) + ROC (right)
    specs.append(
        [{"type": "xy", "colspan": left_cols}] + [None] * (left_cols - 1) +
        [{"type": "xy", "colspan": right_cols}] + [None] * (right_cols - 1)
    )

    titles = (
        ["KPIs"] + [""] * (kpi_cols - 1)
        + ["Cohort Retention"]
        + ["Probability Histogram", "Lift Chart"]
        + ["PR Curve", "ROC Curve"]
    )

    layout = make_subplots(
        rows=4, cols=kpi_cols, specs=specs,
        horizontal_spacing=0.06, vertical_spacing=0.08,
        subplot_titles=tuple(titles)
    )

    if fig_kpi:
        for i, tr in enumerate(fig_kpi.data):
            layout.add_trace(tr, row=1, col=min(i + 1, kpi_cols))
    if fig_cohort:
        for tr in fig_cohort.data:
            layout.add_trace(tr, row=2, col=1)
    if fig_hist:
        for tr in fig_hist.data:
            layout.add_trace(tr, row=3, col=1)
    if fig_lift:
        for tr in fig_lift.data:
            layout.add_trace(
                tr, row=3, col=left_cols + 1,
                secondary_y=(getattr(tr, "type", "") in ("scatter", "scattergl"))
            )
    if fig_pr:
        for tr in fig_pr.data:
            layout.add_trace(tr, row=4, col=1)
    if fig_roc:
        for tr in fig_roc.data:
            layout.add_trace(tr, row=4, col=left_cols + 1)

    layout.update_layout(title=title, height=height, showlegend=False)
    return layout


# ---------------------------
# 2) Experiment dashboard
# ---------------------------
def combine_experiment(
    *,
    fig_funnel=None, fig_ci=None, fig_profit=None, fig_stage=None,
    title="Experiment Dashboard", height=1200
):
    layout = make_subplots(
        rows=3, cols=2,
        specs=[
            [{"type": "xy", "colspan": 2}, None],
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "xy", "colspan": 2}, None],
        ],
        subplot_titles=(
            "Funnel by Strategy",
            "Delivered→Purchased (95% CI)",
            "Net Profit (T vs C)",
            "Stage Rates",
        ),
        vertical_spacing=0.12, horizontal_spacing=0.08,
    )

    if fig_funnel:
        for tr in fig_funnel.data:
            layout.add_trace(tr, row=1, col=1)
    if fig_ci:
        for tr in fig_ci.data:
            layout.add_trace(tr, row=2, col=1)
    if fig_profit:
        for tr in fig_profit.data:
            layout.add_trace(tr, row=2, col=2)
    if fig_stage:
        for tr in fig_stage.data:
            layout.add_trace(tr, row=3, col=1)

    layout.update_layout(height=height, title_text=title)
    return layout


# ---------------------------
# 3) Segments dashboard
# ---------------------------
def combine_segments(
    fig_scatter,
    fig_summary,
    *,
    title="Segments Dashboard",
    height=900,
    copy_left_shapes=True,
    copy_left_annotations=True,
):
    """
    Top row: scatter (full width, copy threshold lines/annotations)
    Bottom row: reuse bars from `fig_summary` (expected 3 bar traces: Users/Revenue/Est. Net Profit)
    """
    # Titles
    lt = fig_scatter.layout.title.text if fig_scatter and fig_scatter.layout.title else "LTV × Risk Segmentation (with thresholds)"
    # Try to derive bar titles from fig_summary; otherwise fallback
    bar_titles = []
    if getattr(fig_summary.layout, "annotations", None):
        # pick first 3 annotations as subplot titles if present
        for ann in list(fig_summary.layout.annotations)[:3]:
            bar_titles.append(ann.text)
    # fallback to fixed titles
    while len(bar_titles) < 3:
        bar_titles += ["Users", "Revenue (GBP)", "Est. Net Profit (GBP)"][len(bar_titles):]

    combo = make_subplots(
        rows=2, cols=3,
        specs=[[{"type": "xy", "colspan": 3}, None, None],
               [{"type": "xy"}, {"type": "xy"}, {"type": "xy"}]],
        subplot_titles=(lt, bar_titles[0], bar_titles[1], bar_titles[2]),
        horizontal_spacing=0.08, vertical_spacing=0.12
    )

    # Row 1: scatter
    if fig_scatter:
        for tr in fig_scatter.data:
            combo.add_trace(tr, row=1, col=1)

    # Copy threshold lines / annotations from the scatter
    if copy_left_shapes and getattr(fig_scatter.layout, "shapes", None):
        for shp in fig_scatter.layout.shapes:
            s = deepcopy(shp)
            s["xref"] = "x1"; s["yref"] = "y1"
            combo.add_shape(s)

    if copy_left_annotations and getattr(fig_scatter.layout, "annotations", None):
        for ann in fig_scatter.layout.annotations:
            a = deepcopy(ann)
            a["xref"] = "x1"; a["yref"] = "y1"
            combo.add_annotation(a)

    # Row 2: reuse bars from fig_summary, map 1st/2nd/3rd trace to (row=2,col=1/2/3)
    # Assumes your make_segment_summary_plot() added 3 bar traces in order.
    for i, tr in enumerate(fig_summary.data):
        col = min(i + 1, 3)
        combo.add_trace(tr, row=2, col=col)

    # Axis labels for the scatter row (optional: keep whatever is in fig_scatter)
    combo.update_yaxes(title_text="Churn Probability", row=1, col=1)
    combo.update_xaxes(title_text="LTV Proxy (GBP)",   row=1, col=1)

    combo.update_layout(title=title, height=height, showlegend=False)
    return combo

# ---------------------------
# Backwards-compat wrapper
# ---------------------------
def combine_dashboard(
    *,
    fig_funnel=None, fig_ci=None, fig_profit=None, fig_stage=None,
    fig_kpi=None, fig_cohort=None, fig_hist=None, fig_lift=None, fig_pr=None, fig_roc=None,
    mode="overview", title="Dashboard", height=None
):
    """
    Deprecated wrapper for backward compatibility.
    Use:
      - combine_overview(...)
      - combine_experiment(...)
      - combine_segments(...)
    """
    if mode == "experiment":
        return combine_experiment(
            fig_funnel=fig_funnel, fig_ci=fig_ci, fig_profit=fig_profit, fig_stage=fig_stage,
            title=title or "Experiment Dashboard", height=height or 1200
        )
    # default overview
    return combine_overview(
        fig_kpi=fig_kpi, fig_cohort=fig_cohort, fig_hist=fig_hist, fig_lift=fig_lift, fig_pr=fig_pr, fig_roc=fig_roc,
        title=title or "Overview Dashboard", height=height or 1500
    )