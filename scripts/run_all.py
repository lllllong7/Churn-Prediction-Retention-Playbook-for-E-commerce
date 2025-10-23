
import os, json, pandas as pd
from src.churn_playbook import modeling, plotly_helpers as ph, experiment, policy, labeling, etl, features

OUT_DIR   = os.environ.get("OUT_DIR", "../outputs")
FIG_DIR   = os.path.join(OUT_DIR, "figs")
os.makedirs(FIG_DIR, exist_ok=True)

df_raw = etl.load_and_clean()
df_txn, df_ret = etl.split_transactions_and_returns(df_raw)
df_txn = etl.add_sales(df_txn)
wh_ids, *_ = etl.detect_wholesalers(df_txn)
df_txn = etl.add_wholesaler_flag(df_txn, wh_ids)

df_obs, df_pred, X_base, y, T_REF = labeling.make_slice_and_label(df_txn, df_ret)
X = features.build_features(df_txn, df_ret, df_obs, T_REF, X_base)
ltv = labeling.compute_ltv_proxy(df_txn, T_REF)
X = X.merge(ltv, on="CustomerID", how="left").fillna({"Monetary_LTV_Proxy":0}).set_index("CustomerID")

model, y_test, p_test, metrics = modeling.train_and_eval(X, y)
with open(os.path.join(OUT_DIR, "model_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

segment_df, segment_summary = policy.segment_and_summarize(X, y_test, p_test)
segment_df.to_csv(os.path.join(OUT_DIR, "latest_scores_and_segments.csv"), index=False)

# === Overview dashboards ===
kpis = ph.compute_kpis_from_slice(df_obs, y, X, ltv_col="Monetary_LTV_Proxy")
fig_overview = ph.combine_overview(
    fig_kpi=ph.make_kpi_cards(kpis),
    fig_cohort=ph.make_cohort_heatmap(df_txn),
    fig_hist=ph.make_prob_histogram(p_test),
    fig_lift=ph.make_lift_chart(y_test, p_test, n_bins=10),
    fig_pr=ph.make_pr_curve(y_test, p_test)[0],
    fig_roc=ph.make_roc_curve(y_test, p_test)[0],
    title="Overview Dashboard"
)
fig_overview.write_html(os.path.join(FIG_DIR, "overview_dashboard.html"), include_plotlyjs="cdn")


# === Segments dashboards ===
RISK_THR = 0.6
LTV_THR  = segment_df["LTV_Proxy"].quantile(0.75) if "LTV_Proxy" in segment_df.columns else 0.0
fig_seg_scatter = ph.make_ltv_risk_scatter(segment_df, risk_th=RISK_THR, ltv_th=LTV_THR)
fig_seg_summary = ph.make_segment_summary_plot(segment_summary)
fig_segments = ph.combine_segments(fig_scatter=fig_seg_scatter, fig_summary=fig_seg_summary,
                                   title="Segments Dashboard", height=900)
fig_segments.write_html(os.path.join(FIG_DIR, "segments_dashboard.html"), include_plotlyjs="cdn")

# Experiment: simulate from segment_summary
AB_PATH = os.environ.get("AB_PATH", "data/ab_results.csv")
if os.path.exists(AB_PATH):
    ab_df = pd.read_csv(AB_PATH)
    print(f"Loaded A/B results from {AB_PATH}")
else:
    ab_df = experiment.simulate_ab_from_segments(
        segment_summary=segment_summary,
        email_per_user=100,
        seed=42,
        revenue_share_of_ltv=0.6,
    )
    sim_out = os.path.join(OUT_DIR, "ab_results_simulated.csv")
    ab_df.to_csv(sim_out, index=False)
    print(f"Simulated A/B results saved to {sim_out}")

# Normalize columns and group labels
ab_df.columns = [c.lower() for c in ab_df.columns]
if "group" in ab_df.columns:
    ab_df["group"] = ab_df["group"].astype(str).str.lower().map(
        {"control": "control", "treatment": "treatment"}
    ).fillna(ab_df["group"])

# Build Experiment dashboard
fig_funnel = ph.make_ab_funnel(ab_df, by=("strategy","group"))
fig_ci     = ph.make_ab_ci(ab_df, by=("strategy","group"), target=("purchased","delivered"))
fig_profit = ph.make_ab_profit(ab_df, by=("strategy","group"))
fig_stage  = ph.make_ab_stage(ab_df, by=("strategy","group"), rate_mode="stage")

fig_exp = ph.combine_experiment(
    fig_funnel=fig_funnel,
    fig_ci=fig_ci,
    fig_profit=fig_profit,
    fig_stage=fig_stage,
    title="Experiment Dashboard"
)
fig_exp.write_html(os.path.join(FIG_DIR, "experiment_dashboard.html"), include_plotlyjs="cdn")
