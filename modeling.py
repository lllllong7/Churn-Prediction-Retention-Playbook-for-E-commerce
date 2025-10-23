
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score, average_precision_score

def train_and_eval(
    X: pd.DataFrame, 
    y: pd.Series, 
    test_size: float = 0.3, 
    seed: int = 42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=seed, class_weight="balanced")
    rf_model.fit(X_train, y_train)
    p_test = rf_model.predict_proba(X_test)[:, 1]
    y_pred = (p_test > 0.5).astype(int)
    metrics = dict(
        recall=float(recall_score(y_test, y_pred)),
        precision=float(precision_score(y_test, y_pred)),
        f1=float(f1_score(y_test, y_pred)),
        roc_auc=float(roc_auc_score(y_test, p_test)),
        pr_auc=float(average_precision_score(y_test, p_test))
    )
    return rf_model, y_test, p_test, metrics

def get_feature_importance(model, feature_names):
    if hasattr(model, "feature_importances_"):
        return pd.DataFrame({"Feature": feature_names, "Importance": model.feature_importances_}).sort_values("Importance", ascending=False)
    return pd.DataFrame(columns=["Feature","Importance"])

def evaluate_at_threshold(y_true, y_score, thr: float):
    y_pred = (y_score >= thr).astype(int)
    return dict(recall=float(__import__("sklearn.metrics").metrics.recall_score(y_true, y_pred)),
                precision=float(__import__("sklearn.metrics").metrics.precision_score(y_true, y_pred)),
                f1=float(__import__("sklearn.metrics").metrics.f1_score(y_true, y_pred)),
                threshold=float(thr))
