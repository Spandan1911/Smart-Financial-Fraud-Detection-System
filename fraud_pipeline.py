"""
fraud_pipeline.py - Core ML pipeline for Smart Financial Fraud Detection System.
Algorithms: K-Means, Isolation Forest, Random Forest, LightGBM, SHAP
"""

import os
import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report

try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

FEATURE_COLS = [f"V{i}" for i in range(1, 15)] + ["Amount", "Time"]


def train_all_models(df: pd.DataFrame, n_clusters: int = 5, random_state: int = 42) -> dict:
    X = df[FEATURE_COLS].copy()

    # If Class column exists use it, else treat all as unknown
    has_labels = "Class" in df.columns and df["Class"].nunique() > 1
    if has_labels:
        y = df["Class"].copy()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, stratify=y, random_state=random_state
        )
    else:
        y = pd.Series([0] * len(df))
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=random_state
        )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    metrics = {}

    # 1. K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    train_clusters = kmeans.fit_predict(X_train_scaled)
    cluster_fraud_rate = (
        pd.DataFrame({"cluster": train_clusters, "Class": y_train.values})
        .groupby("cluster")["Class"].mean()
        .to_dict()
    )

    # 2. Isolation Forest
    contamination = float(y_train.mean()) if has_labels and y_train.mean() > 0 else 0.01
    iso_forest = IsolationForest(n_estimators=200, contamination=contamination, random_state=random_state)
    iso_forest.fit(X_train_scaled)

    train_iso_raw = -iso_forest.decision_function(X_train_scaled)
    test_iso_raw = -iso_forest.decision_function(X_test_scaled)
    iso_min, iso_max = train_iso_raw.min(), train_iso_raw.max()

    def norm_iso(scores):
        return np.clip((scores - iso_min) / (iso_max - iso_min + 1e-9), 0, 1)

    train_iso_score = norm_iso(train_iso_raw)
    test_iso_score = norm_iso(test_iso_raw)

    def augment(X_raw, X_scaled, clusters_model, fraud_rate_map, iso_scores):
        clusters = clusters_model.predict(X_scaled)
        cluster_risk = np.array([fraud_rate_map.get(c, 0.0) for c in clusters])
        X_aug = X_raw.copy().reset_index(drop=True)
        X_aug["cluster"] = clusters
        X_aug["cluster_risk"] = cluster_risk
        X_aug["iso_score"] = iso_scores
        return X_aug

    X_train_aug = augment(X_train, X_train_scaled, kmeans, cluster_fraud_rate, train_iso_score)
    X_test_aug = augment(X_test, X_test_scaled, kmeans, cluster_fraud_rate, test_iso_score)
    AUG_COLS = FEATURE_COLS + ["cluster", "cluster_risk", "iso_score"]

    # 3. Random Forest
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=10, class_weight="balanced",
        random_state=random_state, n_jobs=-1
    )
    rf.fit(X_train_aug[AUG_COLS], y_train)

    # 4. LightGBM
    if LGBM_AVAILABLE:
        scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
        lgbm = lgb.LGBMClassifier(
            n_estimators=300, learning_rate=0.05,
            scale_pos_weight=scale_pos_weight, random_state=random_state, verbose=-1
        )
        lgbm.fit(X_train_aug[AUG_COLS], y_train)
    else:
        lgbm = None

    # Save all artifacts
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))
    joblib.dump(kmeans, os.path.join(MODEL_DIR, "kmeans.joblib"))
    joblib.dump(iso_forest, os.path.join(MODEL_DIR, "isolation_forest.joblib"))
    joblib.dump(rf, os.path.join(MODEL_DIR, "random_forest.joblib"))
    joblib.dump(cluster_fraud_rate, os.path.join(MODEL_DIR, "cluster_fraud_rate.joblib"))
    joblib.dump({"iso_min": iso_min, "iso_max": iso_max}, os.path.join(MODEL_DIR, "iso_norm.joblib"))
    if lgbm is not None:
        joblib.dump(lgbm, os.path.join(MODEL_DIR, "lightgbm.joblib"))

    metrics["lgbm_available"] = LGBM_AVAILABLE
    metrics["shap_available"] = SHAP_AVAILABLE
    return metrics


class FraudDetectionEngine:
    def __init__(self):
        self.scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
        self.kmeans = joblib.load(os.path.join(MODEL_DIR, "kmeans.joblib"))
        self.iso_forest = joblib.load(os.path.join(MODEL_DIR, "isolation_forest.joblib"))
        self.rf = joblib.load(os.path.join(MODEL_DIR, "random_forest.joblib"))
        self.cluster_fraud_rate = joblib.load(os.path.join(MODEL_DIR, "cluster_fraud_rate.joblib"))
        self.iso_norm = joblib.load(os.path.join(MODEL_DIR, "iso_norm.joblib"))

        lgbm_path = os.path.join(MODEL_DIR, "lightgbm.joblib")
        self.lgbm = joblib.load(lgbm_path) if os.path.exists(lgbm_path) else None
        self.aug_feature_cols = FEATURE_COLS + ["cluster", "cluster_risk", "iso_score"]

        self.shap_explainer = None
        if SHAP_AVAILABLE:
            model_for_shap = self.lgbm if self.lgbm is not None else self.rf
            try:
                self.shap_explainer = shap.TreeExplainer(model_for_shap)
            except Exception:
                self.shap_explainer = None

    def _build_augmented_features(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[FEATURE_COLS].copy()
        X_scaled = self.scaler.transform(X)
        clusters = self.kmeans.predict(X_scaled)
        cluster_risk = np.array([self.cluster_fraud_rate.get(c, 0.0) for c in clusters])
        iso_raw = -self.iso_forest.decision_function(X_scaled)
        iso_min, iso_max = self.iso_norm["iso_min"], self.iso_norm["iso_max"]
        iso_score = np.clip((iso_raw - iso_min) / (iso_max - iso_min + 1e-9), 0, 1)
        X_aug = X.copy().reset_index(drop=True)
        X_aug["cluster"] = clusters
        X_aug["cluster_risk"] = cluster_risk
        X_aug["iso_score"] = iso_score
        return X_aug

    def score_transactions(self, df: pd.DataFrame) -> pd.DataFrame:
        X_aug = self._build_augmented_features(df)
        rf_proba = self.rf.predict_proba(X_aug[self.aug_feature_cols])[:, 1]
        lgbm_proba = self.lgbm.predict_proba(X_aug[self.aug_feature_cols])[:, 1] if self.lgbm else rf_proba

        risk = (
            0.40 * lgbm_proba +
            0.25 * rf_proba +
            0.20 * X_aug["iso_score"].values +
            0.15 * X_aug["cluster_risk"].values
        )
        risk_score = np.clip(risk * 100, 0, 100)

        result = df.copy().reset_index(drop=True)
        result["cluster"] = X_aug["cluster"]
        result["cluster_risk"] = X_aug["cluster_risk"]
        result["iso_score"] = X_aug["iso_score"]
        result["rf_proba"] = rf_proba
        result["lgbm_proba"] = lgbm_proba
        result["risk_score"] = risk_score
        result["is_fraud_predicted"] = (risk_score >= 50)
        return result

    def explain_transaction(self, df_row: pd.DataFrame, top_n: int = 5) -> dict:
        X_aug = self._build_augmented_features(df_row)

        if self.shap_explainer is not None:
            shap_values = self.shap_explainer.shap_values(X_aug[self.aug_feature_cols])
            if isinstance(shap_values, list):
                sv = shap_values[1][0]
                base_value = self.shap_explainer.expected_value[1]
            else:
                sv = shap_values[0]
                base_value = self.shap_explainer.expected_value

            contributions = list(zip(self.aug_feature_cols, sv, X_aug.iloc[0].values))
            contributions_sorted = sorted(contributions, key=lambda x: abs(x[1]), reverse=True)[:top_n]
            top_features = [
                {"feature": f, "value": float(v), "shap_value": float(s),
                 "direction": "increases fraud risk" if s > 0 else "decreases fraud risk"}
                for f, s, v in contributions_sorted
            ]
            return {"top_features": top_features, "base_value": float(base_value), "method": "shap"}

        model = self.lgbm if self.lgbm is not None else self.rf
        importances = model.feature_importances_
        contributions = list(zip(self.aug_feature_cols, importances, X_aug.iloc[0].values))
        contributions_sorted = sorted(contributions, key=lambda x: x[1], reverse=True)[:top_n]
        top_features = [
            {"feature": f, "value": float(v), "shap_value": float(imp),
             "direction": "high importance feature"}
            for f, imp, v in contributions_sorted
        ]
        return {"top_features": top_features, "base_value": None, "method": "feature_importance_fallback"}
