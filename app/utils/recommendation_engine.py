import pandas as pd
import numpy as np
from app.utils.json_sanitizer import sanitize

def recommend_regression_columns(df: pd.DataFrame):
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(exclude="number").columns.tolist()

    recommendations = {
        "target_candidates": [],
        "feature_candidates": [],
        "drop_recommendations": []
    }

    # Drop rules
    for col in df.columns:
        null_ratio = df[col].isnull().mean()
        unique_ratio = df[col].nunique() / len(df)

        if null_ratio > 0.5:
            recommendations["drop_recommendations"].append({
                "column": col,
                "reason": "High null ratio (>50%)"
            })
        elif unique_ratio > 0.95:
            recommendations["drop_recommendations"].append({
                "column": col,
                "reason": "Likely ID / high cardinality"
            })

    # Correlation matrix
    corr = df[numeric_cols].corr() if len(numeric_cols) >= 2 else None

    # Target recommendation
    for col in numeric_cols:
        if df[col].std() == 0:
            continue

        score = 0
        if corr is not None:
            score = (corr[col].abs() > 0.3).sum() - 1

        recommendations["target_candidates"].append({
            "column": col,
            "score": int(score),
            "std": df[col].std()
        })

    recommendations["target_candidates"].sort(
        key=lambda x: (x["score"], x["std"]),
        reverse=True
    )

    # Feature recommendation (based on top target)
    if recommendations["target_candidates"]:
        best_target = recommendations["target_candidates"][0]["column"]

        for col in df.columns:
            if col == best_target:
                continue

            if col in numeric_cols:
                corr_value = (
                    corr.loc[col, best_target]
                    if corr is not None and col in corr.index
                    else 0
                )

                if abs(corr_value) > 0.2:
                    recommendations["feature_candidates"].append({
                        "column": col,
                        "type": "numeric",
                        "correlation": corr_value
                    })

            elif col in categorical_cols:
                recommendations["feature_candidates"].append({
                    "column": col,
                    "type": "categorical",
                    "note": "Will be OneHotEncoded"
                })

 # ===== DEFAULT AUTO-FILL =====
    default_target = None
    default_features = []

    if recommendations["target_candidates"]:
        default_target = recommendations["target_candidates"][0]["column"]

        for feat in recommendations["feature_candidates"]:
            if feat["column"] != default_target:
                default_features.append(feat["column"])

        # batasi jumlah feature default
        default_features = default_features[:6]

    recommendations["default_selection"] = {
        "target": default_target,
        "features": default_features
    }

    return sanitize(recommendations)
