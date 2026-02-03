import pandas as pd
from app.utils.json_sanitizer import sanitize

def analyze_eda(df: pd.DataFrame):
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(exclude="number").columns.tolist()

    # Head & Tail
    head = df.head(5).to_dict(orient="records")
    tail = df.tail(5).to_dict(orient="records")

    # Numeric summary
    numeric_summary = {}
    for col in numeric_cols:
        numeric_summary[col] = {
            "mean": df[col].mean(),
            "std": df[col].std(),
            "min": df[col].min(),
            "max": df[col].max(),
            "null_count": int(df[col].isnull().sum())
        }

    # Categorical summary
    categorical_summary = {}
    for col in categorical_cols:
        categorical_summary[col] = {
            "unique_count": int(df[col].nunique()),
            "null_count": int(df[col].isnull().sum())
        }

    # Correlation matrix
    correlation_matrix = {}
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr()
        correlation_matrix = corr.to_dict()

    response = {
        "head": head,
        "tail": tail,
        "columns": {
            "numeric": numeric_cols,
            "categorical": categorical_cols
        },
        "summary_statistics": numeric_summary,
        "categorical_summary": categorical_summary,
        "correlation_matrix": correlation_matrix
    }

    return sanitize(response)
