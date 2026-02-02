import pandas as pd
from fastapi import HTTPException

def clean_dataframe(df: pd.DataFrame, features: list, target: str, strategy: str):
    selected_cols = features + [target]
    df = df[selected_cols]

    if strategy == "drop":
        df = df.dropna()

    elif strategy == "mean":
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                if df[col].isnull().any():
                    df[col] = df[col].fillna(df[col].mean())
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Column '{col}' is not numeric and cannot be mean-imputed"
                )
    else:
        raise HTTPException(
            status_code=400,
            detail="Invalid null_strategy. Use 'drop' or 'mean'."
        )

    if df.empty:
        raise HTTPException(
            status_code=400,
            detail="No data left after cleaning"
        )

    return df
