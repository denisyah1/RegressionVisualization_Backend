import pandas as pd
from fastapi import HTTPException

def clean_dataframe(df: pd.DataFrame, features: list, target: str, strategy: str):
    selected_cols = features + [target]
    df = df[selected_cols]

    if strategy == "drop":
        df = df.dropna()

    elif strategy in {"mean", "auto"}:
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                if df[col].isnull().any():
                    df[col] = df[col].fillna(df[col].mean())
            else:
                if df[col].isnull().any():
                    mode = df[col].mode(dropna=True)
                    fill_value = mode.iloc[0] if not mode.empty else "Unknown"
                    df[col] = df[col].fillna(fill_value)
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
