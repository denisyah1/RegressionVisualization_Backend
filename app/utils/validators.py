import pandas as pd
from fastapi import HTTPException

def ensure_numeric(df: pd.DataFrame):
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise HTTPException(
                status_code=400,
                detail=f"Column '{col}' must be numeric"
            )
