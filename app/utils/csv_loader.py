import pandas as pd
from fastapi import HTTPException

def load_csv(file):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="File must be a CSV")

    try:
        df = pd.read_csv(file.file)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid CSV format")

    if df.empty:
        raise HTTPException(status_code=400, detail="CSV file is empty")

    return df
