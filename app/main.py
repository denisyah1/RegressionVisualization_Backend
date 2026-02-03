from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import os
from app.utils.csv_preview import analyze_csv
from app.services.regression_service import run_regression
from app.services.plot_store import PLOT_STORE
from fastapi import HTTPException
from app.utils.csv_loader import load_csv
from app.utils.eda_analyzer import analyze_eda

app = FastAPI(title="Regression Visualization API")

# =========================
# CSV PREVIEW
# =========================
@app.post("/api/csv/preview")
async def preview_csv(
    file: UploadFile = File(...)
):
    return analyze_csv(file)

# =========================
# REGRESSION (CSV RAW → ML)
# =========================
@app.post("/api/regression")
async def regression(
    file: UploadFile = File(...),
    target_column: str = "",
    feature_columns: str = "",
    null_strategy: str = "drop"
):
    features = [c.strip() for c in feature_columns.split(",") if c.strip()]
    return run_regression(
        file=file,
        target_column=target_column,
        feature_columns=features,
        null_strategy=null_strategy
    )
#==========================
# GET PLOT
#==========================

from app.services.plot_store import PLOT_STORE
from fastapi import HTTPException

@app.get("/api/regression/plot")
def get_regression_plot():
    if "last" not in PLOT_STORE:
        raise HTTPException(
            status_code=404,
            detail="No regression plot data available. Run regression first."
        )

    return PLOT_STORE["last"]

# =========================
# DOWNLOAD SAVED MODEL
# =========================
@app.get("/api/model/download")
async def download_model(filename: str):
    path = f"models/saved/{filename}"

    if not os.path.exists(path):
        return {"error": "Model not found"}

    return FileResponse(
        path,
        media_type="application/octet-stream",
        filename=filename
    )

# =========================
# EDA ANALYSIS
# =========================

@app.post("/api/csv/eda")
async def csv_eda(file: UploadFile = File(...)):
    df = load_csv(file)
    return analyze_eda(df)
