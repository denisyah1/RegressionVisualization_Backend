# Regression Visualization Backend

FastAPI service that powers CSV analysis, regression training, and export features.

## Features
- CSV preview endpoint.
- EDA analysis: numeric/categorical columns, summary stats, correlation matrix, head/tail preview.
- Regression training with automatic model comparison and best-model selection.
- Null handling strategy: `auto`, `mean`, or `drop`.
- Regression plot data stored in-memory for quick retrieval.
- Download saved model file (`.pkl`).
- Column recommendation for target and features.

## Tech Stack
- FastAPI, Uvicorn
- Pandas, NumPy, scikit-learn

## Setup
1. Create and activate a virtual environment (recommended).
2. Install dependencies.
   ```bash
   pip install -r requirements.txt
   ```
3. Run the API server.
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```
4. Configure CORS origins (optional for local dev).
   - Copy `.env.example` to `.env` and set `CORS_ORIGINS` as a comma-separated list.
   - Example: `CORS_ORIGINS=http://localhost:5173`

## API Endpoints
- `POST /api/csv/preview` - upload CSV and return a quick preview.
- `POST /api/csv/eda` - return EDA summary and correlation matrix.
- `POST /api/csv/recommendation` - suggest target/features and columns to drop.
- `POST /api/regression` - run regression and return model comparison + saved model filename.
- `GET /api/regression/plot` - return plot data for the last regression run.
- `GET /api/model/download?filename=...` - download the saved model file.

## Notes
- CORS allows `http://localhost:5173` by default for the frontend dev server.
- Plot data is stored in-memory for the latest regression run and will reset on server restart.
- For Railway, set `CORS_ORIGINS` in project environment variables.
