import math
import pandas as pd
from fastapi import HTTPException
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from app.utils.csv_loader import load_csv
from app.utils.data_cleaning import clean_dataframe
from app.utils.feature_detection import detect_feature_types
from app.services.preprocessing import build_preprocessor
from app.services.model_factory import get_regression_models
from app.utils.model_storage import save_model
from app.utils.json_sanitizer import sanitize
from app.services.plot_store import PLOT_STORE
from app.core.config import DEFAULT_NULL_STRATEGY, TRAIN_TEST_SPLIT_RATIO


def run_regression(
    file,
    target_column=None,
    feature_columns=None,
    null_strategy=None,
    drop_columns=None
):
    # =====================================================
    # LOAD CSV
    # =====================================================
    df = load_csv(file)

    # Normalize column names to avoid whitespace mismatches
    original_cols = list(df.columns)
    stripped_cols = [c.strip() if isinstance(c, str) else c for c in original_cols]
    if len(set(stripped_cols)) != len(stripped_cols):
        raise HTTPException(400, "Duplicate columns detected after trimming spaces")
    if stripped_cols != original_cols:
        df.columns = stripped_cols
    if target_column:
        target_column = target_column.strip()
    if feature_columns:
        feature_columns = [c.strip() for c in feature_columns]

    if not feature_columns:
        raise HTTPException(400, "Feature columns cannot be empty")

    if target_column not in df.columns:
        raise HTTPException(400, f"Target column '{target_column}' not found")

    for col in feature_columns:
        if col not in df.columns:
            raise HTTPException(400, f"Feature column '{col}' not found")

    # =====================================================
    # DROP COLUMNS (OPTIONAL)
    # =====================================================
    if drop_columns:
        df = df.drop(columns=drop_columns, errors="ignore")

    strategy = null_strategy or DEFAULT_NULL_STRATEGY

    # =====================================================
    # CLEAN DATA
    # =====================================================
    df = clean_dataframe(df, feature_columns, target_column, strategy)

    X = df[feature_columns]
    y = df[target_column]

    if not pd.api.types.is_numeric_dtype(y):
        raise HTTPException(
            status_code=400,
            detail=f"Target column '{target_column}' must be numeric for regression"
        )

    # =====================================================
    # FEATURE TYPE DETECTION
    # =====================================================
    numeric_features, categorical_features = detect_feature_types(df, feature_columns)

    if not numeric_features and not categorical_features:
        raise HTTPException(400, "No valid features detected")

    # =====================================================
    # PREPROCESSING
    # =====================================================
    preprocessor = build_preprocessor(numeric_features, categorical_features)

    # =====================================================
    # TRAIN / TEST SPLIT
    # =====================================================
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TRAIN_TEST_SPLIT_RATIO,
        random_state=42
    )

    if len(y_test) < 2:
        raise HTTPException(
            status_code=400,
            detail="Not enough test samples to evaluate regression. Please provide more data."
        )

    # =====================================================
    # MODEL TRAINING & SELECTION
    # =====================================================
    models = get_regression_models(preprocessor)

    best_model = None
    best_model_name = None
    best_r2 = -1e9
    best_test_pred = None

    results = {}

    for name, model in models.items():
        try:
            model.fit(X_train, y_train)

            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)

            train_r2 = r2_score(y_train, train_pred)
            test_r2 = r2_score(y_test, test_pred)

            if math.isnan(test_r2):
                test_r2 = None

            test_mse = mean_squared_error(y_test, test_pred)

            results[name] = {
                "train_r2": train_r2,
                "test_r2": test_r2,
                "test_mse": test_mse
            }

            if test_r2 is not None and test_r2 > best_r2:
                best_r2 = test_r2
                best_model = model
                best_model_name = name
                best_test_pred = test_pred

        except Exception as e:
            results[name] = {"error": str(e)}

    if not best_model:
        raise HTTPException(500, "All models failed")

    # =====================================================
    # STORE PLOT DATA (BEST MODEL ONLY)
    # =====================================================
    PLOT_STORE["last"] = {
        "train": {
            "y_actual": y_train.tolist(),
            "y_pred": best_model.predict(X_train).tolist()
        },
        "test": {
            "y_actual": y_test.tolist(),
            "y_pred": best_test_pred.tolist()
        }
    }

    # =====================================================
    # SAVE BEST MODEL
    # =====================================================
    saved_model_info = save_model(best_model, best_model_name)

    response = {
        "best_model": best_model_name,
        "feature_engineering": {
            "numeric_features": numeric_features,
            "categorical_features": categorical_features
        },
        "data_info": {
            "rows": len(df),
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "null_strategy": strategy
        },
        "model_comparison": results,
        "saved_model_filename": saved_model_info["filename"]
    }

    return sanitize(response)
