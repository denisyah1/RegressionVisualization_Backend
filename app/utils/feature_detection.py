import pandas as pd

def detect_feature_types(df, feature_columns):
    numeric_features = []
    categorical_features = []

    for col in feature_columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_features.append(col)
        else:
            categorical_features.append(col)

    return numeric_features, categorical_features
