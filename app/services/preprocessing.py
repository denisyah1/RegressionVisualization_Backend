from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def build_preprocessor(numeric_features, categorical_features):
    transformers = []

    if numeric_features:
        transformers.append(
            ("num", StandardScaler(), numeric_features)
        )

    if categorical_features:
        transformers.append(
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        )

    return ColumnTransformer(transformers)
