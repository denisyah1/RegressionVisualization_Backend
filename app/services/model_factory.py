from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

def get_regression_models(preprocessor):
    return {
        "LinearRegression": Pipeline([
            ("preprocess", preprocessor),
            ("model", LinearRegression())
        ]),
        "Ridge": Pipeline([
            ("preprocess", preprocessor),
            ("model", Ridge(alpha=1.0))
        ]),
        "Lasso": Pipeline([
            ("preprocess", preprocessor),
            ("model", Lasso(alpha=0.01))
        ]),
        "ElasticNet": Pipeline([
            ("preprocess", preprocessor),
            ("model", ElasticNet(alpha=0.01, l1_ratio=0.5))
        ]),
        "PolynomialRegression": Pipeline([
            ("preprocess", preprocessor),
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("model", LinearRegression())
        ])
    }
