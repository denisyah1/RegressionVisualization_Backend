from pydantic import BaseModel
from typing import Dict, List

class PlotData(BaseModel):
    x: Dict[str, List[float]]
    y_actual: List[float]
    y_pred: List[float]

class RegressionResponse(BaseModel):
    coefficients: List[float]
    intercept: float
    mse: float
    r2: float
    plot_data: PlotData
