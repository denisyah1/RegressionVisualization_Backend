from pydantic import BaseModel
from typing import Dict, List, Optional

class ModelMetric(BaseModel):
    train_r2: Optional[float]
    test_r2: Optional[float]
    test_mse: Optional[float]

class RegressionResponse(BaseModel):
    best_model: str
    model_comparison: Dict[str, ModelMetric]
    feature_engineering: Dict[str, List[str]]
    data_info: Dict[str, int | str]
    saved_model_filename: str
