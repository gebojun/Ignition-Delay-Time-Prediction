from sklearn.neural_network import MLPRegressor
from .base_model import BaseModel
import numpy as np


class ANNModel(BaseModel):
    def __init__(self, config):
        super().__init__(config, MLPRegressor)

    def get_feature_importance(self):
        return np.array([])