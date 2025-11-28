from lightgbm import LGBMRegressor
from .base_model import BaseModel


class LightGBMModel(BaseModel):
    def __init__(self, config):
        super().__init__(config, LGBMRegressor)

    def get_feature_importance(self):
        """获取LightGBM特征重要性"""
        return self.model.feature_importances_