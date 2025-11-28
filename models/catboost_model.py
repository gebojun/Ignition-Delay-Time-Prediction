from catboost import CatBoostRegressor
from .base_model import BaseModel


class CatBoostModel(BaseModel):
    def __init__(self, config):
        super().__init__(config, CatBoostRegressor)

    def get_feature_importance(self):
        """获取CatBoost特征重要性"""
        return self.model.get_feature_importance()