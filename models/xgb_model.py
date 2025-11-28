from xgboost import XGBRegressor
from .base_model import BaseModel


class XGBModel(BaseModel):
    def __init__(self, config):
        super().__init__(config, XGBRegressor)

    def get_feature_importance(self):
        """获取XGBoost特征重要性"""
        return self.model.feature_importances_