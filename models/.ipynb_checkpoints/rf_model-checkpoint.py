from sklearn.ensemble import RandomForestRegressor
from .base_model import BaseModel


class RFModel(BaseModel):
    def __init__(self, config):
        super().__init__(config, RandomForestRegressor)

    def get_feature_importance(self):
        """获取随机森林特征重要性"""
        return self.model.feature_importances_