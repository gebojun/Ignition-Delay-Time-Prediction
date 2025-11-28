from sklearn.svm import SVR
from .base_model import BaseModel
import numpy as np


class SVMModel(BaseModel):
    def __init__(self, config):
        super().__init__(config, SVR)

    def get_feature_importance(self):
        """获取SVM特征重要性 - 对于非线性SVM，特征重要性不太直观，返回空数组"""
        if hasattr(self.model, 'coef_') and self.model.kernel == 'linear':
            return np.abs(self.model.coef_[0])
        else:
            return np.array([])