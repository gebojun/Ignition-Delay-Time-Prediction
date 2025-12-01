from tabpfn import TabPFNRegressor
from .base_model import BaseModel
import numpy as np
import pandas as pd

class TabPFNModel(BaseModel):
    def __init__(self, config):
        super().__init__(config, TabPFNRegressor)

    def hyperparameter_tuning(self, X_train, X_val, y_train, y_val):
        """TabPFN没有超参数需要优化，直接训练模型"""
        print("TabPFN没有超参数需要优化，直接训练模型...")

        # 直接训练模型
        self.model = self.model_class(device=self.config.device)
        self.model.fit(X_train, y_train)

        # 创建虚拟的grid_search对象以保持接口一致
        class DummyGridSearch:
            def __init__(self, model, best_params):
                self.best_estimator_ = model
                self.best_params_ = best_params
                self.cv_results_ = {
                    'params': [{}],
                    'mean_test_score': [0],
                    'std_test_score': [0]
                }

        grid_search = DummyGridSearch(self.model, {})

        # 创建虚拟的优化结果
        optimization_results = pd.DataFrame({
            'params': [{}],
            'mean_test_score': [0],
            'std_test_score': [0],
            'rank_test_score': [1]
        })

        self.best_params = {}

        return grid_search, optimization_results

    def train_final_model(self, X_train, y_train):
        """训练最终模型"""
        print("训练TabPFN最终模型...")
        self.model = TabPFNRegressor(device=self.config.device)
        self.model.fit(X_train, y_train)
        return self.model

    def get_feature_importance(self):
        """获取特征重要性 - TabPFN没有特征重要性，返回空数组"""
        return np.array([])