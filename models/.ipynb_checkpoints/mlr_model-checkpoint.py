from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import numpy as np
from .base_model import BaseModel
import pandas as pd


class MLRModel(BaseModel):
    def __init__(self, config):
        super().__init__(config, LinearRegression)
        self.scaler_y = StandardScaler()
        self.target_scaled = False

    def hyperparameter_tuning(self, X_train, X_val, y_train, y_val):
        """MLR没有超参数需要优化，直接训练模型"""
        print("MLR没有超参数需要优化，直接训练模型...")

        # 对目标变量进行标准化（如果配置需要）
        if self.config.scale_target:
            # 修复：y_train 是 Series，需要用 .values 转为 numpy 数组才能 reshape
            y_train_data = y_train.values if isinstance(y_train, pd.Series) else y_train
            y_train_scaled = self.scaler_y.fit_transform(y_train_data.reshape(-1, 1)).flatten()
            self.target_scaled = True
        else:
            y_train_scaled = y_train

        # 直接训练模型
        self.model = self.model_class(**self.config.fixed_params)
        self.model.fit(X_train, y_train_scaled)

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
        print("训练MLR最终模型...")

        # 对目标变量进行标准化（如果配置需要）
        if self.config.scale_target:
            # 注意：这里的 y_train 通常已经是 numpy 数组（来自 combine_train_val），可以直接 reshape
            y_train_scaled = self.scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
            self.target_scaled = True
        else:
            y_train_scaled = y_train

        self.model = self.model_class(**self.best_params)
        self.model.fit(X_train, y_train_scaled)
        return self.model

    def predict(self, X):
        """预测"""
        predictions_scaled = self.model.predict(X)

        # 如果目标变量被标准化了，需要反标准化
        if self.target_scaled:
            predictions = self.scaler_y.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()
        else:
            predictions = predictions_scaled

        return predictions

    def get_feature_importance(self):
        """获取特征重要性 - 使用系数的绝对值"""
        if hasattr(self.model, 'coef_'):
            return np.abs(self.model.coef_)
        else:
            return np.array([])

    def get_model_coefficients(self, feature_names):
        """获取模型系数（MLR特有）"""
        coefficients = {}
        if hasattr(self.model, 'intercept_'):
            coefficients['intercept'] = self.model.intercept_
        if hasattr(self.model, 'coef_'):
            for i, coef in enumerate(self.model.coef_):
                feature_name = feature_names[i] if i < len(feature_names) else f'Feature_{i + 1}'
                coefficients[feature_name] = coef
        return coefficients