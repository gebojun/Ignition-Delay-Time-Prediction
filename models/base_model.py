import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from utils.evaluation import ModelEvaluator


class BaseModel:
    def __init__(self, config, model_class):
        self.config = config
        self.model_class = model_class
        self.evaluator = ModelEvaluator(config)
        self.model = None
        self.best_params = None

    def hyperparameter_tuning(self, X_train, X_val, y_train, y_val):
        """超参数优化"""
        print(f"开始{self.config.__class__.__name__}参数优化...")

        # 创建PredefinedSplit
        X_combined = np.vstack([X_train, X_val])
        y_combined = np.concatenate([y_train, y_val])
        test_fold = np.array([-1] * len(X_train) + [0] * len(X_val))
        ps = PredefinedSplit(test_fold)

        # 初始化模型
        model = self.model_class(**self.config.fixed_params)

        # 网格搜索
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=self.config.param_grid,
            cv=ps,
            scoring='r2',
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(X_combined, y_combined)

        print(f"\n最佳参数: {grid_search.best_params_}")
        print(f"最佳R²分数: {grid_search.best_score_:.5f}")

        self.best_params = grid_search.best_params_.copy()
        self.best_params.update(self.config.fixed_params)

        # 保存优化结果
        optimization_results = pd.DataFrame({
            'params': grid_search.cv_results_['params'],
            'mean_test_score': grid_search.cv_results_['mean_test_score'],
            'std_test_score': grid_search.cv_results_['std_test_score'],
            'rank_test_score': grid_search.cv_results_['rank_test_score']
        }).sort_values('rank_test_score')

        return grid_search, optimization_results

    def train_final_model(self, X_train, y_train):
        """训练最终模型"""
        print("训练最终模型...")
        self.model = self.model_class(**self.best_params)
        self.model.fit(X_train, y_train)
        return self.model

    def predict(self, X):
        """预测"""
        return self.model.predict(X)

    def evaluate(self, X_val, y_val, X_test, y_test):
        """模型评估"""
        # 验证集预测
        val_predictions = self.predict(X_val)
        val_rmse, val_mae, val_r2 = self.evaluator.calculate_metrics(y_val, val_predictions)

        # 测试集预测
        test_predictions = self.predict(X_test)
        test_rmse, test_mae, test_r2 = self.evaluator.calculate_metrics(y_test, test_predictions)

        val_metrics = {'rmse': val_rmse, 'mae': val_mae, 'r2': val_r2}
        test_metrics = {'rmse': test_rmse, 'mae': test_mae, 'r2': test_r2}

        return val_predictions, test_predictions, val_metrics, test_metrics

    def get_feature_importance(self):
        """获取特征重要性 - 需要在子类中实现"""
        raise NotImplementedError("子类必须实现此方法")