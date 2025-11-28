import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_validate


class ModelEvaluator:
    def __init__(self, config):
        self.config = config

    def calculate_metrics(self, y_true, y_pred):
        """计算评估指标"""
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        return rmse, mae, r2

    def cross_validate_model(self, model, X, y):
        """执行交叉验证"""
        print("正在进行交叉验证...")
        cv_results = cross_validate(
            model, X, y,
            cv=self.config.cv_folds,
            scoring=self.config.scoring_metrics,
            return_train_score=True,
            n_jobs=-1
        )

        return self._process_cv_results(cv_results)

    def _process_cv_results(self, cv_results):
        """处理交叉验证结果"""
        cv_data = []
        for i in range(self.config.cv_folds):
            fold_data = {
                'Fold': i + 1,
                'Train_R2': cv_results['train_r2'][i],
                'Test_R2': cv_results['test_r2'][i],
                'Train_MAE': -cv_results['train_neg_mae'][i],
                'Test_MAE': -cv_results['test_neg_mae'][i],
                'Train_RMSE': -cv_results['train_neg_rmse'][i],
                'Test_RMSE': -cv_results['test_neg_rmse'][i]
            }
            cv_data.append(fold_data)

        cv_df = pd.DataFrame(cv_data)

        # 计算平均值和标准差
        mean_values = {
            'Fold': 'Mean',
            'Train_R2': cv_df['Train_R2'].mean(),
            'Test_R2': cv_df['Test_R2'].mean(),
            'Train_MAE': cv_df['Train_MAE'].mean(),
            'Test_MAE': cv_df['Test_MAE'].mean(),
            'Train_RMSE': cv_df['Train_RMSE'].mean(),
            'Test_RMSE': cv_df['Test_RMSE'].mean()
        }

        std_values = {
            'Fold': 'Std',
            'Train_R2': cv_df['Train_R2'].std(),
            'Test_R2': cv_df['Test_R2'].std(),
            'Train_MAE': cv_df['Train_MAE'].std(),
            'Test_MAE': cv_df['Test_MAE'].std(),
            'Train_RMSE': cv_df['Train_RMSE'].std(),
            'Test_RMSE': cv_df['Test_RMSE'].std()
        }

        cv_summary = pd.concat([cv_df, pd.DataFrame([mean_values]), pd.DataFrame([std_values])], ignore_index=True)
        return cv_summary, mean_values, std_values

    def diagnose_model(self, val_r2, test_r2, model_name):
        """模型诊断"""
        print(f"\n=== {model_name} 诊断信息 ===")
        print(f"验证集R²: {val_r2:.5f}")
        print(f"测试集R²: {test_r2:.5f}")

        # 检查是否过拟合
        overfitting_gap = val_r2 - test_r2
        print(f"过拟合程度 (验证集R² - 测试集R²): {overfitting_gap:.5f}")
        if overfitting_gap > 0.1:
            print("警告: 模型可能存在过拟合")
        elif overfitting_gap < -0.05:
            print("警告: 模型可能存在欠拟合")
        else:
            print("模型拟合程度良好")

        return overfitting_gap