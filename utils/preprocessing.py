from sklearn.preprocessing import StandardScaler
import numpy as np


class Preprocessor:
    def __init__(self):
        self.scaler = StandardScaler()

    def fit_transform(self, X_train, X_val, X_test):
        """特征缩放"""
        print("进行特征缩放...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        return X_train_scaled, X_val_scaled, X_test_scaled

    def combine_train_val(self, X_train, X_val, y_train, y_val):
        """合并训练集和验证集"""
        X_combined = np.vstack([X_train, X_val])
        y_combined = np.concatenate([y_train, y_val])
        return X_combined, y_combined