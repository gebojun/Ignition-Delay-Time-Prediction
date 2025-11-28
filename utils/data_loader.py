import pandas as pd
from sklearn.model_selection import train_test_split


class DataLoader:
    def __init__(self, config):
        self.config = config

    def load_data(self):
        """加载数据"""
        data = pd.read_excel(self.config.data_path, header=0)
        print(f"数据形状: {data.shape}")
        print(f"数据列名: {data.columns.tolist()}")
        return data

    def split_data(self, X, y):
        """划分训练集、验证集、测试集"""
        # 第一次分割：80%训练+验证，20%测试
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=self.config.test_size, random_state=self.config.random_state
        )

        # 第二次分割：从80%中分割出训练集和验证集
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=self.config.val_size,
            random_state=self.config.random_state
        )

        print(f"训练集大小: {X_train.shape[0]} ({X_train.shape[0] / len(X) * 100:.1f}%)")
        print(f"验证集大小: {X_val.shape[0]} ({X_val.shape[0] / len(X) * 100:.1f}%)")
        print(f"测试集大小: {X_test.shape[0]} ({X_test.shape[0] / len(X) * 100:.1f}%)")

        return X_train, X_val, X_test, y_train, y_val, y_test