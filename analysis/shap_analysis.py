import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tabpfn import TabPFNRegressor
import numpy as np
import os

class SHAPAnalyzer:
    def __init__(self, data_path, output_dir):
        self.data_path = data_path
        self.output_dir = output_dir

    def analyze(self):
        """执行SHAP分析"""
        # 加载数据
        # 修改：使用 read_csv
        data = pd.read_csv(self.data_path)

        # 数据准备
        X = data.iloc[:, 1:]
        y = data.iloc[:, 0]

        # 划分数据集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.4, random_state=42
        )

        # 训练模型（使用GPU）
        # 注意：如果您没有Nvidia显卡，建议将 device='cuda' 改为 device='cpu'
        reg = TabPFNRegressor(device='cuda')
        reg.fit(X_train, y_train)

        # SHAP分析
        explainer = shap.KernelExplainer(reg.predict, X_train)
        shap_values = explainer.shap_values(X_train)

        # 确保数据格式正确
        shap_values = np.array(shap_values)
        X_train_df = pd.DataFrame(X_train, columns=X.columns)

        # 创建自定义图形并保存
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_train_df, plot_type="dot", show=False)

        output_path = os.path.join(self.output_dir, "shap_beeswarm.png")
        plt.savefig(
            output_path,
            dpi=300,
            bbox_inches='tight',
            transparent=False,
            facecolor='white'
        )

        plt.close()
        print(f"SHAP蜂群图已保存至：{output_path}")

        return shap_values