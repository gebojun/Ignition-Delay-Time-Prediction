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
        # 全局强制设置 Arial 字体
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Arial']
        plt.rcParams['axes.unicode_minus'] = False

    def analyze(self):
        """执行SHAP分析"""
        # 加载数据
        data = pd.read_csv(self.data_path)

        # 数据准备
        X = data.iloc[:, 1:]
        y = data.iloc[:, 0]

        # 划分数据集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.4, random_state=42
        )

        # 训练模型（注意：如果没有GPU，请改为 device='cpu'）
        reg = TabPFNRegressor(device='cuda')
        reg.fit(X_train, y_train)

        # SHAP分析
        explainer = shap.KernelExplainer(reg.predict, X_train)
        shap_values = explainer.shap_values(X_train)

        # 确保数据格式正确
        shap_values = np.array(shap_values)
        X_train_df = pd.DataFrame(X_train, columns=X.columns)

        # 创建图形
        plt.figure(figsize=(10, 8))

        # 绘制 SHAP 蜂群图
        # alpha=0.6: 增加透明度，解决点密集重叠问题
        # cmap: 强制使用 coolwarm 色系
        shap.summary_plot(
            shap_values,
            X_train_df,
            plot_type="dot",
            show=False,
            cmap=plt.get_cmap("coolwarm"),
            alpha=0.6
        )

        # 【核心修改】强制显示完整的黑色边框
        # SHAP 默认会隐藏 top 和 right 边框，这里必须手动把它们找回来
        ax = plt.gca()
        for spine in ['top', 'bottom', 'left', 'right']:
            ax.spines[spine].set_visible(True)  # 设为可见
            ax.spines[spine].set_color('black')  # 设为黑色
            ax.spines[spine].set_linewidth(1.5)  # 设为1.5线宽，加粗一点确边框清晰

        # 确保刻度线存在
        ax.tick_params(axis='both', which='both', bottom=True, left=True, color='black')

        # 设置字体和标题
        plt.title('SHAP Summary Plot', fontsize=18, fontweight='bold', fontfamily='Arial')
        plt.xticks(fontsize=14, fontfamily='Arial')
        plt.yticks(fontsize=14, fontfamily='Arial')

        # 调整横轴标签字体
        plt.xlabel('SHAP value (impact on model output)', fontsize=14, fontfamily='Arial')

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