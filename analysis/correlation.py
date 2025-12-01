import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os


class CorrelationAnalyzer:
    def __init__(self, data_path, output_dir):
        self.data_path = data_path
        self.output_dir = output_dir
        # 修改：设置全局字体
        plt.rcParams['font.family'] = 'Arial'

    def analyze(self):
        # ... (读取数据保持不变)
        data = pd.read_csv(self.data_path, header=0)
        corr_matrix = data.corr(method='pearson')

        # 修改：尺寸可以保持较大，或调整为统一比例，这里保持原样以适应热力图
        plt.figure(figsize=(12, 10), dpi=300)

        # 字体设置已经在 __init__ 中处理，这里移除单独设置，或保留
        # plt.rcParams['font.family'] = 'Arial'

        # 修改：使用 coolwarm 色系
        cmap = 'coolwarm'

        heatmap = sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            cmap=cmap,
            center=0,
            square=True,
            linewidths=0.5,
            linecolor='black',  # 修改：网格线颜色改为黑色以增加边框感
            cbar_kws={"shrink": 0.8, "label": "Correlation Coefficient"},
            annot_kws={
                "size": 16,  # 显著增大字体大小（原为10）
                "weight": "bold",  # 粗体显示
                "color": "black"  # 确保黑色字体
            }
        )

        plt.title('Pearson Correlation Matrix', fontsize=16, pad=20, weight='bold')
        plt.xlabel('Features', fontsize=14)
        plt.ylabel('Features', fontsize=14)

        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.yticks(rotation=0, fontsize=12)

        # 修改：添加外部边框
        for spine in heatmap.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1)
            spine.set_color('black')

        plt.tight_layout()
        output_path = os.path.join(self.output_dir, "pearson_correlation_matrix.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()  # 如果在服务器运行，可能需要移除这行

        # ... (return 保持不变)
        return corr_matrix