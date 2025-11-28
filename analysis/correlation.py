import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os


class CorrelationAnalyzer:
    def __init__(self, data_path, output_dir):
        self.data_path = data_path
        self.output_dir = output_dir

    def analyze(self):
        """执行相关性分析"""
        # 数据加载
        data = pd.read_excel(self.data_path, header=0)

        # 计算Pearson相关系数矩阵
        corr_matrix = data.corr(method='pearson')

        plt.figure(figsize=(16, 14), dpi=150)
        plt.rcParams['font.family'] = 'Arial'
        plt.rcParams['axes.edgecolor'] = 'navy'

        cmap = sns.diverging_palette(220, 10, as_cmap=True)

        heatmap = sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            cmap=cmap,
            center=0,
            square=True,
            linewidths=0.5,
            linecolor='white',
            cbar_kws={"shrink": 0.8, "label": "Correlation Coefficient"},
            annot_kws={
                "size": 22,
                "weight": "bold",
                "color": "black"
            }
        )

        plt.title('Pearson Correlation Matrix',
                  fontsize=21, pad=20, weight='bold')
        plt.xlabel('Features', fontsize=22)
        plt.ylabel('Features', fontsize=22)

        plt.xticks(rotation=45, ha='right', fontsize=21)
        plt.yticks(rotation=0, fontsize=21)

        heatmap.grid(False)
        for spine in heatmap.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(2)
            spine.set_color('navy')

        # 调整色带样式
        cbar = heatmap.collections[0].colorbar
        cbar.ax.tick_params(labelsize=21)
        cbar.ax.set_ylabel('Correlation Coefficient',
                           rotation=270,
                           labelpad=25,
                           fontsize=21,
                           weight='bold')

        # 优化布局并保存
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, "pearson_correlation_matrix.png")
        plt.savefig(output_path,
                    dpi=300,
                    bbox_inches='tight',
                    facecolor='white')
        plt.show()

        print(f"相关性分析图已保存到: {output_path}")

        return corr_matrix