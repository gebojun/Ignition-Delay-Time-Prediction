import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np


class Visualizer:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        # [修改1] 强制设置全局字体为 Arial
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Arial']
        plt.rcParams['axes.unicode_minus'] = False

    def _set_common_style(self, ax):
        """设置通用的样式：黑色边框、网格等"""
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(1)
        ax.grid(True, alpha=0.3, linestyle='--')

    def plot_parameter_optimization(self, results, param_grid, best_params, filename, colors):
        """绘制参数优化图"""
        print("绘制参数优化图...")

        results_df = pd.DataFrame(results['params'])
        results_df['score'] = results['mean_test_score']

        param_names = list(param_grid.keys())
        n_rows = 2
        n_cols = 2

        # 保持尺寸为 (10, 8)
        plt.figure(figsize=(10, 8))

        for idx, param_name in enumerate(param_names):
            if idx >= n_rows * n_cols: break

            ax = plt.subplot(n_rows, n_cols, idx + 1)

            param_values = param_grid[param_name]
            param_scores = []

            for param_val in param_values:
                mask = (results_df[param_name] == param_val)
                for p_name, p_val in best_params.items():
                    if p_name != param_name:
                        mask &= (results_df[p_name] == p_val)

                matching_scores = results_df[mask]['score']
                if not matching_scores.empty:
                    param_scores.append(matching_scores.mean())
                else:
                    fallback_mask = (results_df[param_name] == param_val)
                    fallback_scores = results_df[fallback_mask]['score']
                    param_scores.append(fallback_scores.max() if not fallback_scores.empty else np.nan)

            if len(param_values) == len(param_scores):
                param_labels = [str(val) for val in param_values]
                x_pos = range(len(param_values))

                # 此处保持使用单色，因为是曲线图
                plt.plot(x_pos, param_scores, 'o-', linewidth=2, markersize=8, color=colors['primary'])
                plt.xlabel(param_name, fontsize=12)

                rotation = 45 if len(param_labels) > 10 else 0
                plt.xticks(x_pos, param_labels, rotation=rotation)

                try:
                    best_val = best_params[param_name]
                    best_idx = param_labels.index(str(best_val))
                    plt.axvline(x=best_idx, color=colors['secondary'], linestyle='--', alpha=0.7,
                                label=f'Best: {best_val}')
                except ValueError:
                    plt.axvline(x=0, color=colors['secondary'], linestyle='--', alpha=0.7,
                                label=f'Best: {best_params[param_name]}')

                plt.legend()
                plt.ylabel('R² Score', fontsize=12)
                plt.title(f'R² Score vs {param_name}', fontsize=14, fontweight='bold')

                self._set_common_style(ax)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()

    def plot_prediction_comparison(self, y_true_val, y_pred_val, y_true_test, y_pred_test,
                                   val_metrics, test_metrics, filename, colors, model_name):
        """绘制预测结果对比图"""
        plt.figure(figsize=(14, 6), dpi=300)

        # 验证集子图
        ax1 = plt.subplot(1, 2, 1)
        # 注意：这里不再传递单一 color，而是由 _plot_scatter 内部使用 coolwarm
        self._plot_scatter(ax1, y_true_val, y_pred_val, val_metrics, f'Validation Set - {model_name}')

        # 测试集子图
        ax2 = plt.subplot(1, 2, 2)
        self._plot_scatter(ax2, y_true_test, y_pred_test, test_metrics, f'Test Set - {model_name}')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_scatter(self, ax, y_true, y_pred, metrics, title):
        # 恢复 Ideal Fit 为灰色粗线
        ax.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)],
                color='#9D9EA3', linewidth=3, label='Ideal Fit')

        # 散点使用 coolwarm 映射 (c=y_true)
        ax.scatter(y_true, y_pred, alpha=0.7, c=y_true, cmap='coolwarm', s=80,
                   edgecolors='black', linewidth=0.5, label='Predicted vs. True')

        # ... (其余代码)

        ax.text(
            0.05, 0.95,
            f'R² = {metrics["r2"]:.5f}\nMAE = {metrics["mae"]:.5f}\nRMSE = {metrics["rmse"]:.5f}',
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black')
        )

        ax.set_xlabel('True Values (us)', fontsize=16)
        ax.set_ylabel('Predicted Values (us)', fontsize=16)
        ax.legend(fontsize=12, loc='lower right')
        ax.tick_params(axis='both', labelsize=14)
        ax.set_title(title, fontsize=16, fontweight='bold')

        self._set_common_style(ax)

    def plot_feature_importance(self, feature_importance, feature_names, filename, color, model_name):
        """绘制特征重要性图"""
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importance
        }).sort_values('Importance', ascending=False)

        # 保持较大的画布尺寸
        plt.figure(figsize=(14, 8))

        bars = plt.barh(importance_df['Feature'], importance_df['Importance'], color=color)

        # 1. 动态调整X轴范围
        max_importance = importance_df['Importance'].max()
        plt.xlim(0, max_importance * 1.25)

        # 2. 绘制数值标签
        for bar, importance in zip(bars, importance_df['Importance']):
            width = bar.get_width()
            plt.text(width + max_importance * 0.01,
                     bar.get_y() + bar.get_height() / 2,
                     f'{importance:.4f}',
                     ha='left',
                     va='center',
                     fontsize=20,  # 大字号
                     fontweight='bold')  # 加粗数值

        # 3. 设置标签和标题
        plt.xlabel('Feature Importance', fontsize=20, fontweight='bold', labelpad=15)
        plt.title(f'{model_name} Feature Importance', fontsize=22, fontweight='bold', pad=20)

        # 4. 设置刻度标签 (X轴数值 和 Y轴特征名)
        plt.tick_params(axis='both', which='major', labelsize=16, width=2.5, length=6)  # width设置刻度线本身的粗细
        plt.xticks(fontweight='bold')
        plt.yticks(fontweight='bold')

        # === 5. 新增：加粗坐标轴边框线 (Spines) ===
        ax = plt.gca()  # 获取当前坐标轴对象
        for spine in ax.spines.values():
            spine.set_linewidth(2.5)  # 设置边框线宽，2.5 看起来比较粗壮
            spine.set_color('black')  # 确保是黑色

        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.show()

        return importance_df

    def plot_prediction_analysis(self, y_val, val_predictions, y_test, test_predictions, model_name):
        # 保持不变
        pass

    def plot_prediction_analysis(self, y_val, val_predictions, y_test, test_predictions, model_name):
        """预测值分析"""
        print(f"\n=== {model_name} 预测值分析 ===")
        print(f"验证集真实值范围: {y_val.min():.2f} ~ {y_val.max():.2f}")
        print(f"验证集预测值范围: {val_predictions.min():.2f} ~ {val_predictions.max():.2f}")
        print(f"测试集真实值范围: {y_test.min():.2f} ~ {y_test.max():.2f}")
        print(f"测试集预测值范围: {test_predictions.min():.2f} ~ {test_predictions.max():.2f}")

        # 检查预测值是否集中在某个区间
        unique_predictions = np.unique(test_predictions.round(2))
        print(f"测试集唯一预测值数量: {len(unique_predictions)}")
        print(f"前10个唯一预测值: {unique_predictions[:10]}")