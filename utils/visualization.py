import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np


class Visualizer:
    def __init__(self, output_dir):
        self.output_dir = output_dir

    def plot_parameter_optimization(self, results, param_grid, best_params, filename, colors):
        """绘制参数优化图"""
        print("绘制参数优化图...")

        results_df = pd.DataFrame(results['params'])
        results_df['score'] = results['mean_test_score']

        param_names = list(param_grid.keys())
        n_rows = 2
        n_cols = 2

        plt.figure(figsize=(15, 12))

        for idx, param_name in enumerate(param_names):
            plt.subplot(n_rows, n_cols, idx + 1)

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

                plt.plot(x_pos, param_scores, 'o-', linewidth=2, markersize=8, color=colors['primary'])
                plt.xlabel(param_name, fontsize=12)

                # 根据参数名调整标签旋转角度
                rotation = 45 if len(param_labels) > 10 else 0
                plt.xticks(x_pos, param_labels, rotation=rotation)

                try:
                    best_val = best_params[param_name]
                    best_idx = param_labels.index(str(best_val))
                    plt.axvline(x=best_idx, color='red', linestyle='--', alpha=0.7, label=f'Best: {best_val}')
                except ValueError:
                    plt.axvline(x=0, color='red', linestyle='--', alpha=0.7, label=f'Best: {best_params[param_name]}')

                plt.legend()
                plt.ylabel('R² Score', fontsize=12)
                plt.title(f'R² Score vs {param_name}', fontsize=14, fontweight='bold')
                plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.show()

    def plot_prediction_comparison(self, y_true_val, y_pred_val, y_true_test, y_pred_test,
                                   val_metrics, test_metrics, filename, colors, model_name):
        """绘制预测结果对比图"""
        plt.figure(figsize=(14, 6), dpi=300)

        # 验证集子图
        plt.subplot(1, 2, 1)
        self._plot_scatter(y_true_val, y_pred_val, val_metrics, colors['primary'], f'Validation Set - {model_name}')

        # 测试集子图
        plt.subplot(1, 2, 2)
        self._plot_scatter(y_true_test, y_pred_test, test_metrics, colors['secondary'], f'Test Set - {model_name}')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=500, bbox_inches='tight')
        plt.show()

    def _plot_scatter(self, y_true, y_pred, metrics, color, title):
        """绘制单个散点图"""
        plt.scatter(y_true, y_pred, alpha=0.6, color=color, s=80, label='Predicted vs. True')
        plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)],
                 color='#9D9EA3', linewidth=3, label='Ideal Fit')

        plt.gca().text(
            0.05, 0.95,
            f'R² = {metrics["r2"]:.5f}\nMAE = {metrics["mae"]:.5f}\nRMSE = {metrics["rmse"]:.5f}',
            transform=plt.gca().transAxes,
            fontsize=12,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )

        plt.xlabel('True Values (us)', fontsize=16)
        plt.ylabel('Predicted Values (us)', fontsize=16)
        plt.legend(fontsize=12, loc='lower right')
        plt.tick_params(axis='both', labelsize=14)
        plt.title(title, fontsize=16, fontweight='bold')

        for spine in plt.gca().spines.values():
            spine.set_linewidth(2)

    def plot_feature_importance(self, feature_importance, feature_names, filename, color, model_name):
        """绘制特征重要性图"""
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importance
        }).sort_values('Importance', ascending=False)

        plt.figure(figsize=(10, 6))
        bars = plt.barh(importance_df['Feature'], importance_df['Importance'], color=color)

        for bar, importance in zip(bars, importance_df['Importance']):
            width = bar.get_width()
            plt.text(width + 0.001, bar.get_y() + bar.get_height() / 2,
                     f'{importance:.4f}', ha='left', va='center', fontsize=10)

        plt.xlabel('Feature Importance', fontsize=14)
        plt.title(f'{model_name} Feature Importance', fontsize=16, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.show()

        return importance_df

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