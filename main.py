import os
import argparse
import pandas as pd
from config.paths import PathConfig
from config.models_config import XGBConfig, CatBoostConfig, LightGBMConfig, RFConfig, MLRConfig, SVMConfig, \
    TabPFNConfig, ANNConfig
from utils.data_loader import DataLoader
from utils.preprocessing import Preprocessor
from utils.visualization import Visualizer
from models.xgb_model import XGBModel
from models.catboost_model import CatBoostModel
from models.lightgbm_model import LightGBMModel
from models.rf_model import RFModel
from models.mlr_model import MLRModel
from models.svm_model import SVMModel
from models.tabpfn_model import TabPFNModel
from models.ann_model import ANNModel
from analysis.correlation import CorrelationAnalyzer
from analysis.shap_analysis import SHAPAnalyzer


def main():
    """主函数"""
    parser = argparse.ArgumentParser("Run machine learning models")
    parser.add_argument('--model', type=str, default='xgb',
                        help="specify the model type from: xgb, catboost, lightgbm, rf, mlr, svm, tabpfn, ann (default: xgb)")
    parser.add_argument('--analysis', type=str, default='none',
                        help="specify the analysis type: none, correlation, shap, all (default: none)")

    args = parser.parse_args()

    # 运行指定的分析
    if args.analysis in ['correlation', 'all']:
        print("开始相关性分析...")
        correlation_output_dir = r"C:\Users\Administrator\Desktop\Analysis"
        os.makedirs(correlation_output_dir, exist_ok=True)
        correlation_analyzer = CorrelationAnalyzer(
            data_path=r"C:\Users\Administrator\Desktop\train.xlsx",
            output_dir=correlation_output_dir
        )
        correlation_analyzer.analyze()

    if args.analysis in ['shap', 'all']:
        print("开始SHAP分析...")
        shap_analyzer = SHAPAnalyzer(
            data_path=r"C:\Users\Administrator\Desktop\trainshap.xlsx",
            output_dir=correlation_output_dir
        )
        shap_analyzer.analyze()

    # 运行指定的模型
    models_map = {
        'xgb': (XGBModel, XGBConfig, 'XGBoost'),
        'catboost': (CatBoostModel, CatBoostConfig, 'CatBoost'),
        'lightgbm': (LightGBMModel, LightGBMConfig, 'LightGBM'),
        'rf': (RFModel, RFConfig, 'RandomForest'),
        'mlr': (MLRModel, MLRConfig, 'MLR'),
        'svm': (SVMModel, SVMConfig, 'SVM'),
        'tabpfn': (TabPFNModel, TabPFNConfig, 'TabPFN'),
        'ann': (ANNModel, ANNConfig, 'ANN')
    }

    if args.model in models_map:
        model_class, config_class, model_name = models_map[args.model]
        run_model(model_name, model_class, config_class)
    else:
        print(f"未知模型: {args.model}")
        print("可用模型: xgb, catboost, lightgbm, rf, mlr, svm, tabpfn, ann")


def run_model(model_name, model_class, config_class):
    """运行单个模型"""
    print(f"\n{'=' * 50}")
    print(f"开始训练 {model_name} 模型")
    print(f"{'=' * 50}")

    # 初始化配置
    path_config = PathConfig(model_name)
    model_config = config_class()
    output_dir = path_config.create_dirs()

    # 初始化组件
    data_loader = DataLoader(path_config)
    preprocessor = Preprocessor()
    visualizer = Visualizer(output_dir)

    # 加载数据
    data = data_loader.load_data()
    X = data.iloc[:, 1:]  # 特征数据
    y = data.iloc[:, 0]  # 目标变量

    # 划分数据
    X_train, X_val, X_test, y_train, y_val, y_test = data_loader.split_data(X, y)

    # 数据预处理（只对特征进行标准化）
    X_train_scaled, X_val_scaled, X_test_scaled = preprocessor.fit_transform(X_train, X_val, X_test)

    # 初始化模型
    model = model_class(model_config)

    # 超参数优化（对于MLR和TabPFN，这是一个空操作）
    grid_search, optimization_results = model.hyperparameter_tuning(
        X_train_scaled, X_val_scaled, y_train, y_val
    )

    # 保存优化结果
    if not optimization_results.empty:
        optimization_path = os.path.join(output_dir, f"{model_name.lower()}_optimization_results.xlsx")
        optimization_results.to_excel(optimization_path, index=False)
        print(f"参数优化结果已保存到: {optimization_path}")

    # 绘制参数优化图（对于MLR和TabPFN，跳过）
    if model_name not in ['MLR', 'TabPFN'] and len(model_config.param_grid) > 0:
        visualizer.plot_parameter_optimization(
            grid_search.cv_results_, model_config.param_grid,
            grid_search.best_params_, f"{model_name.lower()}_parameter_optimization.png",
            model_config.colors
        )

    # 使用完整训练数据训练最终模型
    X_final_train, y_final_train = preprocessor.combine_train_val(
        X_train_scaled, X_val_scaled, y_train, y_val
    )
    model.train_final_model(X_final_train, y_final_train)

    # 模型评估
    val_predictions, test_predictions, val_metrics, test_metrics = model.evaluate(
        X_val_scaled, y_val, X_test_scaled, y_test
    )

    # 绘制预测结果
    visualizer.plot_prediction_comparison(
        y_val, val_predictions, y_test, test_predictions,
        val_metrics, test_metrics, f"{model_name.lower()}_final_model.png",
        model_config.colors, model_name
    )

    # 交叉验证
    cv_summary, mean_values, std_values = model.evaluator.cross_validate_model(
        model.model, X_final_train, y_final_train
    )

    # 保存交叉验证结果
    cv_path = os.path.join(output_dir, f"{model_name.lower()}_cross_validation.xlsx")
    cv_summary.to_excel(cv_path, index=False)
    print(f"交叉验证结果已保存到: {cv_path}")

    # 特征重要性分析
    feature_importance = model.get_feature_importance()
    feature_names = data.columns[1:]  # 获取特征名称

    if len(feature_importance) > 0:
        importance_df = visualizer.plot_feature_importance(
            feature_importance, feature_names, f"{model_name.lower()}_feature_importance.png",
            model_config.colors['feature_importance'], model_name
        )

        # 保存特征重要性
        importance_path = os.path.join(output_dir, f"{model_name.lower()}_feature_importance.xlsx")
        importance_df.to_excel(importance_path, index=False)
        print(f"特征重要性结果已保存到: {importance_path}")
    else:
        print(f"注意: {model_name} 模型没有可用的特征重要性信息")

    # 对于MLR，额外保存系数
    if model_name == 'MLR' and hasattr(model, 'get_model_coefficients'):
        coefficients = model.get_model_coefficients(feature_names)
        coefficients_df = pd.DataFrame(list(coefficients.items()), columns=['Parameter', 'Value'])
        coefficients_path = os.path.join(output_dir, "mlr_coefficients.xlsx")
        coefficients_df.to_excel(coefficients_path, index=False)
        print(f"模型系数已保存到: {coefficients_path}")

        # 打印系数
        print(f"\n{model_name} 模型系数:")
        for param, value in coefficients.items():
            print(f"  {param}: {value:.5f}")

    # 预测值分析
    visualizer.plot_prediction_analysis(y_val, val_predictions, y_test, test_predictions, model_name)

    # 模型诊断
    model.evaluator.diagnose_model(val_metrics['r2'], test_metrics['r2'], model_name)

    # 输出最终结果
    print_results(val_metrics, test_metrics, mean_values, std_values, model_name)


def print_results(val_metrics, test_metrics, mean_values, std_values, model_name):
    """打印最终结果"""
    print(f"\n=== {model_name} 最终结果 ===")
    print("\n验证集指标:")
    print(f"R²: {val_metrics['r2']:.5f}")
    print(f"MAE: {val_metrics['mae']:.5f}")
    print(f"RMSE: {val_metrics['rmse']:.5f}")

    print("\n测试集指标:")
    print(f"R²: {test_metrics['r2']:.5f}")
    print(f"MAE: {test_metrics['mae']:.5f}")
    print(f"RMSE: {test_metrics['rmse']:.5f}")

    print("\n交叉验证平均结果:")
    print(f"测试集平均 R²: {mean_values['Test_R2']:.5f} (±{std_values['Test_R2']:.5f})")
    print(f"测试集平均 MAE: {mean_values['Test_MAE']:.5f} (±{std_values['Test_MAE']:.5f})")
    print(f"测试集平均 RMSE: {mean_values['Test_RMSE']:.5f} (±{std_values['Test_RMSE']:.5f})")


if __name__ == "__main__":
    main()