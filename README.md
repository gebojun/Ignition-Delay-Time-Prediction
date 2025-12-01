这是一个为你生成的专业 README.md 文件，你可以直接复制粘贴到你的项目根目录下。这份文档是基于你提供的代码（main.py、models_config.py 等）自动生成的，涵盖了项目介绍、安装、使用方法、支持的模型以及目录结构等核心信息。Markdown# Ignition Delay Time Prediction (点火延迟时间预测)

这是一个基于 Python 的机器学习框架，专为预测燃料的点火延迟时间而设计。项目集成了多种主流的回归算法，提供从数据预处理、模型训练、超参数自动优化到结果可视化的完整工作流。此外，还内置了 SHAP 可解释性分析和相关性分析工具，帮助用户深入理解数据特征。

## 🌟 项目特性

* **多模型集成**：内置 XGBoost, LightGBM, CatBoost, 随机森林 (RF), SVM, ANN, TabPFN 以及多元线性回归 (MLR) 等 8 种模型。
* **自动化流程**：一键完成数据清洗、标准化、数据集划分（训练/验证/测试）。
* **超参数优化**：基于网格搜索 (Grid Search) 自动寻找最优超参数组合。
* **全面评估**：输出 R²、MAE、RMSE 等关键指标，并支持交叉验证 (Cross Validation)。
* **可视化报告**：自动生成预测散点图、特征重要性排序图、超参数优化热力图等。
* **可解释性分析**：集成 Pearson 相关性分析与 SHAP (SHapley Additive exPlanations) 模型归因分析。

## 🛠️ 环境依赖

请确保你的 Python 环境（建议 Python 3.8+）已安装以下依赖库：

```bash
pip install pandas numpy scikit-learn matplotlib seaborn xgboost catboost lightgbm shap tabpfn openpyxl
📂 目录结构Plaintext.
├── analysis/           # 数据分析模块 (Correlation, SHAP)
├── config/             # 配置文件 (模型参数, 路径配置)
├── data/               # 数据集存放目录
├── models/             # 模型封装类 (XGB, RF, ANN, etc.)
├── result/             # 结果输出目录 (自动生成)
├── utils/              # 工具模块 (数据加载, 预处理, 可视化)
├── main.py             # 程序主入口
└── README.md           # 项目说明文档
🚀 快速开始1. 准备数据请将你的数据集命名为 dataset.csv 并放置在 data/ 目录下。文件格式：CSV数据格式：第一列必须为目标变量（即点火延迟时间），后续列为特征变量。2. 运行模型你可以通过命令行参数 --model 指定要训练的模型，默认使用 xgb。基本用法：Bash# 运行 XGBoost 模型 (默认)
python main.py --model xgb

# 运行 随机森林 (Random Forest)
python main.py --model rf

# 运行 神经网络 (ANN)
python main.py --model ann
支持的模型代码 (--model)：代码模型全称说明xgbXGBoost高效的梯度提升决策树catboostCatBoost擅长处理类别特征的梯度提升库lightgbmLightGBM轻量级、高效的梯度提升机rfRandom Forest随机森林回归mlrMultiple Linear Regression多元线性回归 (基准模型)svmSupport Vector Machine支持向量机回归tabpfnTabPFN针对表格数据的预训练 TransformerannArtificial Neural Network人工神经网络 (MLP)3. 运行分析工具你可以通过 --analysis 参数单独运行数据分析任务，或结合模型训练一起运行。Bash# 仅运行相关性分析 (生成热力图)
python main.py --analysis correlation

# 仅运行 SHAP 可解释性分析
python main.py --analysis shap

# 运行所有分析任务 (相关性 + SHAP)
python main.py --analysis all
📊 输出结果程序运行完成后，结果将保存在 result/ 目录下对应的子文件夹中（例如 result/XGBoost/）。可视化图表：*_final_model.png: 预测值 vs 真实值对比图（验证集与测试集）。*_feature_importance.png: 特征重要性条形图。*_parameter_optimization.png: 超参数寻优过程可视化。数据报告 (Excel)：*_cross_validation.xlsx: 5折交叉验证详细结果。*_optimization_results.xlsx: 网格搜索过程中的参数评估记录。*_feature_importance.xlsx: 具体的特征重要性数值。⚙️ 配置说明所有模型的超参数配置均位于 config/models_config.py 文件中。修改参数范围：在各模型的 Config 类（如 XGBConfig）中修改 self.param_grid 即可调整网格搜索的范围。固定参数：在 self.fixed_params 中设置不需要搜索的固定参数（如随机种子、线程数等）。基础配置：在 BaseModelConfig 中可以调整测试集比例 (test_size) 和验证集比例 (val_size)。Python# 示例：修改 XGBoost 的搜索空间 (config/models_config.py)
class XGBConfig(BaseModelConfig):
    def __init__(self):
        super().__init__()
        self.param_grid = {
            'n_estimators': [100, 200, 300],  # 修改树的数量
            'max_depth': [3, 5, 7],           # 修改树深度
            # ...
        }
Created by [Your Name/Team Name]
