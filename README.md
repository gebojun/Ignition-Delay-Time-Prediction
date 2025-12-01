Ignition Delay Time Prediction (点火延迟时间预测)
这是一个基于机器学习的框架，旨在预测燃料的点火延迟时间。该项目集成了多种主流的机器学习算法，提供了从数据预处理、超参数优化、模型训练、交叉验证到结果可视化的完整工作流。此外，还包含相关性分析和 SHAP 模型解释性分析工具。
🌟 项目特性
多模型支持：集成了 8 种回归模型，涵盖线性模型、树模型和深度学习模型。
自动超参数优化：基于网格搜索 (Grid Search) 自动寻找最优模型参数。
全面评估：提供 R²、MAE、RMSE 等评估指标，并包含交叉验证 (Cross Validation) 结果。
可视化分析：自动生成预测对比图、特征重要性图、参数优化热力图等。
模型解释性：支持 Pearson 相关性分析和 SHAP (SHapley Additive exPlanations) 分析，帮助理解特征对预测结果的影响。

🛠️ 支持的模型
可以通过命令行参数 --model 选择以下模型：

xgb：XGBoost,高效的梯度提升决策树 (默认模型)
catboost：CatBoost,擅长处理类别特征的梯度提升库
lightgbm：LightGBM,轻量级梯度提升机
rf：Random Forest,随机森林回归
mlr：Multiple Linear Regression,多元线性回归 (作为基准模型)
svm：Support Vector Machine,支持向量机回归 (SVM)
tabpfn:TabPFN,针对表格数据的预训练 Transformer 模型
ann:Artificial Neural Network,人工神经网络 (MLP)

📂 目录结构
├── analysis/           # 数据分析脚本 (Correlation, SHAP)
├── config/             # 配置文件 (模型参数, 路径配置)
├── data/               # 数据集存放目录 (默认读取 dataset.csv)
├── models/             # 各机器学习模型的封装类
├── result/             # 输出目录 (生成的图片、Excel 结果)
├── utils/              # 工具模块 (数据加载, 预处理, 可视化)
├── main.py             # 程序主入口
└── README.md           # 项目说明文档

🚀 准备工作
1. 环境准备
请确保您的 Python 环境已安装以下依赖库（建议 Python 3.8+）：
pip install pandas numpy scikit-learn matplotlib seaborn xgboost catboost lightgbm shap tabpfn openpyxl

3. 准备数据
请将您的数据集命名为 dataset.csv 并放置在 data/ 目录下。
格式：CSV 文件。
结构：第一列应为目标变量（点火延迟时间），随后的列为特征变量。

3. 运行模型
使用 main.py 运行特定的模型或分析任务。
# 运行指定模型 (例如：随机森林)
python main.py --model rf

# 仅运行相关性分析
python main.py --analysis correlation

#仅运行 SHAP 解释性分析
python main.py --analysis shap

# 运行所有分析任务 (所有模型+相关性+SHAP)
python main.py --analysis all

4. 查看结果
程序运行结束后，结果将保存在 result/ 目录下对应的子文件夹中（例如 result/XGBoost/）。输出文件包括：
Excel 报告：
*_cross_validation.xlsx: 交叉验证详细结果。
*_optimization_results.xlsx: 超参数寻优过程记录。
*_feature_importance.xlsx: 特征重要性数值。

可视化图表：
*_final_model.png: 验证集与测试集的预测值 vs 真实值对比图。
*_feature_importance.png: 特征重要性条形图。
*_parameter_optimization.png: 超参数优化过程的可视化。

⚙️ 配置说明
您可以修改 config/models_config.py 来调整模型的超参数搜索空间或固定参数。
BaseModelConfig: 定义通用的数据划分比例（测试集 20%，验证集 25%）和随机种子。
各模型 Config 类（如 XGBConfig）: 定义了 param_grid（网格搜索范围）和 fixed_params（固定参数）。

