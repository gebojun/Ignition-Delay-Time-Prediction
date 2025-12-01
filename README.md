# Ignition Delay Time Prediction (点火延迟时间预测)

这是一个基于机器学习的框架，旨在预测燃料的点火延迟时间。该项目集成了多种主流的机器学习算法，提供了从数据预处理、超参数优化、模型训练、交叉验证到结果可视化的完整工作流。此外，还包含相关性分析和 SHAP 模型解释性分析工具。

---

## 🌟 项目特性

* **多模型支持**：集成了 8 种回归模型，涵盖线性模型、树模型和深度学习模型。
* **自动超参数优化**：基于网格搜索 (Grid Search) 自动寻找最优模型参数。
* **全面评估**：提供 R²、MAE、RMSE 等评估指标，并包含交叉验证 (Cross Validation) 结果。
* **可视化分析**：自动生成预测对比图、特征重要性图、参数优化热力图等。
* **模型解释性**：支持 Pearson 相关性分析和 SHAP (SHapley Additive exPlanations) 分析。

---

## 🛠️ 支持的模型

可以通过命令行参数 `--model` 选择以下模型：

| 参数代码 | 模型全称 | 说明 |
| :--- | :--- | :--- |
| `xgb` | **XGBoost** | 高效的梯度提升决策树 (默认模型) |
| `catboost` | **CatBoost** | 擅长处理类别特征的梯度提升库 |
| `lightgbm` | **LightGBM** | 轻量级、高效的梯度提升机 |
| `rf` | **Random Forest** | 随机森林回归 |
| `mlr` | **MLR** | 多元线性回归 (基准模型) |
| `svm` | **SVM** | 支持向量机回归 |
| `tabpfn` | **TabPFN** | 针对表格数据的预训练 Transformer 模型 |
| `ann` | **ANN** | 人工神经网络 (MLP) |

---

## 📂 目录结构

```text
├── analysis/           # 数据分析脚本 (Correlation, SHAP)
├── config/             # 配置文件 (模型参数, 路径配置)
├── data/               # 数据集存放目录 (需放置 dataset.csv)
├── models/             # 各机器学习模型的封装类
├── result/             # 输出目录 (自动生成的图片、Excel 结果)
├── utils/              # 工具模块 (数据加载, 预处理, 可视化)
├── main.py             # 程序主入口
└── README.md           # 项目说明文档
