class BaseModelConfig:
    def __init__(self):
        # 数据划分配置
        self.test_size = 0.2
        self.val_size = 0.25
        self.random_state = 41
        self.cv_folds = 5

        # 评分指标
        self.scoring_metrics = {
            'r2': 'r2',
            'neg_mae': 'neg_mean_absolute_error',
            'neg_rmse': 'neg_root_mean_squared_error'
        }


class XGBConfig(BaseModelConfig):
    def __init__(self):
        super().__init__()
        # 超参数网格
        self.param_grid = {
            'n_estimators': [5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 200],
            'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 18, 20],
            'learning_rate': [0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1],
            'subsample': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        }

        # 固定参数
        self.fixed_params = {
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1
        }

        # 可视化配置
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#2ca02c',
            'feature_importance': '#2ca02c'
        }


class CatBoostConfig(BaseModelConfig):
    def __init__(self):
        super().__init__()
        # 超参数网格
        self.param_grid = {
            'iterations': [5, 10, 15, 20, 40, 50, 60, 80, 100, 150, 200, 300, 500, 600, 700, 800, 900],
            'depth': [4, 6, 8, 10, 12, 14, 16],
            'learning_rate': [0.01, 0.02, 0.03, 0.1, 0.2, 0.3, 0.4, 0.5],
            'l2_leaf_reg': [1, 3, 5, 7, 9, 12, 15, 18, 20, 30]
        }

        # 固定参数
        self.fixed_params = {
            'random_seed': 42,
            'verbose': False,
            'thread_count': -1
        }

        # 可视化配置
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#2ca02c',
            'feature_importance': '#9467bd'
        }


class LightGBMConfig(BaseModelConfig):
    def __init__(self):
        super().__init__()
        # 超参数网格
        self.param_grid = {
            'n_estimators': [5, 10, 15, 20, 40, 60, 80, 100, 200, 300, 500],
            'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20],
            'learning_rate': [0.01, 0.02, 0.03, 0.1, 0.4, 0.5, 0.6, 0.7, 1, 1.5, 2],
            'num_leaves': [5, 10, 12, 15, 18, 20, 22, 25]
        }

        # 固定参数
        self.fixed_params = {
            'subsample': 0.1,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }

        # 可视化配置
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#2ca02c',
            'feature_importance': '#9467bd'
        }


class RFConfig(BaseModelConfig):
    def __init__(self):
        super().__init__()
        # 超参数网格
        self.param_grid = {
            'n_estimators': [5, 10, 15, 20, 25, 50, 75, 100, 150, 200, 250, 300],
            'max_depth': [1, 3, 5, 10, 20, 30, 50, 100],
            'min_samples_split': [3, 5, 8, 10, 15],
            'min_samples_leaf': [1, 2, 3, 4, 5]
        }

        # 固定参数
        self.fixed_params = {
            'max_features': 'log2',
            'random_state': 42,
            'n_jobs': -1
        }

        # 可视化配置
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#2ca02c',
            'feature_importance': '#1f77b4'
        }


class MLRConfig(BaseModelConfig):
    def __init__(self):
        super().__init__()
        # MLR没有超参数需要优化，所以param_grid为空
        self.param_grid = {}

        # 固定参数
        self.fixed_params = {}

        # 可视化配置
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#2ca02c',
            'feature_importance': '#ff7f0e'
        }

        # MLR特有配置
        self.scale_target = True


class SVMConfig(BaseModelConfig):
    def __init__(self):
        super().__init__()
        # SVM参数网格
        self.param_grid = {
            'C': [100, 500, 1000, 2000, 3500, 4000, 4500, 4600, 4800, 4900, 5000, 5100, 5200, 5300, 5500, 6000],
            'gamma': [0.001, 0.01, 0.1, 1, 5, 8, 9, 10, 11, 12, 15, 18],
            'kernel': ['rbf'],
            'epsilon': [0.01, 0.1, 0.15, 0.18, 0.2, 0.22, 0.25, 0.3, 0.35, 0.4, 0.5, 1, 2, 3, 4, 10, 20, 30, 40, 50, 60,
                        70, 80, 90, 100, 120, 150, 180, 200, 250, 300]
        }

        # 固定参数
        self.fixed_params = {
            'random_state': 42
        }

        # 可视化配置
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#2ca02c',
            'feature_importance': '#d62728'
        }


class TabPFNConfig(BaseModelConfig):
    def __init__(self):
        super().__init__()
        # TabPFN没有超参数需要优化
        self.param_grid = {}

        # 固定参数
        self.fixed_params = {}

        # 可视化配置
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#2ca02c',
            'feature_importance': '#8c564b'
        }

        # TabPFN特有配置
        self.device = 'cuda'


class ANNConfig(BaseModelConfig):
    def __init__(self):
        super().__init__()
        # ANN参数网格
        self.param_grid = {
            'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50), (100, 100)],
            'activation': ['relu', 'tanh'],
            'alpha': [0.0001, 0.001, 0.01, 0.1],
            'learning_rate_init': [0.001, 0.01, 0.1],
            'max_iter': [500, 1000, 2000]
        }

        # 固定参数
        self.fixed_params = {
            'random_state': 42,
            'early_stopping': True,
            'n_iter_no_change': 50,
            'verbose': False
        }

        # 可视化配置
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#2ca02c',
            'feature_importance': '#e377c2'
        }