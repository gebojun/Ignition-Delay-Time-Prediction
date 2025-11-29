import os

BASE_OUTPUT_DIR = "result"


class PathConfig:
    # 修改：文件名更新为 dataset.csv
    DATA_PATH = "data/dataset.csv"

    def __init__(self, model_name):
        self.model_name = model_name
        self.model_output_dir = os.path.join(BASE_OUTPUT_DIR, self.model_name)
        # 保持不变：初始化实例属性时使用类属性
        self.data_path = PathConfig.DATA_PATH

    def create_model_dirs(self):
        os.makedirs(self.model_output_dir, exist_ok=True)
        print(f"模型输出目录: {self.model_output_dir}")
        return self.model_output_dir

    @staticmethod
    def create_base_output_dir():
        os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
        return BASE_OUTPUT_DIR