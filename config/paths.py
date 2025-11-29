import os

BASE_OUTPUT_DIR = "result"

class PathConfig:
    def __init__(self, model_name):
        self.model_name = model_name
        self.model_output_dir = os.path.join(BASE_OUTPUT_DIR, self.model_name)
        # 数据路径保持不变
        self.data_path = r"C:\Users\Administrator\Desktop\train.xlsx"

    def create_model_dirs(self):
        os.makedirs(self.model_output_dir, exist_ok=True)
        print(f"模型输出目录: {self.model_output_dir}")
        return self.model_output_dir

    @staticmethod
    def create_base_output_dir():
        os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
        return BASE_OUTPUT_DIR