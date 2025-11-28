import os


class PathConfig:
    def __init__(self, model_name):
        self.model_name = model_name
        self.output_dir = fr"C:\Users\Administrator\Desktop\{model_name}"
        self.data_path = r"C:\Users\Administrator\Desktop\train.xlsx"

    def create_dirs(self):
        """创建必要的目录"""
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"输出目录: {self.output_dir}")
        return self.output_dir