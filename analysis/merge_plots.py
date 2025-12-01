import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

# 定义路径配置 (需与 config/paths.py 和 main.py 中的定义一致)
BASE_OUTPUT_DIR = "result"

# 定义模型顺序和名称 (对应 main.py 中的 model_name)
# 格式: (文件夹名称, 文件名前缀)
MODELS = [
    ('SVM', 'svm'),  # 最左边
    ('CatBoost', 'catboost'),  # 中间
    ('TabPFN', 'tabpfn')  # 最右边
]


def crop_image(image_path):
    """
    读取图片并将其从中间裁剪为左右两部分 (验证集部分, 测试集部分)
    """
    if not os.path.exists(image_path):
        print(f"错误: 找不到文件 {image_path}")
        return None, None

    # 使用 PIL 打开图片以进行精确裁剪
    img = Image.open(image_path)
    width, height = img.size

    # 假设原图是 1行2列 (Validation | Test)，从中间分割
    # 注意：根据 visualization.py 的 figsize=(14, 6)，左右是对称的
    mid_point = width // 2

    # 裁剪左边 (验证集)
    # left, upper, right, lower
    img_val = img.crop((0, 0, mid_point, height))

    # 裁剪右边 (测试集)
    img_test = img.crop((mid_point, 0, width, height))

    return img_val, img_test


def merge_plots():
    print("开始合并模型图像...")

    # 准备容器
    val_images = []
    test_images = []

    # 1. 读取并裁剪所有图片
    for model_dir, file_prefix in MODELS:
        # 构建完整路径: result/SVM/svm_final_model.png
        img_path = os.path.join(BASE_OUTPUT_DIR, model_dir, f"{file_prefix}_final_model.png")
        print(f"处理图片: {img_path}")

        val_img, test_img = crop_image(img_path)

        if val_img is None:
            print("合并失败：缺少必要的文件。请确保您已经运行了 SVM, CatBoost 和 TabPFN 模型。")
            return

        val_images.append(val_img)
        test_images.append(test_img)

    # 2. 创建画布
    # 获取单张子图的尺寸
    w, h = val_images[0].size

    # 创建新图像：宽 = 3 * 单图宽，高 = 2 * 单图高
    # 这里的 w, h 是像素值
    total_width = w * 3
    total_height = h * 2

    new_im = Image.new('RGB', (total_width, total_height), (255, 255, 255))

    # 3. 粘贴图片
    # 粘贴第一行 (验证集)
    for i, img in enumerate(val_images):
        new_im.paste(img, (i * w, 0))

    # 粘贴第二行 (测试集)
    for i, img in enumerate(test_images):
        new_im.paste(img, (i * w, h))

    # 4. 保存结果
    output_path = os.path.join(BASE_OUTPUT_DIR, "merged_model_comparison.png")
    new_im.save(output_path, quality=95)
    print(f"\n成功！合并后的图片已保存至: {output_path}")

    # 5. (可选) 使用 Matplotlib 显示结果以便查看
    plt.figure(figsize=(18, 10))
    plt.imshow(new_im)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    merge_plots()