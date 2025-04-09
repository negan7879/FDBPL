import os
import shutil
import random

def sample_images(src_dir, dest_dir, num_samples=16):
    # 确保目标目录存在
    os.makedirs(dest_dir, exist_ok=True)

    # 获取类别目录列表
    categories = [d for d in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, d))]

    for category in categories:
        src_category_path = os.path.join(src_dir, category)
        dest_category_path = os.path.join(dest_dir, category)

        # 创建目标类别目录
        os.makedirs(dest_category_path, exist_ok=True)

        # 获取类别下的所有图像文件
        images = [img for img in os.listdir(src_category_path) if os.path.isfile(os.path.join(src_category_path, img))]

        # 随机采样图像
        sampled_images = random.sample(images, min(num_samples, len(images)))

        # 复制采样的图像到目标目录
        for img in sampled_images:
            src_img_path = os.path.join(src_category_path, img)
            dest_img_path = os.path.join(dest_category_path, img)
            shutil.copy(src_img_path, dest_img_path)

# 定义源和目标路径
src_path = '/media/data1/zhangzherui/data/DATA/imagenet/images/train'
dest_path = '/media/data1/zhangzherui/data/DATA/small_imagenet/images/train'

# 调用函数
sample_images(src_path, dest_path)
