import os
import shutil
import random

# 本地源数据文件夹路径
source_dir = "../data_raw"
# 本地目标测试集文件夹路径
target_dir = "../test_data"

# 要抽取的类别列表
classes = ["bend", "fall", "lie_down", "run", "sitdown", "standup", "walk"]
# 每个类别抽取的文件数量
num_files_per_class = 12


# 新增：清空目标目录及其子文件夹
def clear_directory(directory):
    if os.path.exists(directory):
        # 遍历目录下的所有内容
        for item in os.listdir(directory):
            item_path = os.path.join(directory, item)
            # 删除文件或文件夹
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.unlink(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
        print(f"已清空目标目录：{directory}")
    else:
        print(f"目标目录不存在，无需清空：{directory}")


# 先清空目标目录，再创建新的结构
clear_directory(target_dir)
os.makedirs(target_dir, exist_ok=True)  # 确保目标目录存在

for class_name in classes:
    # 构建源类别文件夹路径
    class_source_dir = os.path.join(source_dir, class_name)
    # 构建目标类别文件夹路径
    class_target_dir = os.path.join(target_dir, class_name)
    # 确保目标类别文件夹存在
    os.makedirs(class_target_dir, exist_ok=True)

    # 检查源类别文件夹是否存在
    if not os.path.exists(class_source_dir):
        print(f"警告：源类别文件夹 '{class_source_dir}' 不存在，跳过该类别")
        continue

    # 获取源类别文件夹下的所有文件
    files = [f for f in os.listdir(class_source_dir) if os.path.isfile(os.path.join(class_source_dir, f))]

    # 检查文件数量是否足够
    if len(files) < num_files_per_class:
        print(f"警告：类别 '{class_name}' 的文件数量不足 {num_files_per_class} 个，仅抽取 {len(files)} 个")
        selected_files = files
    else:
        # 随机打乱文件列表并抽取指定数量的文件
        random.shuffle(files)
        selected_files = files[:num_files_per_class]

    # 复制选中的文件到目标类别文件夹
    for file_name in selected_files:
        source_file = os.path.join(class_source_dir, file_name)
        target_file = os.path.join(class_target_dir, file_name)
        shutil.copy2(source_file, target_file)
        print(f"已复制：{source_file} -> {target_file}")

print(f"\n测试集抽取完成，已保存至：{target_dir}")
