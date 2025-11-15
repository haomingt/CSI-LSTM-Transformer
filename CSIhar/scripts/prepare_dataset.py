import os, re, shutil, csv

SOURCE_ROOT = "../data_raw"     # 下载后的原始目录
TARGET_ROOT = "../data_raw"
CLASS_NAME_MAP = {
    "lie down": "lie_down",   # 修正规范化
    "liedown": "lie_down",    # 有些数据集版本可能是这种
    "lie_down": "lie_down"    # 安全映射
}
VALID_CLASSES = ["bend","fall","lie_down","run","sitdown","standup","walk"]
EXPECTED_SUBCARRIERS = 52
MOVE_FILES = True   # True=移动; False=复制

def normalize_class_dir(name: str):
    nm = name.strip().lower()
    if nm in CLASS_NAME_MAP:
        return CLASS_NAME_MAP[nm]
    return nm

def safe_makedirs(p):
    os.makedirs(p, exist_ok=True)

def normalize_filename(fname: str):
    # 1. 去掉多余空格开头结尾
    new = fname.strip()
    # 2. 中间空格替换为下划线
    new = re.sub(r"\s+", "_", new)
    # 3. 可选：将 liedown → lie_down
    new = new.replace("liedown", "lie_down").replace("lie-down","lie_down")
    return new

def is_csv(f):
    return f.lower().endswith(".csv")

def check_csv_columns(path):
    # 读取第一行看列数；若为空则返回 False
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) == 0:
                continue
            return len(row)
    return 0

def main():
    safe_makedirs(TARGET_ROOT)
    for entry in os.listdir(SOURCE_ROOT):
        full_dir = os.path.join(SOURCE_ROOT, entry)
        if not os.path.isdir(full_dir):
            continue
        cls_norm = normalize_class_dir(entry)
        if cls_norm not in VALID_CLASSES:
            print(f"[WARN] 忽略未知类别目录: {entry} -> {cls_norm}")
            continue

        out_dir = os.path.join(TARGET_ROOT, cls_norm)
        safe_makedirs(out_dir)

        for f in os.listdir(full_dir):
            if not is_csv(f):
                continue
            old_path = os.path.join(full_dir, f)
            new_name = normalize_filename(f)
            new_path = os.path.join(out_dir, new_name)

            # 校验列数
            col_cnt = check_csv_columns(old_path)
            if col_cnt != EXPECTED_SUBCARRIERS:
                print(f"[ERROR] 列数不等于 {EXPECTED_SUBCARRIERS}: {old_path} (检测到 {col_cnt}) -> 跳过")
                continue

            if MOVE_FILES:
                shutil.move(old_path, new_path)
            else:
                shutil.copy2(old_path, new_path)

        print(f"[OK] 类别 {entry} -> {cls_norm} 处理完成。输出目录: {out_dir}")

    print("全部处理完成。建议检查 data_raw/ 下是否包含 7 个规范目录。")

if __name__ == "__main__":
    main()
