import json
import os
import shutil
import random
import sys
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8')

# ================= 配置区域 =================
JSON_FILE = './data/annotations_350x.json'
IMAGE_ROOT = r'.\data'  # 包含 processed 文件夹的根目录
OUTPUT_DIR = r'.\data\dataset' # 整理后的目标目录
TRAIN_RATIO = 0.8
# ===========================================

def parse_label_studio_path(url):
    # 解析 Label Studio 的特殊路径格式
    if '?d=' in url:
        return url.split('?d=')[-1]
    return url

def organize_dataset():
    # 1. 加载数据
    with open(JSON_FILE, 'r', encoding='utf-8') as f:
        tasks = json.load(f)

    # 2. 建立 类别 -> 文件路径 的映射
    label_to_paths = defaultdict(list)
    
    for task in tasks:
        # 获取图片路径
        img_map = {
            'img_front': parse_label_studio_path(task['data']['front']),
            'img_back': parse_label_studio_path(task['data']['back']),
            'img_top': parse_label_studio_path(task['data']['top'])
        }
        
        # 获取标注结果 (匹配 to_name 和 label)
        if not task.get('annotations'): continue
        results = task['annotations'][0]['result']
        
        for res in results:
            target_img_key = res['to_name'] # 如 img_front
            if target_img_key in img_map:
                label = res['value']['choices'][0]
                full_path = os.path.join(IMAGE_ROOT, img_map[target_img_key])
                if os.path.exists(full_path):
                    label_to_paths[label].append(full_path)

    # 3. 分层抽样并复制文件
    for label, files in label_to_paths.items():
        random.shuffle(files)
        
        # 计算切分点，确保 val 至少有一个
        num_val = max(1, int(len(files) * (1 - TRAIN_RATIO)))
        # 如果总数只有1个，全部给val以满足“必须存在”的需求
        if len(files) == 1:
            val_files = files
            train_files = []
        else:
            val_files = files[:num_val]
            train_files = files[num_val:]

        # 执行复制
        for set_name, file_list in [('train', train_files), ('val', val_files)]:
            dest_dir = os.path.join(OUTPUT_DIR, set_name, label)
            os.makedirs(dest_dir, exist_ok=True)
            
            for fpath in file_list:
                fname = os.path.basename(fpath)
                # 为了防止不同文件夹下同名文件冲突，可以给文件名加个前缀（可选）
                # 这里假设原 event 文件夹名是唯一的
                event_name = os.path.basename(os.path.dirname(fpath))
                new_name = f"{event_name}_{fname}"
                shutil.copy(fpath, os.path.join(dest_dir, new_name))

    print(f"整理完成！数据已保存至: {OUTPUT_DIR}")
    # 打印统计信息
    for label in label_to_paths:
        t_count = len([x for x in os.listdir(os.path.join(OUTPUT_DIR, 'train', label))]) if os.path.exists(os.path.join(OUTPUT_DIR, 'train', label)) else 0
        v_count = len(os.listdir(os.path.join(OUTPUT_DIR, 'val', label)))
        print(f"类别 [{label}]: 训练集 {t_count} 张, 验证集 {v_count} 张")

if __name__ == '__main__':
    organize_dataset()