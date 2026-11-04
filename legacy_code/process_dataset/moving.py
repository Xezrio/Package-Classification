import json
import os
import shutil
from tqdm import tqdm

def start_clean_move(json_path, src_base, dst_base):
    # 1. 加载 JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 2. 确保目标根目录存在
    if not os.path.exists(dst_base):
        os.makedirs(dst_base)
        print(f"创建目标目录: {dst_base}")

    moved_count = 0
    missing_count = 0

    # 3. 开始遍历
    for item in tqdm(data, desc="Relocating Labeled Data"):
        # 从 JSON 中提取文件夹名
        # 路径示例: /data/local-files/?d=processed/event_20260110075629975_NR/front.jpg
        sample_path = item['data'].get('front', '')
        if not sample_path or 'processed/' not in sample_path:
            continue
            
        try:
            folder_name = sample_path.split('processed/')[1].split('/')[0]
            src_path = os.path.join(src_base, folder_name)
            dst_path = os.path.join(dst_base, folder_name)

            if os.path.exists(src_path):
                # 核心逻辑：如果目的地已经有了（万一没删干净），直接替换
                if os.path.exists(dst_path):
                    shutil.rmtree(dst_path)
                
                shutil.move(src_path, dst_path)
                moved_count += 1
            else:
                # 记录一下哪些文件夹在源目录里找不到了
                missing_count += 1
        except Exception as e:
            print(f"处理 {folder_name} 时出错: {e}")

    print(f"\n任务完成！")
    print(f"--- 成功移动: {moved_count} 个文件夹")
    print(f"--- 源目录缺失: {missing_count} 个 (可能之前已经手动删过)")

if __name__ == "__main__":
    # 根据实际路径调整
    JSON_FILE = 'dataset/annotations_350x.json'
    SOURCE = 'dataset/raw'
    TARGET = 'dataset/labeled_data'
    
    start_clean_move(JSON_FILE, SOURCE, TARGET)