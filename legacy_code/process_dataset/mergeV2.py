import os
import shutil
from tqdm import tqdm

# --- 路径配置 ---
# 1. 你现有的手动标注训练集路径
GOLDEN_TRAIN_DIR = "/mnt/F/xezrio/PackageClassification/dataset/dataset_9_class/train"
# 2. 刚才预测出来并经过你清洗的 9000+ 张银色数据
SILVER_DATA_DIR = "auto_label_results/high_conf"

CLASSES = ['BlurryFocus', 'BlurryWaybill', 'InsufficientLighting', 'NoPackage', 
           'NoWaybill', 'None', 'Reflection', 'TruncatedBarcode', 'WrinkledWaybill']

def merge_to_v2():
    print("正在合并银色数据到原始训练集...")
    
    for cls in CLASSES:
        src_silver = os.path.join(SILVER_DATA_DIR, cls)
        dst_train = os.path.join(GOLDEN_TRAIN_DIR, cls)
        
        # 确保目标类文件夹存在
        os.makedirs(dst_train, exist_ok=True)
        
        if os.path.exists(src_silver):
            files = [f for f in os.listdir(src_silver) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            for f in tqdm(files, desc=f"Adding silver to {cls}"):
                src_file = os.path.join(src_silver, f)
                # 为了防止重名，加上 silver_ 前缀
                dst_file = os.path.join(dst_train, f"silver_{f}")
                
                # 使用 copy2 保留元数据
                shutil.copy2(src_file, dst_file)

    print("\n合并完成！你的训练集已显著扩充。")
    print(f"验证集路径保持不变，依然使用: .../dataset_9_class/val")

if __name__ == "__main__":
    merge_to_v2()