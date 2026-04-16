import os
from pathlib import Path
import shutil

# ================= 配置 =================
AUDIT_ROOT = "./audit_output"         # 你的“指挥部”
DATASET_ROOT = "./dataset/dataset_9_class/train" # 原始数据集根目录
DRY_RUN = False  # ✨ 建议先保持 True 运行一次看日志，没问题再改 False
# ========================================

print(f"当前工作目录: {os.getcwd()}")
print(f"准备扫描的路径: {Path(AUDIT_ROOT).absolute()}")
    
def execute_sync():
    audit_path = Path(AUDIT_ROOT)
    
    for action_folder in audit_path.iterdir():
        if not action_folder.is_dir(): continue
            
        folder_name = action_folder.name
        
        # 1. 确定指令
        if folder_name == "TO_DELETE":
            mode = "DELETE"
        elif folder_name.startswith("TO_"):
            mode = "MOVE"
            target_class = folder_name.replace("TO_", "")
        else:
            continue

        print(f"\n🚀 正在处理指令: {folder_name}")

        # 2. 遍历该指令文件夹里的快捷方式/硬链接
        count = 0
        for item in action_folder.iterdir():
            # 兼容软链接和下载回来的真实文件
            if item.is_symlink():
                original_file = Path(os.readlink(item)).absolute()
            else:
                # 如果是下载后传回来的，通过文件名寻找原图
                # 假设之前审计的文件名格式是: Loss_X.X_Conf_X.X_original_name.jpg
                original_name = item.name.split('_')[-1]
                # 这里建议根据你的实际文件名结构微调
                search_results = list(Path(DATASET_ROOT).rglob(original_name))
                if not search_results: continue
                original_file = search_results[0].absolute()

            if not original_file.exists(): continue

            # 3. 执行物理操作
            if mode == "DELETE":
                print(f"🔥 [删除] {original_file.name}")
                if not DRY_RUN: os.remove(original_file)
                count += 1
            elif mode == "MOVE":
                dest_dir = Path(DATASET_ROOT) / target_class
                dest_path = dest_dir / original_file.name
                if original_file == dest_path: continue
                
                print(f"🚚 [移动] {original_file.parent.name} -> {target_class}")
                if not DRY_RUN:
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(original_file), str(dest_path))
                count += 1

        print(f"✅ 处理完成，影响了 {count} 个文件。")

if __name__ == "__main__":
    execute_sync()