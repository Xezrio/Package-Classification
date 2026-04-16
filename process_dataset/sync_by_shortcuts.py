import os
from pathlib import Path
import shutil

# ================= 配置 =================
AUDIT_ROOT = "./audit_output"         # 你的“虚拟指挥部”
DATASET_ROOT = "./dataset/dataset_9_class/train" # 原始数据集根目录
DRY_RUN = True  # ✨ 默认为 True：只打印不实际操作。确认没问题后再改成 False。
# ========================================

def execute_sync():
    audit_path = Path(AUDIT_ROOT)
    
    # 遍历你在指挥部建立的每个文件夹
    for action_folder in audit_path.iterdir():
        if not action_folder.is_dir():
            continue
            
        folder_name = action_folder.name
        print(f"\n📂 正在扫描目录: {folder_name}")

        # 逻辑判断：是删除还是移动
        if folder_name == "TO_DELETE":
            mode = "DELETE"
        elif folder_name.startswith("TO_"):
            mode = "MOVE"
            target_class = folder_name.replace("TO_", "")
        else:
            print(f"⚠️ 跳过未知指令文件夹: {folder_name}")
            continue

        # 遍历文件夹里的每一个快捷方式
        count = 0
        for shortcut in action_folder.iterdir():
            if not shortcut.is_symlink():
                continue
            
            # 1. 找到真身
            try:
                original_file = Path(os.readlink(shortcut)).absolute()
            except OSError:
                print(f"❌ 链接失效: {shortcut.name}")
                continue

            if not original_file.exists():
                print(f"❌ 原文件已不存在: {original_file}")
                continue

            # 2. 执行操作
            if mode == "DELETE":
                print(f"🔥 [准备删除] {original_file.name}")
                if not DRY_RUN:
                    os.remove(original_file)
                count += 1

            elif mode == "MOVE":
                dest_path = Path(DATASET_ROOT) / target_class / original_file.name
                if original_file == dest_path:
                    continue # 已经在目标位置了
                
                print(f"🚚 [准备移动] {original_file.parent.name} -> {target_class}")
                if not DRY_RUN:
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(original_file), str(dest_path))
                count += 1

        print(f"✅ 完成！该操作影响了 {count} 个原文件。")

if __name__ == "__main__":
    if DRY_RUN:
        print("🧪 当前处于 [演示模式 (DRY_RUN)]，不会真的改动文件。")
    execute_sync()
    if DRY_RUN:
        print("\n💡 如果打印的日志没问题，请将脚本中的 DRY_RUN 改为 False 后正式运行。")