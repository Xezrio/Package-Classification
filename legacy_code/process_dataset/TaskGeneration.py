import os
import json

OUT_JSON = r"D:\Documents\Code\Project\PackageClassification\data\tasks.json"

# 本地真实目录（扫描用）
PROCESSED_DIR = r"D:\Documents\Code\Project\PackageClassification\data\processed"

# Label Studio 访问路径（相对于 DOCUMENT_ROOT）
LS_RELATIVE_ROOT = "processed"

tasks = []

for event_folder in os.listdir(PROCESSED_DIR):
    event_path = os.path.join(PROCESSED_DIR, event_folder)
    if not os.path.isdir(event_path):
        continue

    front_path = os.path.join(event_path, "front.jpg")
    back_path  = os.path.join(event_path, "back.jpg")
    top_path   = os.path.join(event_path, "top.jpg")
    meta_path  = os.path.join(event_path, "meta.json")

    if not (
        os.path.exists(front_path)
        and os.path.exists(back_path)
        and os.path.exists(top_path)
        and os.path.exists(meta_path)
    ):
        print(f"Skipping incomplete event: {event_folder}")
        continue

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
        event_type = meta.get("event_read_type", "UNKNOWN")

    # Label Studio 本地文件路径
    front_ls = f"/data/local-files/?d={LS_RELATIVE_ROOT}/{event_folder}/front.jpg"
    back_ls  = f"/data/local-files/?d={LS_RELATIVE_ROOT}/{event_folder}/back.jpg"
    top_ls   = f"/data/local-files/?d={LS_RELATIVE_ROOT}/{event_folder}/top.jpg"

    task = {
        "data": {
            "front": front_ls,
            "back": back_ls,
            "top": top_ls,
            "event_read_type": event_type
        }
    }

    tasks.append(task)

with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(tasks, f, indent=2, ensure_ascii=False)

print(f"Generated {len(tasks)} tasks to {OUT_JSON}")
