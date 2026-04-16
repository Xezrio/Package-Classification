import os
import shutil
import json

RAW_ROOT = r".\data\raw"
OUT_ROOT = r".\data\processed"

CAMERA_NAME_MAP = {
    "qian": "front",
    "hou": "back",
    "ding": "top"
}

IMAGE_READ_TYPES = ["NR", "MR", "GR"]


def extract_event_id(filename):

    # 从文件名从后往前提取时间戳

    name = os.path.splitext(filename)[0]
    parts = name.split("_")

    for part in reversed(parts):
        if part.isdigit() and len(part) >= 14:
            return part

    return None


def extract_camera_name(filename):

    # 从文件名从后往前提取相机名
    # 并映射为 front / back / top

    name = os.path.splitext(filename)[0]
    parts = name.split("_")

    for part in reversed(parts):
        if part in CAMERA_NAME_MAP:
            return CAMERA_NAME_MAP[part]

    return "unknown"


def extract_image_read_type(filename):

    # 从文件名从后往前提取图片状态（GR / NR / MR）

    name = os.path.splitext(filename)[0]
    parts = name.split("_")

    for part in reversed(parts):
        if part in IMAGE_READ_TYPES:
            return part

    return "UNKNOWN"


def extract_event_read_type_from_path(path):

    # 从路径中的事件文件夹提取事件状态（MR / NR）

    parts = path.split(os.sep)
    for p in parts:
        if p.endswith("_NR"):
            return "NR"
        if p.endswith("_MR"):
            return "MR"
    return "UNKNOWN"


def collect_all_images(root):

    # 遍历 raw 目录，收集所有 jpg 图片路径

    image_paths = []
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if fname.lower().endswith(".jpg"):
                image_paths.append(os.path.join(dirpath, fname))
    return image_paths


def organize_dataset():
    print("Scanning raw dataset...")
    image_paths = collect_all_images(RAW_ROOT)
    total = len(image_paths)
    print(f"Found {total} images.")

    events = {}
    processed = 0

    for img_path in image_paths:
        processed += 1
        fname = os.path.basename(img_path)

        event_id = extract_event_id(fname)
        cam = extract_camera_name(fname)
        image_read_type = extract_image_read_type(fname)
        event_read_type = extract_event_read_type_from_path(img_path)

        valid = True
        if event_id is None or event_read_type == "UNKNOWN":
            valid = False

        if not valid:
            continue

        if event_id not in events:
            events[event_id] = {
                "event_read_type": event_read_type,
                "images": {}
            }

        events[event_id]["images"][cam] = {
            "path": img_path,
            "image_read_type": image_read_type
        }

    print(f"Collected {len(events)} events.")
    print("Writing organized dataset...")

    for idx, (event_id, event_data) in enumerate(events.items(), 1):
        event_type = event_data["event_read_type"]
        event_dir = os.path.join(
            OUT_ROOT, f"event_{event_id}_{event_type}"
        )
        os.makedirs(event_dir, exist_ok=True)

        meta = {
            "event_id": event_id,
            "event_read_type": event_type,
            "images": {}
        }

        for cam, img_info in event_data["images"].items():
            src = img_info["path"]
            dst = os.path.join(event_dir, f"{cam}.jpg")

            shutil.copy2(src, dst)

            meta["images"][cam] = {
                "image_read_type": img_info["image_read_type"],
                "original_path": src
            }

        meta_path = os.path.join(event_dir, "meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        print(f"[{idx}/{len(events)}] Event {event_id} written.")

    print("Dataset organization finished.")


if __name__ == "__main__":
    organize_dataset()