import os
import csv
import math
import json
from pathlib import Path

import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader


# =========================================================
# 1. 配置
# =========================================================
class Config:
    DATA_DIR = Path("/mnt/ramdisk/dataset_9_class")
    MODEL_PATH = Path("./checkpoints/best_f1_model.pth")   # 改成你的旧最强模型路径
    OUTPUT_DIR = Path("./dataset/audit_results_old_best")

    TARGET_CLASSES = [
        "NoPackage",
        "NoWaybill",
        "TruncatedBarcode",
        "WrinkledWaybill",
    ]

    IMAGE_SIZE = 224
    BATCH_SIZE = 128
    NUM_WORKERS = min(32, max(2, (os.cpu_count() or 8) - 2))
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    # 审计阈值，可自己调
    HIGH_CONF_WRONG_THRESHOLD = 0.90
    LOW_CONF_THRESHOLD = 0.50

    # 导出前 N 个高风险样本
    TOPK_HIGH_LOSS = 500
    TOPK_HIGH_CONF_WRONG = 500
    TOPK_LOW_CONF = 500


# =========================================================
# 2. 数据集过滤
# =========================================================
def filter_imagefolder_by_classnames(dataset: datasets.ImageFolder, target_classnames):
    old_class_to_idx = dataset.class_to_idx

    missing = [c for c in target_classnames if c not in old_class_to_idx]
    if missing:
        raise ValueError(f"数据集中缺少类别: {missing}")

    old_to_new = {
        old_class_to_idx[class_name]: new_idx
        for new_idx, class_name in enumerate(target_classnames)
    }

    new_samples = []
    for path, old_label in dataset.samples:
        if old_label in old_to_new:
            new_samples.append((path, old_to_new[old_label]))

    dataset.samples = new_samples
    dataset.targets = [label for _, label in new_samples]
    dataset.classes = list(target_classnames)
    dataset.class_to_idx = {name: i for i, name in enumerate(target_classnames)}
    return dataset


def build_transform(cfg: Config):
    return transforms.Compose([
        transforms.Resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(cfg.MEAN, cfg.STD),
    ])


# =========================================================
# 3. 带路径的数据集
# =========================================================
class ImageFolderWithPath(datasets.ImageFolder):
    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        path, _ = self.samples[index]
        return image, label, path


def build_dataset(split: str, cfg: Config):
    dataset = ImageFolderWithPath(cfg.DATA_DIR / split, transform=build_transform(cfg))
    dataset = filter_imagefolder_by_classnames(dataset, cfg.TARGET_CLASSES)
    return dataset


# =========================================================
# 4. 模型
# =========================================================
def build_model(num_classes: int):
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def load_model(cfg: Config):
    model = build_model(len(cfg.TARGET_CLASSES))
    ckpt = torch.load(cfg.MODEL_PATH, map_location="cpu")

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict, strict=True)
    model = model.to(cfg.DEVICE)
    model.eval()
    return model


# =========================================================
# 5. 审计核心
# =========================================================
@torch.no_grad()
def audit_split(model, dataset, split_name: str, cfg: Config):
    loader = DataLoader(
        dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(cfg.NUM_WORKERS > 0),
    )

    criterion = nn.CrossEntropyLoss(reduction="none")

    rows = []
    class_names = dataset.classes

    for images, labels, paths in tqdm(loader, desc=f"Auditing {split_name}"):
        images = images.to(cfg.DEVICE, non_blocking=True)
        labels = labels.to(cfg.DEVICE, non_blocking=True)

        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(outputs, dim=1)

        losses = criterion(outputs, labels)

        max_probs, _ = torch.max(probs, dim=1)
        true_probs = probs.gather(1, labels.unsqueeze(1)).squeeze(1)

        for i in range(len(paths)):
            true_idx = labels[i].item()
            pred_idx = preds[i].item()

            row = {
                "split": split_name,
                "path": paths[i],
                "filename": Path(paths[i]).name,

                "true_idx": true_idx,
                "true_label": class_names[true_idx],

                "pred_idx": pred_idx,
                "pred_label": class_names[pred_idx],

                "is_wrong": int(pred_idx != true_idx),

                "max_conf": float(max_probs[i].item()),
                "true_label_prob": float(true_probs[i].item()),
                "loss": float(losses[i].item()),
            }

            # 保存每个类别的概率，方便后续筛查
            sample_probs = probs[i].detach().cpu().tolist()
            for cls_name, p in zip(class_names, sample_probs):
                row[f"prob_{cls_name}"] = float(p)

            rows.append(row)

    return rows


# =========================================================
# 6. 导出结果
# =========================================================
def save_csv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def export_audit_reports(df: pd.DataFrame, cfg: Config):
    out = cfg.OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    # 总表
    save_csv(df, out / "audit_all.csv")

    # 高置信度错判
    high_conf_wrong = df[
        (df["is_wrong"] == 1) &
        (df["max_conf"] >= cfg.HIGH_CONF_WRONG_THRESHOLD)
    ].sort_values(by=["max_conf", "loss"], ascending=[False, False])

    save_csv(high_conf_wrong.head(cfg.TOPK_HIGH_CONF_WRONG), out / "high_conf_wrong.csv")

    # 高 loss 样本
    high_loss = df.sort_values(by="loss", ascending=False)
    save_csv(high_loss.head(cfg.TOPK_HIGH_LOSS), out / "high_loss.csv")

    # 低置信度样本（不管对错都值得看）
    low_conf = df[df["max_conf"] <= cfg.LOW_CONF_THRESHOLD].sort_values(by="max_conf", ascending=True)
    save_csv(low_conf.head(cfg.TOPK_LOW_CONF), out / "low_conf.csv")

    # 常见混淆对
    def save_pair(true_label, pred_label, filename):
        pair_df = df[
            (df["true_label"] == true_label) &
            (df["pred_label"] == pred_label)
        ].sort_values(by=["max_conf", "loss"], ascending=[False, False])
        save_csv(pair_df, out / filename)

    save_pair("TruncatedBarcode", "NoWaybill", "confusion_truncated_to_nowaybill.csv")
    save_pair("NoWaybill", "TruncatedBarcode", "confusion_nowaybill_to_truncated.csv")
    save_pair("WrinkledWaybill", "TruncatedBarcode", "confusion_wrinkled_to_truncated.csv")
    save_pair("TruncatedBarcode", "WrinkledWaybill", "confusion_truncated_to_wrinkled.csv")
    save_pair("NoPackage", "NoWaybill", "confusion_nopackage_to_nowaybill.csv")
    save_pair("NoPackage", "TruncatedBarcode", "confusion_nopackage_to_truncated.csv")

    # 每个真实类别内部，最值得检查的错样本
    for cls_name in cfg.TARGET_CLASSES:
        sub = df[
            (df["true_label"] == cls_name) &
            (df["is_wrong"] == 1)
        ].sort_values(by=["loss", "max_conf"], ascending=[False, False])
        save_csv(sub, out / f"wrong_from_{cls_name}.csv")

    # 每个预测类别内部，高置信度“吸错”的样本
    for cls_name in cfg.TARGET_CLASSES:
        sub = df[
            (df["pred_label"] == cls_name) &
            (df["is_wrong"] == 1)
        ].sort_values(by=["max_conf", "loss"], ascending=[False, False])
        save_csv(sub, out / f"wrong_into_{cls_name}.csv")

    # 摘要统计
    summary = {
        "num_samples": int(len(df)),
        "num_wrong": int(df["is_wrong"].sum()),
        "overall_acc_from_audit": float(1.0 - df["is_wrong"].mean()),
        "high_conf_wrong_count": int(len(high_conf_wrong)),
        "low_conf_count": int(len(low_conf)),
    }

    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


# =========================================================
# 7. 人工复查建议输出
# =========================================================
def print_review_guide(df: pd.DataFrame, cfg: Config):
    print("\n" + "=" * 70)
    print("建议你优先人工复查这些文件：")
    print("=" * 70)

    high_conf_wrong = df[
        (df["is_wrong"] == 1) &
        (df["max_conf"] >= cfg.HIGH_CONF_WRONG_THRESHOLD)
    ].sort_values(by=["max_conf", "loss"], ascending=[False, False])

    print(f"\n1) 高置信度错判样本: {len(high_conf_wrong)} 个")
    print("   文件: audit_results_old_best/high_conf_wrong.csv")

    pair = df[
        (df["true_label"] == "TruncatedBarcode") &
        (df["pred_label"] == "NoWaybill")
    ]
    print(f"\n2) TruncatedBarcode -> NoWaybill 混淆样本: {len(pair)} 个")
    print("   文件: audit_results_old_best/confusion_truncated_to_nowaybill.csv")

    pair2 = df[
        (df["true_label"] == "TruncatedBarcode") &
        (df["pred_label"] == "WrinkledWaybill")
    ]
    print(f"\n3) TruncatedBarcode -> WrinkledWaybill 混淆样本: {len(pair2)} 个")
    print("   文件: audit_results_old_best/confusion_truncated_to_wrinkled.csv")

    high_loss = df.sort_values(by="loss", ascending=False)
    print(f"\n4) 高 loss 样本: 前 {cfg.TOPK_HIGH_LOSS} 个")
    print("   文件: audit_results_old_best/high_loss.csv")

    low_conf = df[df["max_conf"] <= cfg.LOW_CONF_THRESHOLD].sort_values(by="max_conf", ascending=True)
    print(f"\n5) 低置信度样本: {len(low_conf)} 个")
    print("   文件: audit_results_old_best/low_conf.csv")

    print("\n人工复查时重点判断：")
    print("- 这张图是不是其实同时有多种缺陷，但被强行标成单标签？")
    print("- 当前标签是不是明显不如模型预测标签合理？")
    print("- 这张图是不是本身质量差/裁切异常/几乎不可判？")
    print("- TruncatedBarcode 和 NoWaybill 的边界是不是被标得不一致？")
    print("=" * 70 + "\n")


# =========================================================
# 8. 主函数
# =========================================================
def main():
    cfg = Config()
    cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from: {cfg.MODEL_PATH}")
    model = load_model(cfg)

    train_set = build_dataset("train", cfg)
    val_set = build_dataset("val", cfg)

    print(f"Train size: {len(train_set)}")
    print(f"Val size: {len(val_set)}")

    train_rows = audit_split(model, train_set, "train", cfg)
    val_rows = audit_split(model, val_set, "val", cfg)

    df = pd.DataFrame(train_rows + val_rows)

    # 按 loss 从高到低看通常最直观
    df = df.sort_values(by="loss", ascending=False).reset_index(drop=True)

    export_audit_reports(df, cfg)
    print_review_guide(df, cfg)

    print(f"审计完成，结果已保存到: {cfg.OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()