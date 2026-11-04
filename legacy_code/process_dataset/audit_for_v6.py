import os
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


# =========================================================
# 1. 配置
# =========================================================
class Config:
    DATA_DIR = Path("/mnt/ramdisk/dataset_9_class")
    MODEL_PATH = Path("./checkpoints/checkpoints_resnet34_v6_round_3/best_f1_model.pth")
    OUTPUT_DIR = Path("./dataset/audit_for_v6_round_3_deep")

    TARGET_CLASSES = [
        "NoPackage",
        "NoWaybill",
        "TruncatedBarcode",
        "WrinkledWaybill",
    ]

    INPUT_H = 560
    INPUT_W = 700

    BATCH_SIZE = 16
    NUM_WORKERS = min(8, max(2, (os.cpu_count() or 8) - 2))
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    MASK_THUMBNAIL = True
    THUMB_H_RATIO = 0.17
    THUMB_W_RATIO = 0.42

    HIGH_CONF_WRONG_THRESHOLD = 0.90
    LOW_CONF_THRESHOLD = 0.50

    TOPK_HIGH_LOSS = 500
    TOPK_HIGH_CONF_WRONG = 300
    TOPK_LOW_CONF = 300

    # 是否复制样本到文件夹里，方便肉眼复查
    COPY_FILES = False
    COPY_LIMIT_PER_GROUP = 200


# =========================================================
# 2. 预处理
# =========================================================
class MaskBottomLeftThumbnail:
    def __init__(self, h_ratio=0.17, w_ratio=0.42, enabled=True):
        self.h_ratio = h_ratio
        self.w_ratio = w_ratio
        self.enabled = enabled

    def __call__(self, img):
        if not self.enabled:
            return img

        arr = np.array(img)
        if arr.ndim == 3:
            arr = arr[..., 0]

        h, w = arr.shape[:2]
        mask_h = int(h * self.h_ratio)
        mask_w = int(w * self.w_ratio)

        fill_value = int(np.median(arr))
        arr[h - mask_h:h, 0:mask_w] = fill_value

        return Image.fromarray(arr)


class ToGray1:
    def __call__(self, img):
        return img.convert("L")


def build_transform(cfg: Config):
    return transforms.Compose([
        MaskBottomLeftThumbnail(
            h_ratio=cfg.THUMB_H_RATIO,
            w_ratio=cfg.THUMB_W_RATIO,
            enabled=cfg.MASK_THUMBNAIL
        ),
        ToGray1(),
        transforms.Resize((cfg.INPUT_H, cfg.INPUT_W)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.25]),
    ])


# =========================================================
# 3. 数据集
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
    model = models.resnet34(weights=None)

    old_conv = model.conv1
    new_conv = nn.Conv2d(
        in_channels=1,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False
    )
    with torch.no_grad():
        new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
    model.conv1 = new_conv

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def load_model(cfg: Config):
    ckpt = torch.load(cfg.MODEL_PATH, map_location="cpu")

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt

    model = build_model(len(cfg.TARGET_CLASSES))
    model.load_state_dict(state_dict, strict=True)
    model = model.to(cfg.DEVICE)
    model.eval()
    return model


# =========================================================
# 5. 审计
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
                "true_label": dataset.classes[true_idx],

                "pred_idx": pred_idx,
                "pred_label": dataset.classes[pred_idx],

                "is_wrong": int(pred_idx != true_idx),

                "max_conf": float(max_probs[i].item()),
                "true_label_prob": float(true_probs[i].item()),
                "loss": float(losses[i].item()),
            }

            sample_probs = probs[i].detach().cpu().tolist()
            for cls_name, p in zip(dataset.classes, sample_probs):
                row[f"prob_{cls_name}"] = float(p)

            rows.append(row)

    return rows


# =========================================================
# 6. 导出
# =========================================================
def save_csv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def copy_rows_to_dir(df: pd.DataFrame, out_dir: Path, limit: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, (_, row) in enumerate(df.head(limit).iterrows(), start=1):
        src = Path(row["path"])
        if not src.exists():
            continue
        dst_name = f"{i:04d}__{row['split']}__true-{row['true_label']}__pred-{row['pred_label']}__conf-{row['max_conf']:.3f}__{src.name}"
        dst = out_dir / dst_name
        try:
            shutil.copy2(src, dst)
        except Exception:
            pass


def export_reports(df: pd.DataFrame, cfg: Config):
    out = cfg.OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    save_csv(df, out / "audit_all.csv")

    high_conf_wrong = df[
        (df["is_wrong"] == 1) &
        (df["max_conf"] >= cfg.HIGH_CONF_WRONG_THRESHOLD)
    ].sort_values(by=["max_conf", "loss"], ascending=[False, False])
    save_csv(high_conf_wrong.head(cfg.TOPK_HIGH_CONF_WRONG), out / "high_conf_wrong.csv")

    high_loss = df.sort_values(by="loss", ascending=False)
    save_csv(high_loss.head(cfg.TOPK_HIGH_LOSS), out / "high_loss.csv")

    low_conf = df[df["max_conf"] <= cfg.LOW_CONF_THRESHOLD].sort_values(by="max_conf", ascending=True)
    save_csv(low_conf.head(cfg.TOPK_LOW_CONF), out / "low_conf.csv")

    def pair_df(true_label, pred_label):
        return df[
            (df["true_label"] == true_label) &
            (df["pred_label"] == pred_label)
        ].sort_values(by=["max_conf", "loss"], ascending=[False, False])

    # 自动生成全部 12 种有向混淆
    pairs = {}
    for true_label in cfg.TARGET_CLASSES:
        for pred_label in cfg.TARGET_CLASSES:
            if true_label == pred_label:
                continue
            key = f"confusion_{true_label.lower()}_to_{pred_label.lower()}"
            pairs[key] = pair_df(true_label, pred_label)

    for name, subdf in pairs.items():
        save_csv(subdf, out / f"{name}.csv")

    # 每个真实类别内部的错样本
    for cls_name in cfg.TARGET_CLASSES:
        sub = df[
            (df["true_label"] == cls_name) &
            (df["is_wrong"] == 1)
        ].sort_values(by=["loss", "max_conf"], ascending=[False, False])
        save_csv(sub, out / f"wrong_from_{cls_name}.csv")

    # ========= 你现在最需要的定向导出 =========

    # 1) TruncatedBarcode 中最像 NoPackage 的样本
    tb_like_nopackage = df[
        df["true_label"] == "TruncatedBarcode"
    ].sort_values(by=["prob_NoPackage", "loss"], ascending=[False, False])
    save_csv(tb_like_nopackage.head(200), out / "truncatedbarcode_most_like_nopackage.csv")

    # 2) TruncatedBarcode 中高 loss 样本
    tb_high_loss = df[
        df["true_label"] == "TruncatedBarcode"
    ].sort_values(by="loss", ascending=False)
    save_csv(tb_high_loss.head(200), out / "truncatedbarcode_high_loss.csv")

    # 3) TruncatedBarcode 中最像 NoWaybill 的样本
    tb_like_nowaybill = df[
        df["true_label"] == "TruncatedBarcode"
    ].sort_values(by=["prob_NoWaybill", "loss"], ascending=[False, False])
    save_csv(tb_like_nowaybill.head(200), out / "truncatedbarcode_most_like_nowaybill.csv")

    # 4) NoWaybill 中最像 TruncatedBarcode 的样本
    nw_like_truncated = df[
        df["true_label"] == "NoWaybill"
    ].sort_values(by=["prob_TruncatedBarcode", "loss"], ascending=[False, False])
    save_csv(nw_like_truncated.head(200), out / "nowaybill_most_like_truncatedbarcode.csv")

    # 5) 直接导出最关键的双向混淆（如果有的话）
    tb_to_nw = pair_df("TruncatedBarcode", "NoWaybill")
    nw_to_tb = pair_df("NoWaybill", "TruncatedBarcode")
    save_csv(tb_to_nw, out / "focus_truncatedbarcode_to_nowaybill.csv")
    save_csv(nw_to_tb, out / "focus_nowaybill_to_truncatedbarcode.csv")

    # 6) TruncatedBarcode 中低置信度样本
    tb_low_conf = df[
        df["true_label"] == "TruncatedBarcode"
    ].sort_values(by="max_conf", ascending=True)
    save_csv(tb_low_conf.head(200), out / "truncatedbarcode_low_conf.csv")

    # 7) NoWaybill 中低置信度样本
    nw_low_conf = df[
        df["true_label"] == "NoWaybill"
    ].sort_values(by="max_conf", ascending=True)
    save_csv(nw_low_conf.head(200), out / "nowaybill_low_conf.csv")

    # 混淆对汇总表（按数量排序）
    pair_summary_rows = []
    for name, subdf in pairs.items():
        pair_summary_rows.append({
            "pair_name": name,
            "count": int(len(subdf))
        })

    pair_summary_df = pd.DataFrame(pair_summary_rows).sort_values(by="count", ascending=False)
    save_csv(pair_summary_df, out / "pair_summary.csv")

    summary = {
        "num_samples": int(len(df)),
        "num_wrong": int(df["is_wrong"].sum()),
        "overall_acc_from_audit": float(1.0 - df["is_wrong"].mean()),
        "high_conf_wrong_count": int(len(high_conf_wrong)),
        "low_conf_count": int(len(low_conf)),
        "pair_counts": {k: int(len(v)) for k, v in pairs.items()},
        "focus_counts": {
            "truncatedbarcode_to_nowaybill": int(len(tb_to_nw)),
            "nowaybill_to_truncatedbarcode": int(len(nw_to_tb)),
        }
    }
    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if cfg.COPY_FILES:
        copy_base = out / "review_folders"

        copy_rows_to_dir(
            high_conf_wrong,
            copy_base / "00_high_conf_wrong",
            cfg.COPY_LIMIT_PER_GROUP
        )
        copy_rows_to_dir(
            high_loss,
            copy_base / "01_high_loss",
            cfg.COPY_LIMIT_PER_GROUP
        )
        copy_rows_to_dir(
            low_conf,
            copy_base / "02_low_conf",
            cfg.COPY_LIMIT_PER_GROUP
        )

        # 重点人工审查目录
        copy_rows_to_dir(
            tb_to_nw,
            copy_base / "03_focus_truncatedbarcode_to_nowaybill",
            cfg.COPY_LIMIT_PER_GROUP
        )
        copy_rows_to_dir(
            nw_to_tb,
            copy_base / "04_focus_nowaybill_to_truncatedbarcode",
            cfg.COPY_LIMIT_PER_GROUP
        )
        copy_rows_to_dir(
            tb_like_nopackage,
            copy_base / "05_truncatedbarcode_most_like_nopackage",
            cfg.COPY_LIMIT_PER_GROUP
        )
        copy_rows_to_dir(
            tb_like_nowaybill,
            copy_base / "06_truncatedbarcode_most_like_nowaybill",
            cfg.COPY_LIMIT_PER_GROUP
        )
        copy_rows_to_dir(
            nw_like_truncated,
            copy_base / "07_nowaybill_most_like_truncatedbarcode",
            cfg.COPY_LIMIT_PER_GROUP
        )

        for idx, (name, subdf) in enumerate(sorted(pairs.items()), start=8):
            copy_rows_to_dir(
                subdf,
                copy_base / f"{idx:02d}_{name}",
                cfg.COPY_LIMIT_PER_GROUP
            )

    return summary


def print_summary(summary, cfg: Config):
    print("\n" + "=" * 72)
    print("审计完成")
    print("=" * 72)
    print(f"总样本数: {summary['num_samples']}")
    print(f"错误样本数: {summary['num_wrong']}")
    print(f"审计准确率: {summary['overall_acc_from_audit']:.4f}")
    print(f"高置信错判: {summary['high_conf_wrong_count']}")
    print(f"低置信样本: {summary['low_conf_count']}")
    print("\n重点混淆对数量:")
    for k, v in summary["pair_counts"].items():
        print(f"  - {k}: {v}")

    print("\n建议你优先人工看：")
    print("1. review_folders/03_truncated_to_nowaybill")
    print("2. review_folders/04_nowaybill_to_truncated")
    print("3. review_folders/01_high_conf_wrong")
    print("4. review_folders/02_high_loss")
    print("\n输出目录:")
    print(cfg.OUTPUT_DIR.resolve())
    print("=" * 72 + "\n")


# =========================================================
# 7. 主函数
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
    df = df.sort_values(by="loss", ascending=False).reset_index(drop=True)

    summary = export_reports(df, cfg)
    print_summary(summary, cfg)


if __name__ == "__main__":
    main()