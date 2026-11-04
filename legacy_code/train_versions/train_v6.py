import os
import json
import random
import logging
from pathlib import Path
from collections import Counter

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from tqdm import tqdm


# =========================================================
# 1. 配置
# =========================================================
class Config:
    DATA_DIR = Path("/mnt/ramdisk/dataset_9_class")
    CHECKPOINT_DIR = Path("./checkpoints/checkpoints_resnet34_v6_round_final")
    LOG_FILE = CHECKPOINT_DIR / "train.log"

    # 注意：这里顺序要和你真正训练时保持一致
    TARGET_CLASSES = [
        "NoPackage",
        "NoWaybill",
        "TruncatedBarcode",
        "WrinkledWaybill",
    ]

    # 保持原始比例：4352 x 5440 ≈ 4 : 5
    # try 560 x 700 next time (lr to 1e-4, batch to 16?)
    # we chose 448 x 560 for round 1 and 2
    INPUT_H = 560
    INPUT_W = 700   

    # we have chosen batch 32, lr 2e-4 for round 2
    BATCH_SIZE = 16
    EPOCHS = 30
    LR = 1e-4
    WEIGHT_DECAY = 5e-4

    # too many pd_data_workers!
    # NUM_WORKERS = min(24, max(2, (os.cpu_count() or 8) - 4))
    TRAIN_NUM_WORKERS = 12
    VAL_NUM_WORKERS = 4
    PREFETCH_FACTOR = 2

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    AMP = torch.cuda.is_available()
    SEED = 42

    EARLY_STOPPING_PATIENCE = 8     # we have chose 6 in round 1
    SAVE_EVERY = 5

    # 左下角缩略图区域 mask 比例，按你图像大致布局设置
    MASK_THUMBNAIL = True
    THUMB_H_RATIO = 0.17
    THUMB_W_RATIO = 0.42

    BEST_F1_PATH = CHECKPOINT_DIR / "best_f1_model.pth"
    BEST_ACC_PATH = CHECKPOINT_DIR / "best_acc_model.pth"
    LATEST_PATH = CHECKPOINT_DIR / "latest_model.pth"


# =========================================================
# 2. 基础工具
# =========================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def build_logger(log_file: Path):
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("train_logger_v2")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(formatter)

    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger


def format_confusion_matrix(cm, class_names):
    lines = []
    header = "GT\\Pred".ljust(20) + "".join([c[:14].ljust(16) for c in class_names])
    lines.append(header)
    for i, row in enumerate(cm):
        line = class_names[i][:18].ljust(20)
        for v in row:
            line += str(v).ljust(16)
        lines.append(line)
    return "\n".join(lines)


# =========================================================
# 3. 自定义预处理
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
            arr = arr[..., 0]  # 转成单通道处理

        h, w = arr.shape[:2]
        mask_h = int(h * self.h_ratio)
        mask_w = int(w * self.w_ratio)

        fill_value = int(np.median(arr))
        arr[h - mask_h:h, 0:mask_w] = fill_value

        return Image.fromarray(arr)


class ToGray1:
    def __call__(self, img):
        return img.convert("L")


# =========================================================
# 4. 数据集过滤
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


def build_transforms(cfg: Config):
    train_transform = transforms.Compose([
        MaskBottomLeftThumbnail(
            h_ratio=cfg.THUMB_H_RATIO,
            w_ratio=cfg.THUMB_W_RATIO,
            enabled=cfg.MASK_THUMBNAIL
        ),
        ToGray1(),
        transforms.Resize((cfg.INPUT_H, cfg.INPUT_W)),
        transforms.RandomRotation(degrees=4),
        transforms.ColorJitter(brightness=0.08, contrast=0.08),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.25]),
    ])

    val_transform = transforms.Compose([
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

    return train_transform, val_transform


def build_dataloaders(cfg: Config, logger):
    train_tf, val_tf = build_transforms(cfg)

    train_set = datasets.ImageFolder(cfg.DATA_DIR / "train", transform=train_tf)
    val_set = datasets.ImageFolder(cfg.DATA_DIR / "val", transform=val_tf)

    train_set = filter_imagefolder_by_classnames(train_set, cfg.TARGET_CLASSES)
    val_set = filter_imagefolder_by_classnames(val_set, cfg.TARGET_CLASSES)

    hard_val_set = None
    hard_val_loader = None
    hard_val_dir = cfg.DATA_DIR / "hard_val"
    if hard_val_dir.exists():
        hard_val_set = datasets.ImageFolder(hard_val_dir, transform=val_tf)
        hard_val_set = filter_imagefolder_by_classnames(hard_val_set, cfg.TARGET_CLASSES)

    logger.info(f"训练类别: {train_set.classes}")

    train_counter = Counter(train_set.targets)
    val_counter = Counter(val_set.targets)

    logger.info("Train class distribution:")
    for idx, cls_name in enumerate(train_set.classes):
        logger.info(f"  - {cls_name:<20}: {train_counter[idx]}")

    logger.info("Val class distribution:")
    for idx, cls_name in enumerate(val_set.classes):
        logger.info(f"  - {cls_name:<20}: {val_counter[idx]}")

    if hard_val_set is not None:
        hard_counter = Counter(hard_val_set.targets)
        logger.info("Hard Val class distribution:")
        for idx, cls_name in enumerate(hard_val_set.classes):
            logger.info(f"  - {cls_name:<20}: {hard_counter[idx]}")

    # WeightedRandomSampler
    targets = np.array(train_set.targets)
    class_sample_count = np.array([np.sum(targets == t) for t in range(len(train_set.classes))], dtype=np.float32)
    sample_weights = 1.0 / class_sample_count[targets]
    sample_weights = torch.from_numpy(sample_weights).double()

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.BATCH_SIZE,
        sampler=sampler,
        shuffle=False,
        num_workers=cfg.TRAIN_NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(cfg.TRAIN_NUM_WORKERS > 0),
        prefetch_factor=cfg.PREFETCH_FACTOR if cfg.TRAIN_NUM_WORKERS > 0 else None,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.VAL_NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(cfg.VAL_NUM_WORKERS > 0),
        prefetch_factor=cfg.PREFETCH_FACTOR if cfg.VAL_NUM_WORKERS > 0 else None,
    )

    if hard_val_set is not None:
        hard_val_loader = DataLoader(
            hard_val_set,
            batch_size=cfg.BATCH_SIZE,
            shuffle=False,
            num_workers=cfg.VAL_NUM_WORKERS,
            pin_memory=True,
            persistent_workers=(cfg.VAL_NUM_WORKERS > 0),
            prefetch_factor=cfg.PREFETCH_FACTOR if cfg.VAL_NUM_WORKERS > 0 else None,
        )

    return train_loader, val_loader, hard_val_loader, train_set


# =========================================================
# 5. 模型
# =========================================================
def build_model(num_classes: int):
    model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)

    # 把第一层改成单通道，并用原 RGB 权重均值初始化
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


# =========================================================
# 6. Trainer
# =========================================================
class Trainer:
    def __init__(self, cfg, logger, train_loader, val_loader, hard_val_loader, class_names):
        self.cfg = cfg
        self.logger = logger
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.hard_val_loader = hard_val_loader
        self.class_names = class_names

        self.model = build_model(len(class_names)).to(cfg.DEVICE)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.LR,
            weight_decay=cfg.WEIGHT_DECAY
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="max",
            factor=0.5,
            patience=4 # we used 3 in round 1
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=cfg.AMP)

        self.best_f1 = -1.0
        self.best_acc = -1.0
        self.best_epoch = -1
        self.no_improve_epochs = 0
        self.best_report_text = ""

    def train_one_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.cfg.EPOCHS}", leave=False)
        for images, labels in pbar:
            images = images.to(self.cfg.DEVICE, non_blocking=True)
            labels = labels.to(self.cfg.DEVICE, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=self.cfg.AMP):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{self.optimizer.param_groups[0]['lr']:.2e}")

        return running_loss / max(1, len(self.train_loader))

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        all_preds, all_labels = [], []

        for images, labels in self.val_loader:
            images = images.to(self.cfg.DEVICE, non_blocking=True)
            outputs = self.model(images)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.numpy().tolist())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        acc = (all_preds == all_labels).mean().item()
        macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        report_text = classification_report(all_labels, all_preds, target_names=self.class_names, zero_division=0)
        report_dict = classification_report(all_labels, all_preds, target_names=self.class_names, zero_division=0, output_dict=True)
        cm = confusion_matrix(all_labels, all_preds)

        return acc, macro_f1, report_text, report_dict, cm

    def save_checkpoint(self, path: Path, epoch: int, acc: float, macro_f1: float):
        ckpt = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "acc": acc,
            "macro_f1": macro_f1,
            "class_names": self.class_names,
        }
        torch.save(ckpt, path)

    @torch.no_grad()
    def evaluate_loader(self, loader):
        self.model.eval()
        all_preds, all_labels = [], []

        for images, labels in loader:
            images = images.to(self.cfg.DEVICE, non_blocking=True)
            outputs = self.model(images)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.numpy().tolist())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        acc = (all_preds == all_labels).mean().item()
        macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        report_text = classification_report(all_labels, all_preds, target_names=self.class_names, zero_division=0)
        report_dict = classification_report(all_labels, all_preds, target_names=self.class_names, zero_division=0, output_dict=True)
        cm = confusion_matrix(all_labels, all_preds)

        return acc, macro_f1, report_text, report_dict, cm

    def run(self):
        self.cfg.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Training on device: {self.cfg.DEVICE}")
        self.logger.info(f"Classes: {self.class_names}")
        self.logger.info(f"Input size: {self.cfg.INPUT_H}x{self.cfg.INPUT_W}")
        self.logger.info(f"Batch size: {self.cfg.BATCH_SIZE}, Epochs: {self.cfg.EPOCHS}")

        for epoch in range(self.cfg.EPOCHS):
            train_loss = self.train_one_epoch(epoch)
            val_acc, val_macro_f1, report_text, report_dict, cm = self.evaluate_loader(self.val_loader)

            old_lr = self.optimizer.param_groups[0]["lr"]
            self.scheduler.step(val_macro_f1)
            new_lr = self.optimizer.param_groups[0]["lr"]
            if new_lr < old_lr:
                self.logger.info(f"[LR Reduced] {old_lr:.6e} -> {new_lr:.6e}")

            self.logger.info(
                f"Epoch {epoch+1:02d} | Loss: {train_loss:.4f} | "
                f"Val Acc: {val_acc:.4f} | Val Macro-F1: {val_macro_f1:.4f} | "
                f"LR: {new_lr:.6e}"
            )

            for cls_name in self.class_names:
                item = report_dict[cls_name]
                self.logger.info(
                    f"  [{cls_name:<18}] P: {item['precision']:.4f} | "
                    f"R: {item['recall']:.4f} | F1: {item['f1-score']:.4f} | "
                    f"N: {int(item['support'])}"
                )

            self.logger.info("Confusion Matrix:\n" + format_confusion_matrix(cm, self.class_names))

            improved = False
            if val_macro_f1 > self.best_f1:
                self.best_f1 = val_macro_f1
                self.best_epoch = epoch + 1
                self.best_report_text = report_text
                self.no_improve_epochs = 0
                improved = True
                self.save_checkpoint(self.cfg.BEST_F1_PATH, epoch + 1, val_acc, val_macro_f1)
                self.logger.info(f"[F1 Improved] Saved to {self.cfg.BEST_F1_PATH}")
            else:
                self.no_improve_epochs += 1

            if val_acc > self.best_acc:
                self.best_acc = val_acc
                self.save_checkpoint(self.cfg.BEST_ACC_PATH, epoch + 1, val_acc, val_macro_f1)
                self.logger.info(f"[Acc Improved] Saved to {self.cfg.BEST_ACC_PATH}")

            self.save_checkpoint(self.cfg.LATEST_PATH, epoch + 1, val_acc, val_macro_f1)

            if (epoch + 1) % self.cfg.SAVE_EVERY == 0:
                self.logger.info(f"[Periodic Save] latest checkpoint updated at epoch {epoch+1}")

            if improved:
                self.logger.info("Best Classification Report:\n" + self.best_report_text)

            if self.no_improve_epochs >= self.cfg.EARLY_STOPPING_PATIENCE:
                self.logger.info(
                    f"Early stopping triggered: macro-F1 did not improve for "
                    f"{self.cfg.EARLY_STOPPING_PATIENCE} epochs."
                )
                break

        self.logger.info("=" * 60)
        self.logger.info(f"Training finished. Best epoch: {self.best_epoch}")
        self.logger.info(f"Best Macro-F1: {self.best_f1:.4f}")
        self.logger.info(f"Best Acc: {self.best_acc:.4f}")
        self.logger.info("=" * 60)

        if self.hard_val_loader is not None:
            hard_acc, hard_macro_f1, hard_report_text, hard_report_dict, hard_cm = self.evaluate_loader(self.hard_val_loader)

            self.logger.info(
                f"[Hard Val] Acc: {hard_acc:.4f} | Macro-F1: {hard_macro_f1:.4f}"
            )

            for cls_name in self.class_names:
                item = hard_report_dict[cls_name]
                self.logger.info(
                    f"  [Hard {cls_name:<13}] P: {item['precision']:.4f} | "
                    f"R: {item['recall']:.4f} | F1: {item['f1-score']:.4f} | "
                    f"N: {int(item['support'])}"
                )


def main():
    cfg = Config()
    set_seed(cfg.SEED)
    logger = build_logger(cfg.LOG_FILE)

    train_loader, val_loader, hard_val_loader, train_set = build_dataloaders(cfg, logger)

    trainer = Trainer(
        cfg=cfg,
        logger=logger,
        train_loader=train_loader,
        val_loader=val_loader,
        hard_val_loader=hard_val_loader,
        class_names=train_set.classes
    )
    trainer.run()


if __name__ == "__main__":
    main()