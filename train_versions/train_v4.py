import os
import math
import json
import random
import logging
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import (
    classification_report,
    f1_score,
    confusion_matrix,
    precision_recall_fscore_support
)
from tqdm import tqdm


# =========================================================
# 0. 基础配置
# =========================================================
class Config:
    # 数据目录结构:
    # dataset_root/
    #   train/
    #     NoWaybill/
    #     TruncatedBarcode/
    #     WrinkledWaybill/
    #     NoPackage/
    #   val/
    #     ...
    DATA_DIR = Path("/mnt/ramdisk/dataset_9_class")
    CHECKPOINT_DIR = Path("./checkpoints_resnet18_4cls")
    LOG_PATH = CHECKPOINT_DIR / "train.log"

    # 只保留四类
    TARGET_CLASSES = [
        "NoWaybill",
        "TruncatedBarcode",
        "WrinkledWaybill",
        "NoPackage",
    ]

    # 训练参数
    IMAGE_SIZE = 224
    BATCH_SIZE = 128
    EPOCHS = 40
    FREEZE_EPOCHS = 3            # 前几轮只训练分类头
    LR_HEAD = 1e-3               # 只训头部时更大一点
    LR_FINE = 3e-4               # 全量微调时
    WEIGHT_DECAY = 1e-4
    LABEL_SMOOTHING = 0.03
    NUM_WORKERS = min(32, max(2, (os.cpu_count() or 8) - 2))
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    AMP = torch.cuda.is_available()
    SEED = 42

    # 早停
    EARLY_STOP_PATIENCE = 10

    # 输入归一化（ImageNet）
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    # 可选：类别权重上限，防止小类权重过猛
    MAX_CLASS_WEIGHT = 6.0

    # 可选：梯度裁剪
    GRAD_CLIP_NORM = 5.0

    # 保存
    BEST_F1_PATH = CHECKPOINT_DIR / "best_f1_model.pth"
    BEST_ACC_PATH = CHECKPOINT_DIR / "best_acc_model.pth"
    LATEST_PATH = CHECKPOINT_DIR / "latest_model.pth"
    BEST_META_PATH = CHECKPOINT_DIR / "best_metrics.json"


# =========================================================
# 1. 随机种子与日志
# =========================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 为了结果更稳定；如果你想极限提速，可关闭 deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_logger(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("trainer")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger


# =========================================================
# 2. 数据集过滤
# =========================================================
def filter_imagefolder_by_classnames(dataset: datasets.ImageFolder, target_classnames):
    """
    只保留 target_classnames 中的类别，并重新映射 label 到 [0, n-1]
    """
    old_class_to_idx = dataset.class_to_idx
    missing = [c for c in target_classnames if c not in old_class_to_idx]
    if missing:
        raise ValueError(f"以下类别在数据集中不存在: {missing}")

    old_idx_to_new_idx = {
        old_class_to_idx[class_name]: new_idx
        for new_idx, class_name in enumerate(target_classnames)
    }

    new_samples = []
    for path, old_label in dataset.samples:
        if old_label in old_idx_to_new_idx:
            new_samples.append((path, old_idx_to_new_idx[old_label]))

    dataset.samples = new_samples
    dataset.targets = [label for _, label in new_samples]
    dataset.classes = list(target_classnames)
    dataset.class_to_idx = {name: i for i, name in enumerate(target_classnames)}

    return dataset


def build_transforms(cfg: Config):
    # 工业缺陷任务：避免随机 crop / flip 破坏语义
    train_tf = transforms.Compose([
        transforms.Resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.05),
        transforms.RandomRotation(degrees=5),
        transforms.ToTensor(),
        transforms.Normalize(cfg.MEAN, cfg.STD),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(cfg.MEAN, cfg.STD),
    ])
    return train_tf, val_tf


def build_dataloaders(cfg: Config, logger):
    train_tf, val_tf = build_transforms(cfg)

    train_set = datasets.ImageFolder(cfg.DATA_DIR / "train", transform=train_tf)
    val_set = datasets.ImageFolder(cfg.DATA_DIR / "val", transform=val_tf)

    train_set = filter_imagefolder_by_classnames(train_set, cfg.TARGET_CLASSES)
    val_set = filter_imagefolder_by_classnames(val_set, cfg.TARGET_CLASSES)

    logger.info(f"训练类别: {train_set.classes}")

    train_counter = Counter(train_set.targets)
    val_counter = Counter(val_set.targets)

    logger.info("Train class distribution:")
    for idx, cls_name in enumerate(train_set.classes):
        logger.info(f"  - {cls_name:<20}: {train_counter[idx]}")

    logger.info("Val class distribution:")
    for idx, cls_name in enumerate(val_set.classes):
        logger.info(f"  - {cls_name:<20}: {val_counter[idx]}")

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(cfg.NUM_WORKERS > 0),
    )

    val_loader = DataLoader(
        val_set,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(cfg.NUM_WORKERS > 0),
    )

    return train_loader, val_loader, train_set, val_set


# =========================================================
# 3. 模型、损失、优化器
# =========================================================
def build_model(num_classes: int):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features, num_classes)
    )
    return model


def set_trainable_layers(model: nn.Module, train_backbone: bool):
    if train_backbone:
        for p in model.parameters():
            p.requires_grad = True
    else:
        for p in model.parameters():
            p.requires_grad = False
        for p in model.fc.parameters():
            p.requires_grad = True


def build_class_weights(train_targets, num_classes: int, cfg: Config):
    counts = np.bincount(train_targets, minlength=num_classes).astype(np.float32)

    # 使用“总样本数 / (类别数 * 该类样本数)”的平衡权重形式
    total = counts.sum()
    weights = total / (num_classes * counts)

    # 防止极端权重太大
    weights = np.clip(weights, a_min=None, a_max=cfg.MAX_CLASS_WEIGHT)

    # 归一化到均值约 1，便于训练更稳
    weights = weights / weights.mean()

    return torch.tensor(weights, dtype=torch.float32)


def build_optimizer(model: nn.Module, lr: float, weight_decay: float):
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def build_scheduler(optimizer, total_epochs, warmup_epochs=2, min_lr_ratio=0.05):
    """
    Warmup + Cosine
    """
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        progress = (epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


# =========================================================
# 4. 评估工具
# =========================================================
@torch.no_grad()
def evaluate(model, loader, device, class_names):
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []
    running_loss = 0.0
    num_batches = 0

    # 这里只是为了统一接口，evaluate 不再重复算 loss 可选略过
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        preds = outputs.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())
        all_probs.extend(probs.cpu().numpy().tolist())
        num_batches += 1

    acc = (np.array(all_preds) == np.array(all_labels)).mean().item()
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    report_dict = classification_report(
        all_labels,
        all_preds,
        target_names=class_names,
        zero_division=0,
        output_dict=True
    )
    report_text = classification_report(
        all_labels,
        all_preds,
        target_names=class_names,
        zero_division=0
    )
    cm = confusion_matrix(all_labels, all_preds)

    per_cls_p, per_cls_r, per_cls_f1, per_cls_support = precision_recall_fscore_support(
        all_labels, all_preds, labels=list(range(len(class_names))), zero_division=0
    )

    per_class_metrics = []
    for i, cls_name in enumerate(class_names):
        per_class_metrics.append({
            "class_name": cls_name,
            "precision": float(per_cls_p[i]),
            "recall": float(per_cls_r[i]),
            "f1": float(per_cls_f1[i]),
            "support": int(per_cls_support[i]),
        })

    return {
        "acc": float(acc),
        "macro_f1": float(macro_f1),
        "report_text": report_text,
        "report_dict": report_dict,
        "confusion_matrix": cm.tolist(),
        "per_class_metrics": per_class_metrics,
    }


def format_confusion_matrix(cm, class_names):
    lines = []
    header = "GT\\Pred".ljust(20) + "".join([c[:12].ljust(14) for c in class_names])
    lines.append(header)
    for i, row in enumerate(cm):
        line = class_names[i][:18].ljust(20)
        for v in row:
            line += str(v).ljust(14)
        lines.append(line)
    return "\n".join(lines)


# =========================================================
# 5. Trainer
# =========================================================
class Trainer:
    def __init__(self, cfg: Config, logger, train_loader, val_loader, train_set):
        self.cfg = cfg
        self.logger = logger
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.class_names = train_set.classes
        self.num_classes = len(self.class_names)

        self.model = build_model(self.num_classes).to(cfg.DEVICE)

        # 类别权重
        class_weights = build_class_weights(train_set.targets, self.num_classes, cfg).to(cfg.DEVICE)
        self.logger.info(f"Class weights: {class_weights.detach().cpu().numpy().round(4).tolist()}")

        self.criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=cfg.LABEL_SMOOTHING
        )

        # 初始：先冻结 backbone
        self.backbone_unfrozen = False
        set_trainable_layers(self.model, train_backbone=False)
        self.optimizer = build_optimizer(self.model, cfg.LR_HEAD, cfg.WEIGHT_DECAY)
        self.scheduler = build_scheduler(self.optimizer, total_epochs=cfg.FREEZE_EPOCHS, warmup_epochs=1)

        self.scaler = torch.amp.GradScaler("cuda", enabled=cfg.AMP)

        self.best_f1 = -1.0
        self.best_acc = -1.0
        self.best_epoch = -1
        self.no_improve_epochs = 0

    def maybe_unfreeze_backbone(self, epoch):
        if (not self.backbone_unfrozen) and epoch >= self.cfg.FREEZE_EPOCHS:
            self.logger.info("Unfreezing backbone and switching to fine-tuning mode...")
            self.backbone_unfrozen = True

            set_trainable_layers(self.model, train_backbone=True)
            self.optimizer = build_optimizer(self.model, self.cfg.LR_FINE, self.cfg.WEIGHT_DECAY)
            remaining_epochs = max(1, self.cfg.EPOCHS - epoch)
            self.scheduler = build_scheduler(self.optimizer, total_epochs=remaining_epochs, warmup_epochs=2)

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

            if self.cfg.GRAD_CLIP_NORM is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    max_norm=self.cfg.GRAD_CLIP_NORM
                )

            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{self.optimizer.param_groups[0]['lr']:.2e}")

        return running_loss / max(1, len(self.train_loader))

    def save_checkpoint(self, path: Path, epoch: int, metrics: dict):
        ckpt = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "metrics": metrics,
            "class_names": self.class_names,
            "config": {
                "image_size": self.cfg.IMAGE_SIZE,
                "target_classes": self.cfg.TARGET_CLASSES,
            }
        }
        torch.save(ckpt, path)

    def run(self):
        self.cfg.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Training on device: {self.cfg.DEVICE}")
        self.logger.info(f"Classes: {self.class_names}")
        self.logger.info(f"Batch size: {self.cfg.BATCH_SIZE}, Epochs: {self.cfg.EPOCHS}")

        for epoch in range(self.cfg.EPOCHS):
            self.maybe_unfreeze_backbone(epoch)

            train_loss = self.train_one_epoch(epoch)
            val_metrics = evaluate(self.model, self.val_loader, self.cfg.DEVICE, self.class_names)

            self.scheduler.step()

            acc = val_metrics["acc"]
            macro_f1 = val_metrics["macro_f1"]
            current_lr = self.optimizer.param_groups[0]["lr"]

            self.logger.info(
                f"Epoch {epoch+1:02d} | "
                f"Loss: {train_loss:.4f} | "
                f"Val Acc: {acc:.4f} | "
                f"Val Macro-F1: {macro_f1:.4f} | "
                f"LR: {current_lr:.6e}"
            )

            # 打印每类指标
            for item in val_metrics["per_class_metrics"]:
                self.logger.info(
                    f"  [{item['class_name']:<18}] "
                    f"P: {item['precision']:.4f} | "
                    f"R: {item['recall']:.4f} | "
                    f"F1: {item['f1']:.4f} | "
                    f"N: {item['support']}"
                )

            # 打印混淆矩阵
            cm_text = format_confusion_matrix(np.array(val_metrics["confusion_matrix"]), self.class_names)
            self.logger.info("Confusion Matrix:\n" + cm_text)

            improved = False

            if macro_f1 > self.best_f1:
                self.best_f1 = macro_f1
                self.best_epoch = epoch + 1
                self.no_improve_epochs = 0
                improved = True
                self.save_checkpoint(self.cfg.BEST_F1_PATH, epoch + 1, val_metrics)
                self.logger.info(f"[F1 Improved] Saved to {self.cfg.BEST_F1_PATH}")
            else:
                self.no_improve_epochs += 1

            if acc > self.best_acc:
                self.best_acc = acc
                self.save_checkpoint(self.cfg.BEST_ACC_PATH, epoch + 1, val_metrics)
                self.logger.info(f"[Acc Improved] Saved to {self.cfg.BEST_ACC_PATH}")

            # latest
            self.save_checkpoint(self.cfg.LATEST_PATH, epoch + 1, val_metrics)

            # 最佳信息另存 json
            if improved:
                meta = {
                    "best_epoch": self.best_epoch,
                    "best_macro_f1": self.best_f1,
                    "best_acc_so_far": self.best_acc,
                    "class_names": self.class_names,
                    "report": val_metrics["report_dict"],
                    "confusion_matrix": val_metrics["confusion_matrix"],
                }
                with open(self.cfg.BEST_META_PATH, "w", encoding="utf-8") as f:
                    json.dump(meta, f, ensure_ascii=False, indent=2)

                self.logger.info("Best Classification Report:\n" + val_metrics["report_text"])

            if self.no_improve_epochs >= self.cfg.EARLY_STOP_PATIENCE:
                self.logger.info(
                    f"Early stopping triggered: macro-F1 did not improve for "
                    f"{self.cfg.EARLY_STOP_PATIENCE} epochs."
                )
                break

        self.logger.info("=" * 60)
        self.logger.info(f"Training finished. Best epoch: {self.best_epoch}")
        self.logger.info(f"Best Macro-F1: {self.best_f1:.4f}")
        self.logger.info(f"Best Acc: {self.best_acc:.4f}")
        self.logger.info("=" * 60)


# =========================================================
# 6. 主函数
# =========================================================
def main():
    cfg = Config()
    set_seed(cfg.SEED)
    logger = build_logger(cfg.LOG_PATH)

    train_loader, val_loader, train_set, val_set = build_dataloaders(cfg, logger)
    trainer = Trainer(cfg, logger, train_loader, val_loader, train_set)
    trainer.run()


if __name__ == "__main__":
    main()