import os
import json
import time
import copy
import random
import logging
from pathlib import Path
from collections import Counter

import numpy as np
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
    CHECKPOINT_DIR = Path("./checkpoints_resnet18_4cls_vfinal")
    LOG_FILE = CHECKPOINT_DIR / "train.log"

    TARGET_CLASSES = [
        "NoWaybill",
        "TruncatedBarcode",
        "WrinkledWaybill",
        "NoPackage",
    ]

    IMAGE_SIZE = 224
    BATCH_SIZE = 128
    EPOCHS = 40
    LR = 2e-4
    WEIGHT_DECAY = 8e-5
    NUM_WORKERS = min(32, max(2, (os.cpu_count() or 8) - 2))

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    AMP = torch.cuda.is_available()
    SEED = 42

    EARLY_STOPPING_PATIENCE = 10
    SAVE_EVERY = 5

    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    BEST_F1_PATH = CHECKPOINT_DIR / "best_f1_model.pth"
    BEST_ACC_PATH = CHECKPOINT_DIR / "best_acc_model.pth"
    LATEST_PATH = CHECKPOINT_DIR / "latest_model.pth"
    BEST_REPORT_JSON = CHECKPOINT_DIR / "best_report.json"


# =========================================================
# 2. 基础工具
# =========================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def build_logger(log_file: Path):
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("train_logger")
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
# 3. 数据集处理
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
        transforms.Resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)),
        transforms.RandomRotation(degrees=8),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.05),
        transforms.ToTensor(),
        transforms.Normalize(cfg.MEAN, cfg.STD),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(cfg.MEAN, cfg.STD),
    ])

    return train_transform, val_transform


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

    # 使用 WeightedRandomSampler，而不是 loss weight
    targets = np.array(train_set.targets)
    class_sample_count = np.array([np.sum(targets == t) for t in range(len(train_set.classes))], dtype=np.float32)

    # 每个样本的采样权重 = 1 / 该类样本数
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
# 4. 模型与训练器
# =========================================================
def build_model(num_classes: int):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(in_features, num_classes)
    )
    return model


class Trainer:
    def __init__(self, cfg: Config, logger, train_loader, val_loader, class_names):
        self.cfg = cfg
        self.logger = logger
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.class_names = class_names
        self.num_classes = len(class_names)

        self.model = build_model(self.num_classes).to(cfg.DEVICE)

        # 不使用 class weight，不使用 label smoothing
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
            patience=4,
        )

        self.scaler = torch.amp.GradScaler("cuda", enabled=cfg.AMP)

        self.best_f1 = -1.0
        self.best_acc = -1.0
        self.best_epoch = -1
        self.best_report_text = ""
        self.best_report_dict = None
        self.best_cm = None
        self.no_improve_epochs = 0

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
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                lr=f"{self.optimizer.param_groups[0]['lr']:.2e}"
            )

        return running_loss / max(1, len(self.train_loader))

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()

        all_preds = []
        all_labels = []

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
        report_text = classification_report(
            all_labels,
            all_preds,
            target_names=self.class_names,
            zero_division=0
        )
        report_dict = classification_report(
            all_labels,
            all_preds,
            target_names=self.class_names,
            zero_division=0,
            output_dict=True
        )
        cm = confusion_matrix(all_labels, all_preds)

        return acc, macro_f1, report_text, report_dict, cm

    def save_checkpoint(self, path: Path, epoch: int, acc: float, macro_f1: float):
        ckpt = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "acc": acc,
            "macro_f1": macro_f1,
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
            train_loss = self.train_one_epoch(epoch)
            val_acc, val_macro_f1, report_text, report_dict, cm = self.evaluate()

            self.scheduler.step(val_macro_f1)
            current_lr = self.optimizer.param_groups[0]["lr"]

            self.logger.info(
                f"Epoch {epoch+1:02d} | Loss: {train_loss:.4f} | "
                f"Val Acc: {val_acc:.4f} | Val Macro-F1: {val_macro_f1:.4f} | "
                f"LR: {current_lr:.6e}"
            )

            # 打印每类指标
            for cls_name in self.class_names:
                item = report_dict[cls_name]
                self.logger.info(
                    f"  [{cls_name:<18}] "
                    f"P: {item['precision']:.4f} | "
                    f"R: {item['recall']:.4f} | "
                    f"F1: {item['f1-score']:.4f} | "
                    f"N: {int(item['support'])}"
                )

            self.logger.info("Confusion Matrix:\n" + format_confusion_matrix(cm, self.class_names))

            improved = False

            if val_macro_f1 > self.best_f1:
                self.best_f1 = val_macro_f1
                self.best_epoch = epoch + 1
                self.best_report_text = report_text
                self.best_report_dict = report_dict
                self.best_cm = cm.tolist()
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
                with open(self.cfg.BEST_REPORT_JSON, "w", encoding="utf-8") as f:
                    json.dump({
                        "best_epoch": self.best_epoch,
                        "best_macro_f1": self.best_f1,
                        "best_acc": self.best_acc,
                        "report": self.best_report_dict,
                        "confusion_matrix": self.best_cm,
                        "class_names": self.class_names
                    }, f, ensure_ascii=False, indent=2)

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
        self.logger.info("=== Best Model Classification Report ===")
        self.logger.info("\n" + self.best_report_text)


# =========================================================
# 5. 主函数
# =========================================================
def main():
    cfg = Config()
    set_seed(cfg.SEED)
    logger = build_logger(cfg.LOG_FILE)

    train_loader, val_loader, train_set, val_set = build_dataloaders(cfg, logger)

    trainer = Trainer(
        cfg=cfg,
        logger=logger,
        train_loader=train_loader,
        val_loader=val_loader,
        class_names=train_set.classes
    )
    trainer.run()


if __name__ == "__main__":
    main()