import os
import logging
import torch
import torch.nn as nn
import numpy as np
from collections import Counter
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, f1_score

# 配置日志 (Linux 下通常配合 nohup 或 tmux 使用，直接输出标准流和文件)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("train_resnet18.log", encoding="utf-8")
    ]
)
logger = logging.getLogger(__name__)

# =====================
# 1. 集中配置管理
# =====================
class Config:
    DATA_DIR = Path("./dataset/dataset_9_class")
    CHECKPOINT_DIR = Path("./checkpoints")
    MODEL_SAVE_PATH = CHECKPOINT_DIR / "best_resnet18.pth"
    
    BATCH_SIZE = 16         # ResNet18 较轻量，显存允许的话可以加大 Batch Size 提升速度
    EPOCHS = 200            # 建议适当增加 Epoch 观察收敛情况
    LR = 1e-4               # AdamW 配合 ResNet18 可以稍微提高初始学习率
    
    # 动态获取 Linux 服务器的 CPU 核心数，保留 2 个核心给系统
    NUM_WORKERS = min(8, max(1, os.cpu_count() - 2 if os.cpu_count() else 4)) # 最多8核心
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    MIN_SAMPLES = 0 # 最小样本数
    CLASSES = []

# =====================
# 2. 数据处理与动态过滤 (保持不变)
# =====================
def filter_dataset(dataset, valid_classes_dict):
    new_samples = [(path, valid_classes_dict[label]) for path, label in dataset.samples if label in valid_classes_dict]
    dataset.samples = new_samples
    dataset.targets = [s[1] for s in new_samples]
    
    new_classes = [dataset.classes[old_idx] for old_idx in sorted(valid_classes_dict.keys())]
    dataset.classes = new_classes
    dataset.class_to_idx = {cls_name: i for i, cls_name in enumerate(new_classes)}
    return dataset

def get_data_loaders(config):
    # 工业场景面单：增加一点灰度转换以防有些图是RGB混入的，统一步调
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)), # 面单通常占主体，缩小裁剪范围防止裁掉特征
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15), # 增加轻微旋转应对传送带角度
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_set = datasets.ImageFolder(config.DATA_DIR / "train", transform=train_transform)
    val_set = datasets.ImageFolder(config.DATA_DIR / "val", transform=val_transform)

    counts = Counter([s[1] for s in train_set.samples])
    valid_classes_dict = {}
    new_idx = 0
    for old_idx, count in sorted(counts.items()):
        if count >= config.MIN_SAMPLES:
            valid_classes_dict[old_idx] = new_idx
            new_idx += 1
            
    if not valid_classes_dict:
        raise ValueError(f"没有类别的训练样本数达到阈值 {config.MIN_SAMPLES}！")

    train_set = filter_dataset(train_set, valid_classes_dict)
    val_set = filter_dataset(val_set, valid_classes_dict)
    
    config.CLASSES = train_set.classes
    logger.info(f"Loaded {len(config.CLASSES)} categories: {config.CLASSES}")

    # 类别权重计算
    targets = np.array(train_set.targets)
    class_sample_count = np.array([len(np.where(targets == t)[0]) for t in range(len(config.CLASSES))])
    weight = 1. / class_sample_count
    samples_weight = torch.from_numpy(weight[targets]).double()
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

    # Linux 下 pin_memory=True 能加速主内存到显存的传输
    train_loader = DataLoader(train_set, batch_size=config.BATCH_SIZE, sampler=sampler, num_workers=config.NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True)
    
    return train_loader, val_loader, train_set

# =====================
# 3. 训练器引擎
# =====================
class Trainer:
    def __init__(self, config, train_loader, val_loader, train_set):
            self.config = config
            self.train_loader = train_loader
            self.val_loader = val_loader
            self.num_classes = len(config.CLASSES)
            
            # 1. 加载预训练模型
            self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            self.model.fc = nn.Linear(self.model.fc.in_features, self.num_classes)
            self.model = self.model.to(config.DEVICE)

            self.scaler = torch.amp.GradScaler('cuda')

            # 2. 处理类别不平衡 (根据你提供的统计数据)
            # 顺序必须严格对应 CLASSES: 
            # ['BlurryFocus', 'BlurryWaybill', 'InsufficientLighting', 'NoPackage', 
            #  'NoWaybill', 'None', 'Reflection', 'TruncatedBarcode', 'WrinkledWaybill']
            counts = torch.tensor([72, 10, 6, 432, 5241, 806, 50, 2139, 1430], dtype=torch.float)
            
            # 计算权重：使用 1/log(count) 平滑处理，防止权重过激
            # 或者使用简单的反比例权重：weights = 1.0 / counts
            weights = 1.0 / torch.log1p(counts) 
            weights = weights / weights.sum() * self.num_classes  # 归一化到类别数
            weights = weights.to(config.DEVICE)
            
            print(f"--- 应用类别权重 ---")
            for cls, w in zip(config.CLASSES, weights):
                print(f"Label: {cls:<20} | Weight: {w.item():.4f}")

            # 3. 使用带权重的损失函数
            self.criterion = nn.CrossEntropyLoss(weight=weights)
            
            # 4. 优化器与调度器
            # 注意：由于是 V2 训练且数据量增大，建议 lr 设为 1e-5 左右
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.LR, weight_decay=1e-4)
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='max', factor=0.5, patience=4
            )
            
            self.best_macro_f1 = 0.0
            self.best_acc = 0.0
            self.best_report = ""

    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.EPOCHS}", leave=False)
        
        for images, labels in pbar:
            images, labels = images.to(self.config.DEVICE), labels.to(self.config.DEVICE)
            
            self.optimizer.zero_grad()
            
            # 1. 开启混合精度正向传播
            with torch.amp.autocast('cuda'):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
            
            # 2. 使用 scaler 进行反向传播（防止梯度下溢）
            self.scaler.scale(loss).backward()
            
            # 3. 更新参数
            self.scaler.step(self.optimizer)
            
            # 4. 更新 scaler 缩放因子
            self.scaler.update()
            
            running_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        
        return running_loss / len(self.train_loader)

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        all_preds, all_labels = [], []
        
        for images, labels in self.val_loader:
            images = images.to(self.config.DEVICE)
            outputs = self.model(images)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            
        acc = (np.array(all_preds) == np.array(all_labels)).mean()
        # 计算 Macro F1，这对不平衡数据集更公平
        macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        report = classification_report(all_labels, all_preds, target_names=self.config.CLASSES, zero_division=0)
        return acc, macro_f1, report

    def run(self):
        self.config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        logger.info(f"Starting training on {self.config.DEVICE} with ResNet18...")
        
        patience_counter = 0  # 早停计数器
        max_patience = 12     # 如果连续 12 个 epoch F1 都不涨，就停
        
        for epoch in range(self.config.EPOCHS):
            train_loss = self.train_epoch(epoch)
            val_acc, val_macro_f1, report = self.evaluate()
            
            # 使用 Macro F1 更新学习率调度器
            self.scheduler.step(val_macro_f1)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            logger.info(f"Epoch {epoch+1:02d} - Loss: {train_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_macro_f1:.4f} | LR: {current_lr:.6f}")
            
            improved = False

            # 策略 A: 优先保存 Macro F1 最高的模型 (为了识别小类错误)
            if val_macro_f1 > self.best_macro_f1:
                self.best_macro_f1 = val_macro_f1
                self.best_report = report # 记录最好的分类报告
                torch.save(self.model.state_dict(), self.config.CHECKPOINT_DIR / "best_f1_model.pth")
                logger.info(f"      [F1 Improved] ---> Saved best_f1_model.pth")
                improved = True
                patience_counter = 0 # 重置早停计数
            else:
                patience_counter += 1

            # 策略 B: 同时保存 Accuracy 最高的模型 (为了整体稳定性)
            if val_acc > self.best_acc:
                self.best_acc = val_acc
                torch.save(self.model.state_dict(), self.config.CHECKPOINT_DIR / "best_acc_model.pth")
                logger.info(f"      [Acc Improved] ---> Saved best_acc_model.pth")

            # 每 5 个 Epoch 保存一个最新的备份，防止服务器意外断电
            if (epoch + 1) % 5 == 0:
                torch.save(self.model.state_dict(), self.config.CHECKPOINT_DIR / "latest_model.pth")

            # 早停检查
            if patience_counter >= max_patience:
                logger.info(f"Early stopping triggered! No improvement in F1 for {max_patience} epochs.")
                break

        logger.info("\n" + "="*20 + " Final Best Report (F1) " + "="*20)
        logger.info("\n" + self.best_report)
        logger.info(f"Training Complete. Best F1: {self.best_macro_f1:.4f}, Best Acc: {self.best_acc:.4f}")

        # 训练结束后打印的是表现最好那一轮的报告
        logger.info("\n=== Best Model Classification Report ===")
        logger.info("\n" + self.best_report)

if __name__ == '__main__':
    cfg = Config()
    train_dl, val_dl, raw_train_set = get_data_loaders(cfg)
    trainer = Trainer(cfg, train_dl, val_dl, raw_train_set)
    trainer.run()