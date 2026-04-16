import os
import sys
import logging
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix

sys.stdout.reconfigure(encoding='utf-8')

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =====================
# 1. 集中配置管理
# =====================
class Config:
    DATA_DIR = Path("./data/dataset")
    CHECKPOINT_DIR = Path("./checkpoints")
    MODEL_SAVE_PATH = CHECKPOINT_DIR / "best_model.pth"
    
    BATCH_SIZE = 32
    EPOCHS = 20
    LR = 1e-4
    NUM_WORKERS = 4  # Windows下请确保在 __main__ 中运行
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 类别映射（会在运行中自动更新）
    CLASSES = []

# =====================
# 2. 数据处理模块
# =====================
def get_data_loaders(config):
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_set = datasets.ImageFolder(config.DATA_DIR / "train", transform=train_transform)
    val_set = datasets.ImageFolder(config.DATA_DIR / "val", transform=val_transform)

    config.CLASSES = train_set.classes
    
    train_loader = DataLoader(train_set, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=True)
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
        
        # 初始化模型 (使用最新API)
        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.model.fc = nn.Linear(self.model.fc.in_features, self.num_classes)
        self.model = self.model.to(config.DEVICE)

        # 计算类别权重 (处理样本不均衡)
        self.criterion = nn.CrossEntropyLoss(weight=self._compute_weights(train_set))
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.LR)
        self.best_acc = 0.0

    def _compute_weights(self, dataset):
        # 使用有效样本数或平滑权重的公式
        labels = [sample[1] for sample in dataset.samples]
        counts = np.bincount(labels)
        # 公式：$weight = \frac{total}{n\_classes \times count}$
        weights = torch.FloatTensor(len(labels) / (self.num_classes * counts)).to(self.config.DEVICE)
        return weights

    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.EPOCHS}")
        
        for images, labels in pbar:
            images, labels = images.to(self.config.DEVICE), labels.to(self.config.DEVICE)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
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
        return acc, all_labels, all_preds

    def run(self):
        self.config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        logger.info(f"Starting training on {self.config.DEVICE}...")
        
        for epoch in range(self.config.EPOCHS):
            train_loss = self.train_epoch(epoch)
            val_acc, labels, preds = self.evaluate()
            
            logger.info(f"Epoch {epoch+1} - Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f}")
            
            if val_acc > self.best_acc:
                self.best_acc = val_acc
                torch.save(self.model.state_dict(), self.config.MODEL_SAVE_PATH)
                logger.info(f"---> Best model saved with Acc: {val_acc:.4f}")

        # 训练结束，输出最终报告
        logger.info("\n" + classification_report(labels, preds, target_names=self.config.CLASSES))


if __name__ == '__main__':
    # 设置 UTF-8 输出
    # if sys.platform == "win32":
    #       sys.stdout.reconfigure(encoding='utf-8')

    cfg = Config()
    
    # 准备数据
    train_dl, val_dl, raw_train_set = get_data_loaders(cfg)
    
    # 初始化并运行
    trainer = Trainer(cfg, train_dl, val_dl, raw_train_set)
    trainer.run()