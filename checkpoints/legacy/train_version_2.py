import os
import sys
import logging
import torch
import torch.nn as nn
import numpy as np
from collections import Counter
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix

sys.stdout.reconfigure(encoding='utf-8')

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("app.log", encoding="utf-8")
    ]
)
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
    
    # --- 新增：最少样本数阈值 ---
    # 只有训练集样本数大于等于此值的类别，才会被用于训练和测试
    MIN_SAMPLES = 20 
    
    CLASSES = []

# =====================
# 2. 数据处理与动态过滤
# =====================
def filter_dataset(dataset, valid_classes_dict):
    """
    动态过滤数据集，只保留样本充足的类别，并重新映射标签 ID
    """
    # 过滤 samples 并更新标签
    new_samples = [(path, valid_classes_dict[label]) for path, label in dataset.samples if label in valid_classes_dict]
    dataset.samples = new_samples
    dataset.targets = [s[1] for s in new_samples]
    
    # 重新生成 classes 和 class_to_idx
    new_classes = [dataset.classes[old_idx] for old_idx in sorted(valid_classes_dict.keys())]
    dataset.classes = new_classes
    dataset.class_to_idx = {cls_name: i for i, cls_name in enumerate(new_classes)}
    
    return dataset

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

    # --- 新增：动态剔除极少数样本类别 ---
    counts = Counter([s[1] for s in train_set.samples])
    valid_classes_dict = {}
    new_idx = 0
    for old_idx, count in sorted(counts.items()):
        if count >= config.MIN_SAMPLES:
            valid_classes_dict[old_idx] = new_idx
            new_idx += 1
            
    if not valid_classes_dict:
        raise ValueError(f"没有类别的训练样本数达到阈值 {config.MIN_SAMPLES}！请调低阈值或补充数据。")

    train_set = filter_dataset(train_set, valid_classes_dict)
    val_set = filter_dataset(val_set, valid_classes_dict) # 验证集同步剔除
    
    config.CLASSES = train_set.classes
    logger.info(f"Loaded {len(config.CLASSES)} categories: {config.CLASSES}")

    # --- 新增：处理类别不均衡的 Sampler ---
    targets = np.array(train_set.targets)
    class_sample_count = np.array([len(np.where(targets == t)[0]) for t in range(len(config.CLASSES))])
    weight = 1. / class_sample_count
    samples_weight = torch.from_numpy(weight[targets]).double()
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

    # 注意：使用 sampler 时，必须将 shuffle 设为 False
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
        
        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.model.fc = nn.Linear(self.model.fc.in_features, self.num_classes)
        self.model = self.model.to(config.DEVICE)

        # 既然用了 Sampler，Loss 就不需要再加 weight 了，否则会重复校正
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.LR, weight_decay=1e-4)
        
        # --- 新增：学习率调度器 (如果连续 3 个 Epoch 验证准确率没提升，学习率减半) ---
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=3)
        self.best_acc = 0.0

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
            
            # 更新学习率
            self.scheduler.step(val_acc)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            logger.info(f"Epoch {epoch+1} - Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f} | LR: {current_lr:.6f}")
            
            if val_acc > self.best_acc:
                self.best_acc = val_acc
                torch.save(self.model.state_dict(), self.config.MODEL_SAVE_PATH)
                logger.info(f"---> Best model saved with Acc: {val_acc:.4f}")

        logger.info("\n" + classification_report(labels, preds, target_names=self.config.CLASSES, zero_division=0))

if __name__ == '__main__':
    cfg = Config()
    
    # 准备数据
    train_dl, val_dl, raw_train_set = get_data_loaders(cfg)
    
    # 初始化并运行
    trainer = Trainer(cfg, train_dl, val_dl, raw_train_set)
    trainer.run()