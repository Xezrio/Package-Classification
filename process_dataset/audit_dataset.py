import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from pathlib import Path
import shutil
from tqdm import tqdm

# =====================
# 1. 配置与数据集定义 (保持与训练代码一致)
# =====================
class ImageFolderWithPaths(datasets.ImageFolder):
    def __getitem__(self, index):
        img, label = super().__getitem__(index)
        path = self.imgs[index][0] # 获取文件的物理路径
        return img, label, path
    
def filter_dataset_for_audit(dataset, target_classes):
    # 构建一个 类别名 -> 新索引 的映射
    class_to_idx = {cls_name: i for i, cls_name in enumerate(target_classes)}
    
    new_samples = []
    for path, old_idx in dataset.samples:
        cls_name = dataset.classes[old_idx]
        if cls_name in class_to_idx:
            new_samples.append((path, class_to_idx[cls_name]))
            
    dataset.samples = new_samples
    dataset.targets = [s[1] for s in new_samples]
    dataset.classes = target_classes
    dataset.class_to_idx = class_to_idx
    return dataset

# =====================
# 2. 审计核心逻辑
# =====================
def create_audit_folder(model_path, data_dir, output_audit_dir, config_classes, device):
    output_audit_dir = Path(output_audit_dir)
    if output_audit_dir.exists():
        shutil.rmtree(output_audit_dir) # 每次运行清空旧的审计结果
    output_audit_dir.mkdir(parents=True)

    # A. 加载你训练好的“重型武器”
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, len(config_classes))
    # 注意：确保 model_path 指向你最好的那个 .pth 文件
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()

    # B. 准备数据加载器
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # 建议审计训练集，因为那是“脏数据”的发源地
    full_dataset = ImageFolderWithPaths(data_dir, transform=transform)
    
    # 关键修改：只保留模型认识的那几个类
    dataset = filter_dataset_for_audit(full_dataset, config_classes)
    
    # 现在 dataset 里的索引就严格限制在 0 到 len(config_classes)-1 之间了
    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=16)

    criterion = nn.CrossEntropyLoss(reduction='none')
    all_results = []

    print(f"正在审计数据: {data_dir} ...")
    with torch.no_grad():
        for imgs, labels, paths in tqdm(loader):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            
            losses = criterion(outputs, labels)
            probs = torch.softmax(outputs, dim=1)
            conf, preds = torch.max(probs, dim=1)

            for i in range(len(imgs)):
                all_results.append({
                    'path': paths[i],
                    'true_idx': labels[i].item(),
                    'pred_idx': preds[i].item(),
                    'loss': losses[i].item(),
                    'conf': conf[i].item()
                })

    # C. 按 Loss 降序排列 (Loss 越高，说明标注越可疑)
    all_results.sort(key=lambda x: x['loss'], reverse=True)

    print(f"📂 正在生成可视化审计文件夹 (软链接模式)...")
    # 我们看最离谱的前 500 张
    for item in all_results[:500]:
        true_cls = config_classes[item['true_idx']]
        pred_cls = config_classes[item['pred_idx']]
        
        # 建立：原标签_to_模型预测标签 的文件夹
        # 这样你一眼就能看到：哪些 None 其实是 NoWaybill
        sub_dir = output_audit_dir / f"{true_cls}_to_{pred_cls}"
        sub_dir.mkdir(exist_ok=True, parents=True)
        
        src_path = Path(item['path'])
        # 重命名文件：带上 Loss 值，方便排序查看
        link_name = f"Loss_{item['loss']:.2f}_Conf_{item['conf']:.2f}_{src_path.name}"
        
        # 创建软链接，不占磁盘空间
        os.symlink(src_path.absolute(), sub_dir / link_name)

    print(f"✨ 审计完成！请进入目录查看: {output_audit_dir}")

# =====================
# 3. 运行参数设置
# =====================
if __name__ == '__main__':
    # --- 你需要填写的参数在这里 ---
    MODEL_FILE = "./checkpoints/best_f1_model.pth"   # 你训练好的模型路径
    DATA_TO_AUDIT = "./dataset/dataset_9_class/train" # 你想检查哪个文件夹（建议先查 train）
    AUDIT_RESULT_DIR = "./audit_output"              # 审计结果存在哪
    
    # 类别顺序必须和训练时完全一致！
    CURRENT_CLASSES = [
        'NoPackage', 'NoWaybill', 'None', 'TruncatedBarcode', 'WrinkledWaybill'
    ]
    # ----------------------------

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    create_audit_folder(MODEL_FILE, DATA_TO_AUDIT, AUDIT_RESULT_DIR, CURRENT_CLASSES, DEVICE)