import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import shutil
from tqdm import tqdm

# --- 配置 ---
MODEL_PATH = "checkpoints/best_f1_model.pth"
SOURCE_DIR = "/mnt/F/xezrio/PackageClassification/dataset/raw"
OUTPUT_DIR = "auto_label_results"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ['BlurryFocus', 'BlurryWaybill', 'InsufficientLighting', 'NoPackage', 
           'NoWaybill', 'None', 'Reflection', 'TruncatedBarcode', 'WrinkledWaybill']

# 预处理 (必须与训练时完全一致)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict():
    # 1. 加载模型
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    # 注意：如果训练时用了 AMP 或 DataParallel，load_state_dict 通常没问题
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    # 2. 创建目录
    for cls in CLASSES:
        os.makedirs(os.path.join(OUTPUT_DIR, "high_conf", cls), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "low_conf"), exist_ok=True)

    # 3. 递归获取所有图片路径
    all_image_paths = []
    for root, dirs, files in os.walk(SOURCE_DIR):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                all_image_paths.append(os.path.join(root, f))
    
    print(f"找到待预测图片总数: {len(all_image_paths)}")

    # 4. 遍历预测
    with torch.no_grad():
        for img_path in tqdm(all_image_paths, desc="Predicting"):
            try:
                # 获取文件名和父目录名，防止重名覆盖
                img_name = os.path.basename(img_path)
                parent_folder = os.path.basename(os.path.dirname(img_path))
                save_filename = f"{parent_folder}_{img_name}"

                img = Image.open(img_path).convert('RGB')
                img_t = transform(img).unsqueeze(0).to(DEVICE)
                
                outputs = model(img_t)
                probs = torch.softmax(outputs, dim=1)
                conf, pred = torch.max(probs, 1)
                
                target_cls = CLASSES[pred.item()]
                conf_val = conf.item()

                # 5. 根据置信度分流
                if conf_val > 0.95:
                    # 高置信度：按类别存放
                    dest = os.path.join(OUTPUT_DIR, "high_conf", target_cls, save_filename)
                    shutil.copy2(img_path, dest) 
                else:
                    # 低置信度：带上预测标签和分数存放，方便人工快速过滤
                    dest = os.path.join(OUTPUT_DIR, "low_conf", f"{target_cls}_{conf_val:.2f}_{save_filename}")
                    shutil.copy2(img_path, dest)
                    
            except Exception as e:
                # 打印错误但继续执行
                print(f"\nError processing {img_path}: {e}")

if __name__ == "__main__":
    predict()