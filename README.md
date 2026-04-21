# PackageClassification
工业场景包裹图像分类项目。  
当前主要完成的是 **4 类 NR 类型分类**，并基于高分辨率灰度图像完成了数据清洗、验证集重构、难例审计和模型优化。
---
## 项目概述
本项目面向工业场景中的包裹识别失败图像（NR 图像）分类任务，当前核心目标是对以下 4 类进行分类：
- `NoPackage`
- `NoWaybill`
- `TruncatedBarcode`
- `WrinkledWaybill`
项目后续可以扩展到完整的 **9 类 NR 类型体系**。
在项目推进过程中，主要做了以下工作：
- 多轮数据集审计与清洗
- 验证集重构与扩充
- 增加极难样本集合 `hard_val`
- 由 RGB 处理切换为灰度单通道输入
- 屏蔽图像左下角缩略图区域
- 提高输入分辨率
- 针对关键混淆对进行定向分析与清洗
---
## 最终结果
### Final model
- Backbone: **ResNet34**
- Input size: **560 × 700**
- Batch size: **16**
- Best epoch: **14**
- Best Accuracy: **0.9258**
- Best Macro-F1: **0.9205**
### 最终验证集结果
```text
                  precision    recall  f1-score   support
       NoPackage       1.00      0.87      0.93        60
       NoWaybill       0.91      0.99      0.95       112
TruncatedBarcode       0.90      0.85      0.87        61
 WrinkledWaybill       0.92      0.94      0.93        50
        accuracy                           0.93       283
       macro avg       0.93      0.91      0.92       283
    weighted avg       0.93      0.93      0.93       283

最终混淆矩阵

GT\Pred             NoPackage       NoWaybill       TruncatedBarcode  WrinkledWaybill
NoPackage           52              2               4                 2
NoWaybill           0               111             0                 1
TruncatedBarcode    0               8               52                1
WrinkledWaybill     0               1               2                 47

⸻

数据集说明

当前 4 类训练集

总计约 9029 张图像。

train/
WrinkledWaybill: 1465
NoPackage: 507
TruncatedBarcode: 1927
NoWaybill: 5130

当前 4 类验证集

总计 283 张图像。

val/
WrinkledWaybill: 50
NoPackage: 60
TruncatedBarcode: 61
NoWaybill: 112

数据集大小

* 约 12.0 GiB

hard_val

用于保存少量极难样本，仅用于额外参考，不参与 best model 选择。
当前规模较小，因此结果仅作定性参考，不作为核心指标。

⸻

类别定义

NoPackage

当图中被识别对象不是包裹主体，而是散落物、杂物、非包裹目标时，标为 NoPackage。
例如：纸巾、螺丝钉、零食、杂物等。

TruncatedBarcode

前提是图中确实存在包裹及其面单/条码区域，只是条码本身不完整或被截断。

⸻

完整 9 类需求体系

按优先级排序如下：

1. None
2. NoPackage
3. NoWaybill
4. BlurryWaybill
5. WrinkledWaybill
6. TruncatedBarcode
7. Reflection
8. InsufficientLighting
9. BlurryFocus

当前完整 9 类数据分布

train

Reflection: 50
WrinkledWaybill: 1465
NoPackage: 507
InsufficientLighting: 46
None: 806
TruncatedBarcode: 1927
BlurryFocus: 72
BlurryWaybill: 45
NoWaybill: 5130

val

Reflection: 4
WrinkledWaybill: 50
NoPackage: 60
InsufficientLighting: 1
None: 9
TruncatedBarcode: 61
BlurryFocus: 5
BlurryWaybill: 1
NoWaybill: 112


模型与训练

当前最佳配置

* Backbone: ResNet34
* Input size: 560 × 700
* Batch size: 16
* Optimizer: AdamW
* 输入形式：灰度单通道
* 预处理：
    * 屏蔽左下角缩略图区域
    * Resize
    * Normalize

为什么高分辨率有效

本任务高度依赖局部细节，例如：

* 条码截断区域只占很小一部分
* 面单褶皱属于局部纹理特征
* 原始图像分辨率很高，而目标区域可能较小

将输入分辨率从较低分辨率提升到 560 × 700 后，模型性能获得了明确提升。

⸻

环境说明

硬件

* GPU: NVIDIA A40
* 训练显存占用：约 3716 MiB

使用 RAM Disk 加速训练

sudo mkdir -p /mnt/ramdisk
sudo mount -t tmpfs -o size=18G tmpfs /mnt/ramdisk
rsync -av /mnt/F/xezrio/PackageClassification/dataset/dataset_9_class /mnt/ramdisk/

⸻

常用命令

进入项目

cd /mnt/F/xezrio/PackageClassification/
tmux
source myenv/bin/activate

文件数量统计

find . -maxdepth 1 -type d -exec sh -c "echo -n '{}: '; find '{}' -maxdepth 1 -type f | wc -l" \;

查看当前目录下子文件夹大小

du -h --max-depth=1

⸻

结论

本项目从一个初始 4 类 baseline 出发，经过多轮数据治理、验证集重构和模型优化，最终达到：

* Accuracy 0.9258
* Macro-F1 0.9205

当前主验证集上模型已经达到较高且较稳定的性能。
后续进一步提升的主要瓶颈不再是基础模型结构，而更可能来自：

* 极难样本
* 标签边界模糊
* 单标签任务对多特征样本的天然限制

---
