```shell

# daily
cd /mnt/F/xezrio/PackageClassification/
tmux
source myenv/bin/activate


# 文件数量统计
find . -maxdepth 1 -type d -exec sh -c "echo -n '{}: '; find '{}' -maxdepth 1 -type f | wc -l" \;


# 挂载 18GB 的内存空间
sudo mkdir -p /mnt/ramdisk
sudo mount -t tmpfs -o size=18G tmpfs /mnt/ramdisk
rsync -av /mnt/F/xezrio/PackageClassification/dataset/dataset_9_class /mnt/ramdisk/


# 当前目录下的子文件夹大小
du -h --max-depth=1



2026-04-18 21:11:20,594 - INFO - Epoch 24 | Loss: 0.0017 | Val Acc: 0.9223 | Val Macro-F1: 0.9171 | LR: 1.250000e-05
2026-04-18 21:11:20,595 - INFO -   [NoPackage         ] P: 1.0000 | R: 0.8833 | F1: 0.9381 | N: 60
2026-04-18 21:11:20,595 - INFO -   [NoWaybill         ] P: 0.9091 | R: 0.9821 | F1: 0.9442 | N: 112
2026-04-18 21:11:20,595 - INFO -   [TruncatedBarcode  ] P: 0.8947 | R: 0.8361 | F1: 0.8644 | N: 61
2026-04-18 21:11:20,595 - INFO -   [WrinkledWaybill   ] P: 0.9038 | R: 0.9400 | F1: 0.9216 | N: 50
2026-04-18 21:11:20,595 - INFO - Confusion Matrix:
GT\Pred             NoPackage       NoWaybill       TruncatedBarco  WrinkledWaybil  
NoPackage           53              3               3               1               
NoWaybill           0               110             0               2               
TruncatedBarcode    0               8               51              2               
WrinkledWaybill     0               0               3               47              
2026-04-18 21:11:20,916 - INFO - [F1 Improved] Saved to checkpoints/checkpoints_resnet34_v6_round_3/best_f1_model.pth
2026-04-18 21:11:21,480 - INFO - Best Classification Report:
                  precision    recall  f1-score   support

       NoPackage       1.00      0.88      0.94        60
       NoWaybill       0.91      0.98      0.94       112
TruncatedBarcode       0.89      0.84      0.86        61
 WrinkledWaybill       0.90      0.94      0.92        50

        accuracy                           0.92       283
       macro avg       0.93      0.91      0.92       283
    weighted avg       0.92      0.92      0.92       283



on washed dataset:

2026-04-20 17:33:47,694 - INFO - Epoch 14 | Loss: 0.0070 | Val Acc: 0.9258 | Val Macro-F1: 0.9205 | LR: 2.500000e-05
2026-04-20 17:33:47,694 - INFO -   [NoPackage         ] P: 1.0000 | R: 0.8667 | F1: 0.9286 | N: 60
2026-04-20 17:33:47,694 - INFO -   [NoWaybill         ] P: 0.9098 | R: 0.9911 | F1: 0.9487 | N: 112
2026-04-20 17:33:47,694 - INFO -   [TruncatedBarcode  ] P: 0.8966 | R: 0.8525 | F1: 0.8739 | N: 61
2026-04-20 17:33:47,694 - INFO -   [WrinkledWaybill   ] P: 0.9216 | R: 0.9400 | F1: 0.9307 | N: 50
2026-04-20 17:33:47,694 - INFO - Confusion Matrix:
GT\Pred             NoPackage       NoWaybill       TruncatedBarco  WrinkledWaybil  
NoPackage           52              2               4               2               
NoWaybill           0               111             0               1               
TruncatedBarcode    0               8               52              1               
WrinkledWaybill     0               1               2               47              
2026-04-20 17:33:48,084 - INFO - [F1 Improved] Saved to checkpoints/checkpoints_resnet34_v6_round_final/best_f1_model.pth
2026-04-20 17:33:48,778 - INFO - [Acc Improved] Saved to checkpoints/checkpoints_resnet34_v6_round_final/best_acc_model.pth
2026-04-20 17:33:49,260 - INFO - Best Classification Report:
                  precision    recall  f1-score   support

       NoPackage       1.00      0.87      0.93        60
       NoWaybill       0.91      0.99      0.95       112
TruncatedBarcode       0.90      0.85      0.87        61
 WrinkledWaybill       0.92      0.94      0.93        50

        accuracy                           0.93       283
       macro avg       0.93      0.91      0.92       283
    weighted avg       0.93      0.93      0.93       283

2026-04-20 18:23:30,569 - INFO - ============================================================
2026-04-20 18:23:30,569 - INFO - Training finished. Best epoch: 14
2026-04-20 18:23:30,570 - INFO - Best Macro-F1: 0.9205
2026-04-20 18:23:30,570 - INFO - Best Acc: 0.9258
2026-04-20 18:23:30,570 - INFO - ============================================================
2026-04-20 18:23:35,237 - INFO - [Hard Val] Acc: 0.2000 | Macro-F1: 0.3000
2026-04-20 18:23:35,237 - INFO -   [Hard NoPackage    ] P: 0.0000 | R: 0.0000 | F1: 0.0000 | N: 1
2026-04-20 18:23:35,237 - INFO -   [Hard NoWaybill    ] P: 0.0000 | R: 0.0000 | F1: 0.0000 | N: 2
2026-04-20 18:23:35,237 - INFO -   [Hard TruncatedBarcode] P: 0.2500 | R: 0.1667 | F1: 0.2000 | N: 6
2026-04-20 18:23:35,237 - INFO -   [Hard WrinkledWaybill] P: 1.0000 | R: 1.0000 | F1: 1.0000 | N: 1



数据集大小约 12.0 GiB

训练集 共 9029 个数据
train/
./WrinkledWaybill: 1465
./NoPackage: 507
./TruncatedBarcode: 1927
./NoWaybill: 5130

验证集 共 283 个数据
val/
./WrinkledWaybill: 50
./NoPackage: 60
./TruncatedBarcode: 61
./NoWaybill: 112

机器参数
训练使用约 3716 MiB 显存


NR类型定义：

NoPackage

当图中被识别对象不是包裹主体，而是散落物、杂物、非包裹目标时，标为 NoPackage。
例如：纸巾、螺丝钉、零食、杂物等。

TruncatedBarcode

前提是图中确实存在包裹及其面单/条码区域，只是条码本身不完整或被截断。


# 需求 (9类，按照优先级排序)

None

NoPackage

NoWaybill

BlurryWaybill

WrinkledWaybill

TruncatedBarcode

Reflection

InsufficientLighting

BlurryFocus



完整 9 类别

train/
./Reflection: 50
./WrinkledWaybill: 1465
./NoPackage: 507
./InsufficientLighting: 46
./None: 806
./TruncatedBarcode: 1927
./BlurryFocus: 72
./BlurryWaybill: 45
./NoWaybill: 5130

val/
./Reflection: 4
./WrinkledWaybill: 50
./NoPackage: 60
./InsufficientLighting: 1
./None: 9
./TruncatedBarcode: 61
./BlurryFocus: 5
./BlurryWaybill: 1
./NoWaybill: 112

**NR类型**

```java
public enum NoReadType {
 /// <summary>
 /// 无类型（默认值）
 /// </summary>
 [Description("无类型")]
 None = 0,

 /// <summary>
 /// 无包裹（摄像头未识别到任何包裹）
 /// </summary>
 [Description("画面无包裹")]
 NoPackage = 1,

 /// <summary>
 /// 无面单（包裹上未检测到面单）
 /// </summary>
 [Description("无面单")]
 NoWaybill = 2,

 /// <summary>
 /// 面单模糊（图像模糊，无法识别条码）
 /// </summary>
 [Description("面单模糊")]
 BlurryWaybill = 3,

 /// <summary>
 /// 面单褶皱（条码因褶皱导致识别失败）
 /// </summary>
 [Description("面单褶皱")]
 WrinkledWaybill = 4,

 /// <summary>
 /// 条码截断（部分条码缺失）
 /// </summary>
 [Description("条码截断")]
 TruncatedBarcode = 5,

 /// <summary>
 /// 反光（条码区域反光，无法识别）
 /// </summary>
 [Description("反光")]
 Reflection = 6,

 /// <summary>
 /// 光线不足（图像过暗，无法识别）
 /// </summary>
 [Description("光线不足")]
 InsufficientLighting = 7
}
```