# Experiments

本文档记录本项目主要实验阶段，包括 baseline、联合改进阶段、分辨率提升实验以及最终封板结果。

---

## 1. 初始可用版本（Baseline 0）

这是最早能跑通的 4 类 baseline，可作为整个项目的起点。

### 配置

- **Backbone**: ResNet50
- **输入分辨率**: 224 × 224
- **Batch Size**: 32
- **Learning Rate**: 1e-4
- **数据增强**: 旋转、裁切
- **类别权重**: `1 / 该类别样本数`

### 结果

```text
                  precision    recall  f1-score   support
       NoPackage       0.88      0.50      0.64        14
       NoWaybill       0.77      0.93      0.84       113
TruncatedBarcode       0.56      0.48      0.51        42
 WrinkledWaybill       1.00      0.45      0.62        20

        accuracy                           0.75       189
       macro avg       0.80      0.59      0.65       189
    weighted avg       0.76      0.75      0.73       189
```

### 总结

- **Accuracy**: 0.75
- **Macro-F1**: 0.65

### 说明

该结果作为后续所有版本对比的初始 baseline。

---

## 2. v3

### 配置

- **Backbone**: ResNet18
- **输入分辨率**: 224 × 224
- **Batch Size**: 128
- **Learning Rate**: 1e-4
- **数据增强**: 旋转、裁切
- **Label Smoothing**: 0.1
- **类别权重**: 不再使用 log 平滑

### 结果

```text
                  precision    recall  f1-score   support
       NoPackage       0.90      0.69      0.78        13
       NoWaybill       0.85      0.90      0.87       112
TruncatedBarcode       0.70      0.51      0.59        37
 WrinkledWaybill       0.72      0.95      0.82        19

        accuracy                           0.81       181
       macro avg       0.79      0.76      0.77       181
    weighted avg       0.81      0.81      0.80       181
```

### 总结

- **Accuracy**: 0.81
- **Macro-F1**: 0.77

### 说明

相比初始 baseline，v3 已有明显提升，但 `TruncatedBarcode` 仍然偏弱。

---

## 3. v4

### 配置

- **Backbone**: ResNet18
- **输入分辨率**: 224 × 224
- **Batch Size**: 128
- **训练策略**:
    - 先训练头部 3 个 epoch，`lr = 1e-3`
    - 全量微调时 `lr = 3e-4`
- **Label Smoothing**: 0.03
- **数据增强**: 旋转、裁切
- **类别权重**:
    - 采用 `总样本数 / (类别数 × 该类样本数)` 的平衡权重
    - 再归一化到均值约 1

### 结果

```text
                  precision    recall  f1-score   support
       NoWaybill       0.84      0.89      0.87       112
TruncatedBarcode       0.58      0.57      0.58        37
 WrinkledWaybill       0.70      0.74      0.72        19
       NoPackage       1.00      0.46      0.63        13

        accuracy                           0.78       181
       macro avg       0.78      0.66      0.70       181
    weighted avg       0.78      0.78      0.77       181
```

### 总结

- **Accuracy**: 0.78
- **Macro-F1**: 0.70

### 说明

v4 引入了更复杂的训练策略和权重设计，但整体表现不如 v3 稳定。

---

## 4. v5

### 配置

- 基于 v4
- **Learning Rate**: 2e-4
- 加入 `WeightedRandomSampler`
- 加入 class weight
- 不使用 label smoothing

### 结果

```text
                  precision    recall  f1-score   support
       NoWaybill       0.79      0.93      0.86       112
TruncatedBarcode       0.57      0.32      0.41        37
 WrinkledWaybill       0.65      0.79      0.71        19
       NoPackage       0.83      0.38      0.53        13

        accuracy                           0.75       181
       macro avg       0.71      0.61      0.63       181
    weighted avg       0.74      0.75      0.73       181
```

### 总结

- **Accuracy**: 0.75
- **Macro-F1**: 0.63

### 说明

v5 尝试了采样和权重的组合，但整体效果下降，说明当时的主要瓶颈不在于采样策略本身，而更多在于数据质量和输入信息不足。

---

## 5. v6 第一轮（联合改进阶段）

### 配置

- **Backbone**: ResNet34
- **输入分辨率**: 448 × 560
- **Batch Size**: 32
- **Learning Rate**: 2e-4
- 屏蔽左下角缩略图
- 输入改为单通道灰度图
- 扩充验证集

### 结果

```text
                  precision    recall  f1-score   support
       NoPackage       1.00      0.85      0.92        60
       NoWaybill       0.82      0.93      0.87       100
TruncatedBarcode       0.83      0.72      0.77        60
 WrinkledWaybill       0.85      0.92      0.88        50

        accuracy                           0.86       270
       macro avg       0.88      0.85      0.86       270
    weighted avg       0.87      0.86      0.86       270
```

### 总结

- **Accuracy**: 0.86
- **Macro-F1**: 0.86

### 说明

这是一个非常关键的转折点，表明以下因素共同让整体 pipeline 稳定下来：

- 数据清洗
- 更合理的验证集
- 灰度单通道输入
- 更高输入分辨率

但由于该阶段同时改变了多项因素，因此它更适合被视为**联合改进阶段**，而不是严格的单因素消融实验。

---

## 6. v6 第二轮（清洗数据集）

### 配置

- 在 v6 第一轮基础上继续清洗数据集
- 验证集进一步重构
- 类别定义进一步统一

### 结果

```text
                  precision    recall  f1-score   support
       NoPackage       1.00      0.78      0.88        60
       NoWaybill       0.86      0.96      0.91       112
TruncatedBarcode       0.85      0.87      0.86        61
 WrinkledWaybill       0.92      0.92      0.92        50

        accuracy                           0.89       283
       macro avg       0.91      0.88      0.89       283
    weighted avg       0.90      0.89      0.89       283
```

### 总结

- **Accuracy**: 0.89
- **Macro-F1**: 0.89

### 说明

这一轮的提升主要来自数据集清洗与验证集修正，`TruncatedBarcode` 得到显著改善。

---

## 7. v6 第三轮（提升分辨率）

### 配置

- **输入分辨率**提升到：560 × 700
- **Batch Size**: 16
- **Learning Rate**: 1e-4

### 结果

```text
                  precision    recall  f1-score   support
       NoPackage       1.00      0.88      0.94        60
       NoWaybill       0.91      0.98      0.94       112
TruncatedBarcode       0.89      0.84      0.86        61
 WrinkledWaybill       0.90      0.94      0.92        50

        accuracy                           0.92       283
       macro avg       0.93      0.91      0.92       283
    weighted avg       0.92      0.92      0.92       283
```

### 总结

- **Accuracy**: 0.92
- **Macro-F1**: 0.92

### 说明

该阶段证明了：**提高输入分辨率对本任务是有效的。**

尤其对以下类型更有帮助：

- `TruncatedBarcode`
- `WrinkledWaybill`

---

## 8. v6 Final

### 配置

- 在 v6 第三轮基础上
- 对训练集进行小规模进一步清洗
- 划分极难样本验证集 `hard_val`
- 完成最终封板训练

### 结果

```text
                  precision    recall  f1-score   support
       NoPackage       1.00      0.87      0.93        60
       NoWaybill       0.91      0.99      0.95       112
TruncatedBarcode       0.90      0.85      0.87        61
 WrinkledWaybill       0.92      0.94      0.93        50

        accuracy                           0.93       283
       macro avg       0.93      0.91      0.92       283
    weighted avg       0.93      0.93      0.93       283
```

### 总结

- **Accuracy**: 0.9258
- **Macro-F1**: 0.9205

### `hard_val`

- **Hard Val Acc**: 0.2000
- **Hard Val Macro-F1**: 0.3000

### 说明

当前 `hard_val` 规模太小，仅能作为定性参考。最终模型已经在主验证集上达到了较高且较稳定的性能。

---

## 9. 各阶段结果汇总

| 版本       | 主要改动                                            | Accuracy | Macro-F1 | 说明                       |
| ---------- | --------------------------------------------------- | -------: | -------: | -------------------------- |
| Baseline 0 | ResNet50 初始可用模型                               |     0.75 |     0.65 | 初始 baseline              |
| v3         | ResNet18 + 224 输入 + 权重调整                      |     0.81 |     0.77 | 首次较稳定提升             |
| v4         | 头部训练 + 微调策略 + 新权重形式                    |     0.78 |     0.70 | 效果不如 v3 稳定           |
| v5         | WeightedRandomSampler + class weight                |     0.75 |     0.63 | 说明当时主瓶颈不在采样策略 |
| v6 round 1 | ResNet34 + 448×560 + 灰度 + 缩略图屏蔽 + 扩充验证集 |     0.86 |     0.86 | 联合改进阶段               |
| v6 round 2 | 数据清洗 + 验证集重构                               |     0.89 |     0.89 | 数据治理带来明显提升       |
| v6 round 3 | 提升分辨率到 560×700                                |     0.92 |     0.92 | 分辨率提升明确有效         |
| v6 final   | 小规模再清洗 + 最终封板                             |   0.9258 |   0.9205 | 最终版本                   |

---

## 10. 主要结论

### 10.1 数据治理是关键

本项目的提升并不主要来自单纯堆模型，而是来自：

- 标签审计
- 数据清洗
- 类别定义统一
- 验证集重构

### 10.2 提高分辨率有效

输入分辨率从较低分辨率提升到 **560 × 700** 后，性能获得明确改善。

### 10.3 当前瓶颈已发生变化

在最终阶段，模型的主要瓶颈已经不再是基础结构，而更可能来自：

- 极难样本
- 标签边界模糊
- 单标签任务对多特征样本的天然限制

### 10.4 任务背景与主要难点来源

- 多特征共存但强制单标签（单个数据可以有多种错误，但是需求按照优先级只标注单标签）
- 有自动扩标历史（训练集内含脏数据）
- 数据集类别数据非常不均衡，最大类别数量可达最小类别数量约十倍
- 极难样本（特征难以区分）
- 图像巨大、目标不固定、缩略图干扰（包裹位置并不固定，而且图中可能出现多个包裹影响模型判断）

---

## 11. 结果解读注意事项

- `v6 round 1` 属于多因素联合改动结果，不应被视为严格单变量消融。
- `hard_val` 当前规模很小，不适合用于正式统计对比。
- 最终剩余错误中，有一部分来自任务本身的难度，而不完全是模型能力不足。

---

## 12. 实验阶段划分

### 第一阶段：早期探索

- Baseline 0
- v3
- v4
- v5

这一阶段的主要结论是：**单纯调整采样策略和损失设计，并不能从根本上解决问题。**

### 第二阶段：Pipeline 重构

- v6 第一轮

这一阶段通过以下改动让整体方案稳定下来：

- 更高分辨率
- 灰度输入
- mask 缩略图
- 扩充验证集

### 第三阶段：数据治理驱动提升

- v6 第二轮
- v6 第三轮
- v6 Final

这一阶段主要依靠**定向清洗、验证集修正和更合理的输入分辨率**，将结果逐步从 **0.86** 提升到 **0.92+**。


## 13. 极难样本

可参考 samples/hard_samples。

与条码外形太接近
![alt text](samples/hard_samples/image-20260418144140682.png)
![alt text](samples/hard_samples/image-20260418145138519.png)

人工难以判断
无面单 / 无包裹？类别边界清晰不足
![alt text](samples/hard_samples/image-20260418141655666.png)

容易把带文字的封条识别为面单
![alt text](samples/hard_samples/image-20260418144250751.png)

外形与条码非常接近，但不是快递扫描用的条码
![alt text](samples/hard_samples/image-20260418144519881.png)
![alt text](samples/hard_samples/image-20260418144836139.png)

反光亮度高，容易被误判为条码
![alt text](samples/hard_samples/image-20260418145006290.png)

面单位于图像边缘或面单出现面积太小，缩小分辨率后输入特征难以保留
![alt text](samples/hard_samples/image-20260418145353580.png)
![alt text](samples/hard_samples/image-20260418145423706.png)
![alt text](samples/hard_samples/image-20260418145530372.png)
![alt text](samples/hard_samples/image-20260418145744695.png)
本张甚至难以人工判别
![alt text](samples/hard_samples/image-20260418145956443.png)

包裹数量不止一个，而且被识别包裹并非位于图像中央
![alt text](samples/hard_samples/image-20260418150729839.png)
![alt text](samples/hard_samples/image-20260418151619857.png)

多种特征共存，难以取舍
条码截断 与 面单褶皱
![alt text](samples/hard_samples/image-20260418151013079.png)

非包裹外形类似面单
![alt text](samples/hard_samples/image-20260418152439912.png)
![alt text](samples/hard_samples/image-20260418152539521.png)


## 14. 后续改进方向

- 后续可以专门收集难例，做针对的hard_val训练，提高模型鲁棒性；
- 补充训练集小类样本，可逐步扩充为九类别分类模型
- 采集更大规模、不同流水线传送带上的数据集，可以大幅增强模型在不同物流基地的泛化能力。
