cd /mnt/F/xezrio/PackageClassification/
tmux
source myenv/bin/activate
# daily


# 文件数量统计
find . -maxdepth 1 -type d -exec sh -c "echo -n '{}: '; find '{}' -maxdepth 1 -type f | wc -l" \;


# 挂载 18GB 的内存空间
sudo mkdir -p /mnt/ramdisk
sudo mount -t tmpfs -o size=18G tmpfs /mnt/ramdisk
rsync -av /mnt/F/xezrio/PackageClassification/dataset/dataset_9_class /mnt/ramdisk/

# 删除目标目录多余文件
mkdir -p /mnt/ramdisk/dataset_9_class
rsync -av --delete /mnt/F/xezrio/PackageClassification/dataset/dataset_9_class/ /mnt/ramdisk/dataset_9_class/

# 当前目录下的子文件夹大小
du -h --max-depth=1


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