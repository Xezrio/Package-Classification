import json
import sys
from collections import Counter

sys.stdout.reconfigure(encoding='utf-8')

with open("./data/annotations_350x.json", "r", encoding="utf-8") as f:
    data = json.load(f)

labels = []

for item in data:
    if item.get("annotations"):
        for ann in item["annotations"]:
            for result in ann.get("result", []):
                if "value" in result and "choices" in result["value"]:
                    labels.extend(result["value"]["choices"])

counter = Counter(labels)

print("类别统计：")
for k, v in counter.items():
    print(k, v)

print("总标注数:", sum(counter.values()))