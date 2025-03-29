import torch
import random
import numpy as np
import timm
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

# 设置随机种子
seed = 221300079
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# 加载 ViT-Tiny 模型 (timm 版本)
model = timm.create_model("vit_tiny_patch16_224.augreg_in21k", pretrained=True)
model.eval()

# 获取模型的数据处理配置
data_config = timm.data.resolve_model_data_config(model)
transforms = timm.data.create_transform(**data_config, is_training=False)

# 选择计算设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 加载 CUB-200-2011 数据集
dataset = load_dataset("pkuHaowei/cub-200-2011-birds")

# 选择每个类别的一张图片
selected_images = {}
for example in dataset["train"]:
    class_id = example["label"]
    if class_id not in selected_images:
        selected_images[class_id] = example["image"]

print(f"Selected {len(selected_images)} images.")

# 提取 CLS token
F = []
for class_id, image in tqdm(selected_images.items()):
    if isinstance(image, Image.Image):  # 如果已经是 PIL 图像
        img = image.convert("RGB")
    else:
        img = Image.open(image).convert("RGB")  # 否则打开图片

    # 预处理图片
    img_tensor = transforms(img).unsqueeze(0).to(device)

    # 提取 CLS token
    with torch.no_grad():
        outputs = model.forward_features(img_tensor)
    cls_token = outputs[:, 0, :].squeeze().cpu().numpy()

    F.append(cls_token)

# 保存特征
F = np.array(F)
np.save("cls_tokens.npy", F)
print("CLS tokens saved to cls_tokens.npy")
