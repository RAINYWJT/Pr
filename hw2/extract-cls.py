import torch
import random
import numpy as np
from datasets import load_dataset
from transformers import ViTModel, ViTImageProcessor
from PIL import Image
from tqdm import tqdm

seed = 221300079
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# load VIT model 题目中给的都不行，我重新换了一个，一样的。
model = ViTModel.from_pretrained("WinKawaks/vit-tiny-patch16-224")
# print(model)
# assert 0

processor = ViTImageProcessor.from_pretrained("WinKawaks/vit-tiny-patch16-224")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# 加载 CUB-200-2011 数据集
dataset = load_dataset("pkuHaowei/cub-200-2011-birds")
# print(dataset)

# 选择每个类别的一张图片
selected_images = {}
for example in dataset["train"]:
    # print(example)
    class_id = example["label"]
    if class_id not in selected_images:
        selected_images[class_id] = example["image"]
    # if len(selected_images) == 200:
    #     break

print(len(selected_images))

# 提取 CLS token
F = []
for class_id, image in tqdm(selected_images.items()):
    # print(f"Class ID: {class_id}, Image Type: {type(image)}")
    if isinstance(image, Image.Image):  # 如果已经是 PIL 图像对象
        img = image.convert("RGB")  # 直接转换为 RGB
    else:
        img = Image.open(image).convert("RGB")  # 否则，使用路径打开
    inputs = processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    cls_token = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
    F.append(cls_token)

F = np.array(F) 
np.save("cls_tokens.npy", F) 
