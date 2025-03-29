# -*- coding: utf-8 -*-
import torch
import numpy as np
import random
from PIL import Image
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from datasets import load_dataset
from transformers import ViTModel, ViTImageProcessor

# 固定随机种子
seed = 221300079
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# 硬件配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 加载CUB数据集
dataset = load_dataset("pkuHaowei/cub-200-2011-birds")
print("数据集结构:", dataset)

# 设置采样参数
TRAIN_SAMPLES_PER_CLASS = 5
TEST_SAMPLES_PER_CLASS = 1

# 划分训练集和测试集
class_data = {}
for example in dataset["train"]:
    class_id = example["label"]
    if class_id not in class_data:
        class_data[class_id] = []
    class_data[class_id].append(example["image"])

train_samples = {}
test_samples = {}
for class_id in class_data:
    random.shuffle(class_data[class_id])
    test_samples[class_id] = class_data[class_id][:TEST_SAMPLES_PER_CLASS]
    train_samples[class_id] = class_data[class_id][TEST_SAMPLES_PER_CLASS:TEST_SAMPLES_PER_CLASS + TRAIN_SAMPLES_PER_CLASS]

print("\n数据统计:")
print(f"总类别数: {len(class_data)}")
print(f"训练样本数: {sum(len(v) for v in train_samples.values())}")
print(f"测试样本数: {sum(len(v) for v in test_samples.values())}")

# 加载ViT模型
model = ViTModel.from_pretrained("WinKawaks/vit-tiny-patch16-224")
processor = ViTImageProcessor.from_pretrained("WinKawaks/vit-tiny-patch16-224")
model = model.to(device).eval()

# 特征提取函数
def extract_features(samples_dict, is_train=True):
    features = []
    labels = []
    desc = "训练集特征" if is_train else "测试集特征"
    for class_id, images in tqdm(samples_dict.items(), desc=desc):
        for img in images:
            inputs = processor(img.convert("RGB"), return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            cls_token = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
            features.append(cls_token)
            labels.append(class_id)
    return np.array(features), np.array(labels)

# 提取特征
train_features, train_labels = extract_features(train_samples)
test_features, test_labels = extract_features(test_samples, is_train=False)

# PCA分析
pca = PCA(n_components=0.90)
pca.fit(train_features)
train_pca = pca.transform(train_features)
test_pca = pca.transform(test_features)

print("\nPCA分析结果:")
print(f"原始维度: {train_features.shape[1]}")
print(f"压缩后维度: {pca.n_components_}")
print(f"保留方差比例: {pca.explained_variance_ratio_.sum():.2%}")

# 分类验证
def evaluate(X_train, y_train, X_test, y_test):
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train, y_train)
    return knn.score(X_test, y_test)

# 使用独立测试集评估
original_acc = evaluate(train_features, train_labels, test_features, test_labels)
pca_acc = evaluate(train_pca, train_labels, test_pca, test_labels)

print("\n分类性能:")
print(f"原始特征测试准确率: {original_acc:.2%}")
print(f"PCA特征测试准确率: {pca_acc:.2%}")
print(f"准确率变化: {pca_acc - original_acc:+.2f}%")


# 模型压缩层实现
class FeatureCompressor(torch.nn.Module):
    def __init__(self, pca_matrix):
        super().__init__()
        self.projection = torch.nn.Linear(
            in_features=pca_matrix.shape[1],
            out_features=pca_matrix.n_components_,
            bias=False
        )
        self.projection.weight.data = torch.tensor(pca_matrix.components_, dtype=torch.float32)
        
    def forward(self, x):
        return self.projection(x)

# 创建压缩模型
compressed_model = torch.nn.Sequential(
    model,  # 原始ViT模型
    FeatureCompressor(pca).to(device)
)

# 验证压缩模型
with torch.no_grad():
    test_output = compressed_model(test_input)