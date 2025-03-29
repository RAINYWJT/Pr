from sklearn.decomposition import PCA
import numpy as np

F = np.load("cls_tokens.npy")
pca = PCA(n_components=0.90)  # 保留90
F_pca = pca.fit_transform(F)
np.save("F_pca.npy", F_pca) 

d = F.shape[1]  # 原始维度
d_pca = F_pca.shape[1]  # 降维后维度

explained_variance = np.sum(pca.explained_variance_ratio_)  # 累计解释方差比例

print(f"原始维度: {d}, PCA 降维后维度: {d_pca}")
print(f"实际保留的方差比例: {explained_variance:.6f}")  # 期待 >= 0.90