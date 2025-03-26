from sklearn.decomposition import PCA
import numpy as np

F = np.load("cls_tokens.npy")
pca = PCA(n_components=0.90)  # 保留90
F_pca = pca.fit_transform(F)

d = F.shape[1]  # 原始维度
d_pca = F_pca.shape[1]  # 降维后维度
percentage = (d_pca / d) * 100 

print(f"原始维度: {d}, PCA 降维后维度: {d_pca}, 占比: {percentage:.2f}%")
