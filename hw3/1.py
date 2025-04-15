import numpy as np

np.random.seed(42)

# 生成20个10维标准高斯样本
dim = 10
samples = np.random.randn(20, dim)

# 计算每个样本的l2范数
norms = np.linalg.norm(samples, axis=1)

# 计算统计量
mean_norm = np.mean(norms)
min_norm = np.min(norms)
max_norm = np.max(norms)

print(f"10维样本的l2范数统计:")
print(f"均值: {mean_norm:.4f}")
print(f"最小值: {min_norm:.4f}")
print(f"最大值: {max_norm:.4f}")

dimensions = [100, 1000, 10000, 100000]

for dim in dimensions:
    samples = np.random.randn(20, dim)
    norms = np.linalg.norm(samples, axis=1)
    
    print(f"\n{dim}维样本的l2范数统计:")
    # print(np.mean(norms)**2 - dim)
    # print(f"均值: {np.mean(norms):.4f}")
    # print(f"最小值: {np.min(norms):.4f}") 
    # print(f"最大值: {np.max(norms):.4f}")
    print(f'波动: {(np.max(norms) - np.min(norms))/np.mean(norms):.4f}')