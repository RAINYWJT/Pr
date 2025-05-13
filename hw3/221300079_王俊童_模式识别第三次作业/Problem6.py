import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt


# 1. 生成混合高斯分布数据
def generate_data(n_samples):
    samples1 = np.random.normal(0, 1, int(0.25 * n_samples))
    samples2 = np.random.normal(6, 2, int(0.75 * n_samples))
    data = np.concatenate([samples1, samples2])
    np.random.shuffle(data)
    return data.astype(np.float32)

train_data = generate_data(10000) # 训练集
val_data = generate_data(1000) # 验证集
test_data = generate_data(1000) # 测试集

train_tensor = torch.tensor(train_data.tolist(), dtype=torch.float32).unsqueeze(1)
test_tensor = torch.tensor(test_data.tolist(), dtype=torch.float32).unsqueeze(1)

# test_tensor = torch.from_numpy(test_data).unsqueeze(1)

# 2. 定义压缩模型（集成加性噪声离散化）
class CompressionModel(nn.Module):
    def __init__(self, latent_dim=2, lambda_val=1.0):
        super().__init__()
        self.lambda_val = lambda_val
        self.encoder = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, latent_dim)  # 输出连续值
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    # 根据第e问题的实现
    def quantize(self, y, training=False):
        if training:
            # 训练时：添加均匀噪声 U(-0.5, 0.5)
            noise = torch.rand_like(y) - 0.5
            return y + noise
        else:
            # 测试时：直接取整
            return torch.round(y)
    
    def forward(self, x, training=False):
        y = self.encoder(x)
        y_quantized = self.quantize(y, training)  
        x_recon = self.decoder(y_quantized)
        return x_recon, y_quantized
    
    def compute_loss(self, x):
        # I= D + \lambda R
        x_recon, y_quantized = self(x, training=True)
        # 重构误差 (MSE)
        D = torch.mean((x - x_recon) ** 2)
        # 码率 (信息熵估计)
        hist = torch.histc(y_quantized, bins=20, min=-10, max=10)
        prob = hist / torch.sum(hist)
        R = -torch.sum(prob * torch.log2(prob + 1e-10))  # 防止log(0)
        # 总损失
        return D + self.lambda_val * R

# 3. 训练函数
def train_model(lambda_val=1.0):
    model = CompressionModel(lambda_val=lambda_val)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_loader = DataLoader(TensorDataset(train_tensor), batch_size=128, shuffle=True)
    
    for epoch in range(50):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            x = batch[0]
            optimizer.zero_grad()
            loss = model.compute_loss(x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch}, Loss: {total_loss / len(train_loader):.4f}")
    return model

# 4. 实验不同 lambda 值
lambdas = [0.1, 1, 10]
results = {}

for lam in lambdas:
    print(f"\nTraining with λ={lam}...")
    model = train_model(lambda_val=lam)
    
    # 测试评估
    model.eval()
    with torch.no_grad():
        x_recon, y_quantized = model(test_tensor, training=False)
        mse = torch.mean((test_tensor - x_recon) ** 2).item()
        # 计算熵
        hist = torch.histc(y_quantized, bins=20, min=-10, max=10)
        prob = hist / torch.sum(hist)
        entropy = -torch.sum(prob * torch.log2(prob + 1e-10)).item()
        results[lam] = {'MSE': mse, 'Entropy': entropy, 'J': mse + lam * entropy}

# 5. 可视化结果
plt.figure(figsize=(12, 4))
for i, lam in enumerate(lambdas):
    plt.subplot(1, 3, i + 1)
    plt.hist(test_data, bins=50, alpha=0.5, label='Original')
    with torch.no_grad():
        x_recon, _ = model(test_tensor, training=False)
        plt.hist(x_recon.numpy().flatten(), bins=50, alpha=0.5, label='Reconstructed')
    plt.title(f"λ={lam}\nMSE={results[lam]['MSE']:.2f}, Entropy={results[lam]['Entropy']:.2f}")
    plt.legend()
plt.tight_layout()
plt.savefig('compression_results.png')
plt.show()

# 打印结果
print("\nFinal Results:")
for lam in lambdas:
    print(f"λ={lam}: J={results[lam]['J']:.2f}, MSE={results[lam]['MSE']:.2f}, Entropy={results[lam]['Entropy']:.2f}")