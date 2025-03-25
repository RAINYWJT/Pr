# import numpy as np

# def func(x):
#     term1 = (x+1)/3
#     term2 = np.sqrt((8*x-1)/3)
#     return np.cbrt(x + term1 * term2) + np.cbrt(x - term1 * term2)

# print(func(0.75))
# print(func(13/8))
# print(func(1/2))

# import numpy as np
# from scipy.optimize import fsolve

# def func(x):
#     term1 = (x + 1) / 3
#     term2 = np.sqrt((8 * x - 1) / 3)
#     return np.cbrt(x + term1 * term2) + np.cbrt(x - term1 * term2) - 1  # 使其等于0

# # 选择一个初始猜测值，比如 x0 = 1
# x0 = 1  
# solution = fsolve(func, x0)

# print("解:", solution)




# import torch
# import torch.nn as nn
# import torchvision.models as models
# import time

# def merge_conv_bn(conv_layer, bn_layer):
#     """
#     合并 Conv2d 和 BatchNorm2d 层
#     """
#     assert isinstance(conv_layer, nn.Conv2d)
#     assert isinstance(bn_layer, nn.BatchNorm2d)

#     gamma = bn_layer.weight
#     beta = bn_layer.bias
#     mean = bn_layer.running_mean
#     var = bn_layer.running_var
#     eps = bn_layer.eps

#     # 计算缩放因子
#     scale = gamma / torch.sqrt(var + eps)

#     # 计算新的卷积权重
#     new_weight = conv_layer.weight * scale.view(-1, 1, 1, 1)

#     # 计算新的偏置
#     if conv_layer.bias is not None:
#         new_bias = scale * (conv_layer.bias - mean) + beta
#     else:
#         new_bias = beta - (scale * mean)

#     # 创建新的 Conv 层
#     new_conv = nn.Conv2d(
#         conv_layer.in_channels,
#         conv_layer.out_channels,
#         conv_layer.kernel_size,
#         conv_layer.stride,
#         conv_layer.padding,
#         bias=True
#     )
#     new_conv.weight.data = new_weight
#     new_conv.bias.data = new_bias

#     return new_conv

# def fuse_resnet50(model):
#     """
#     遍历模型，将 Conv + BN 替换为新的 Conv
#     """
#     for name, module in model.named_children():
#         if isinstance(module, nn.Sequential):  # 处理 Bottleneck 结构
#             for bottleneck_name, bottleneck in module.named_children():
#                 if isinstance(bottleneck, models.resnet.Bottleneck):
#                     # 替换所有 Bottleneck 内的 Conv + BN
#                     bottleneck.conv1 = merge_conv_bn(bottleneck.conv1, bottleneck.bn1)
#                     bottleneck.conv2 = merge_conv_bn(bottleneck.conv2, bottleneck.bn2)
#                     bottleneck.conv3 = merge_conv_bn(bottleneck.conv3, bottleneck.bn3)
#         else:
#             fuse_resnet50(module)  # 递归处理子模块
#     return model

# def benchmark(model, input_tensor, num_runs=100):
#     """
#     测试模型推理速度
#     """
#     model.eval()
#     with torch.no_grad():
#         start_time = time.time()
#         for _ in range(num_runs):
#             _ = model(input_tensor)
#         end_time = time.time()
#     return (end_time - start_time) / num_runs

# # 加载 ResNet50 预训练模型
# original_model = models.resnet50(pretrained=True)

# # 复制模型并进行 BN + Conv 融合
# optimized_model = models.resnet50(pretrained=True)
# optimized_model = fuse_resnet50(optimized_model)

# # 生成测试数据
# input_tensor = torch.randn(1, 3, 224, 224)  # 1张 224x224 RGB 图片

# # 测试原始模型推理时间
# original_time = benchmark(original_model, input_tensor)
# print(f"原始 ResNet50 推理时间: {original_time:.6f} 秒")

# # 测试优化后模型推理时间
# optimized_time = benchmark(optimized_model, input_tensor)
# print(f"优化后 ResNet50 推理时间: {optimized_time:.6f} 秒")

# # 计算加速比
# speedup = original_time / optimized_time
# print(f"加速比: {speedup:.2f}x")


import matplotlib.pyplot as plt

# 训练集数据点
points = [(0, 0), (0, 1), (0, -1), (-1, 0), (1, 0), (8, 0), (8, 1), (9, 0)]
labels = ['A', 'A', 'A', 'A', 'A', 'B', 'A', 'B']

# 将 A 类和 B 类数据分开
A_points = [p for p, l in zip(points, labels) if l == 'A']
B_points = [p for p, l in zip(points, labels) if l == 'B']

# 分别获取 x, y 坐标
Ax, Ay = zip(*A_points)
Bx, By = zip(*B_points)

# 画图
plt.figure(figsize=(10, 6))
plt.scatter(Ax, Ay, color='blue', label='Class A', s=100)  # A 类标蓝色
plt.scatter(Bx, By, color='red', label='Class B', s=100)   # B 类标红色

# 标注每个点的坐标
for i, (x, y) in enumerate(points):
    plt.text(x + 0.2, y, f'x{i+1}', fontsize=12, verticalalignment='bottom')

# 添加图例和坐标轴
plt.legend()
plt.axhline(0, color='gray', linestyle='--')
plt.axvline(0, color='gray', linestyle='--')
plt.xlim(-2, 10)
plt.ylim(-2, 2)
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Training Data Distribution')

plt.grid(True)
plt.savefig('distribution.png')
# plt.show()
