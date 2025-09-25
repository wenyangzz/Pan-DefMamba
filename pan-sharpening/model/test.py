import torch

# 创建一个形状为 (B=2, D=3, N=4, 2) 的张量
x = torch.arange(2 * 2 * 2 * 2).view(2, 3, 4, 2)
print("原始形状:", x.shape)
print("原始张量:\n", x)

# 使用 flatten(2) 展平从第2维开始的所有维度
y = x.flatten(2)
print("展平后形状:", y.shape)
print("展平后张量:\n", y)