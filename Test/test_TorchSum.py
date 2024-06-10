# 设计师:Pan YuDong
# 编写者:God's hand
# 时间:2022/10/20 16:03
import torch

X = torch.randn((3, 1))
Y = torch.randn((3))
Z = X + Y
print("X:", X)
print("Y:", Y)
print("X+Y:", X+Y)