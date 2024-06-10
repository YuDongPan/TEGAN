# 设计师:Pan YuDong
# 编写者:God's hand
# 时间:2022/12/13 21:36
import numpy as np

c11 = np.array([[1, 2],
              [3, 4]])

c12 = np.array([[5, 6],
              [7, 8]])

c21 = np.array([[9, 10],
              [11, 12]])

c22 = np.array([[13, 14],
               [15, 16]])

C_up = np.column_stack([c11, c12])
C_down = np.column_stack([c21, c22])

C = np.row_stack([C_up, C_down])

print("C_up:", C_up)

print("C_down:", C_down)

print("C:", C)
