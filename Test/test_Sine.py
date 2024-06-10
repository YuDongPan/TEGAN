# 设计师:Pan YuDong
# 编写者:God's hand
# 时间:2023/1/20 11:17
import numpy as np
import matplotlib.pyplot as plt

ws = 10.0
Fs = 256
t = np.arange(0, ws, 1.0 / Fs)

fig, ax = plt.subplots(2, 1, figsize=(15, 6), dpi=160)

# sinusoidal signal
f = 2
y1 = np.sin(2 * np.pi * t * f)

# white gaussian noise
y2 = np.random.randn(round(Fs * ws))

ax[0].plot(t, y1, linewidth=5, color='blue')
ax[1].plot(t, y2, color='brown')

plt.show()