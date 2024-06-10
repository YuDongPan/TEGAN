# 设计师:Pan YuDong
# 编写者:God's hand
# 时间:2022/12/3 20:21
import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots(2, 2, sharex='row', sharey='col')

x = np.arange(10)
y = x ** 2

ax[0][0].plot(x, y)
ax[0][1].plot(x, y)
ax[1][0].plot(x, y)
ax[1][1].plot(x, y)

# lines, labels = fig.axes[-1].get_legend_handles_labels()

plt.xlabel('Stimulus Target Number')
plt.ylabel('Correlation Coefficient')
plt.show()

