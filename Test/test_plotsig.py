# 设计师:Pan YuDong
# 编写者:God's hand
# 时间:2022/12/10 15:46
import math
import numpy as np
import matplotlib.pyplot as plt


def plot_sig(xstart, xend, ystart, yend, sig):
    for i in range(len(xstart)):
        # plot vertical line
        x = np.ones((2)) * xstart[i]
        y = np.arange(ystart[i], yend[i], yend[i] - ystart[i] - 0.1)
        plt.plot(x, y, label="$y$", color="black", linewidth=2)

        # plot horizontal line
        x = np.arange(xstart[i], xend[i] + 0.1, xend[i] - xstart[i])
        y = yend[i] + 0 * x
        plt.plot(x, y, label="$y$", color="black", linewidth=2)

        # plot annotator
        x0 = (xstart[i] + xend[i]) / 2
        y0 = yend[i]
        plt.annotate(r'%s' % sig, xy=(x0, y0), xycoords='data', xytext=(-15, +1),
                     textcoords='offset points', fontsize=16, color="red")
        x = np.ones((2)) * xend[i]
        y = np.arange(ystart[i], yend[i], yend[i] - ystart[i] - 0.1)
        plt.plot(x, y, label="$y$", color="black", linewidth=2)
        plt.ylim(0, math.ceil(max(yend) + 4))  # 使用plt.ylim设置y坐标轴范围

    plt.show()


plot_sig([0.42, 1.42], [1.42, 2.42], [30, 20], [30.8, 20.8], '***')
