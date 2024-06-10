# 设计师:Pan YuDong
# 编写者:God's hand
# 时间:2022/12/2 20:52
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8), dpi=240)
# matplotlib.rcParams['xtick.direction'] = 'in'
# matplotlib.rcParams['ytick.direction'] = 'in'
font1 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 12}

# plt.rcParams['font.sans-serif'] = ['SimHei']    # 用来正常显示中文标签
# plt.rcParams['axes.unicode_minus'] = False   #显示英文
# plt.rcParams['font.size'] = 13  #设置字体大小，全局有效

subject = 5
augmentation = True
way = 'AUG' if augmentation else 'ORG'

Method1 = f'ITCCA'
Method2 = f'TRCA'
# Method3 = f'EEGNet'
# Method4 = f'C-CNN'

data_itcca = pd.read_csv(f'./Corr_Analysis/{Method1}_{way}_S{subject}.csv')
data_trca = pd.read_csv(f'./Corr_Analysis/{Method2}_{way}_S{subject}.csv')
# data_eegnet = pd.read_csv(f'./Corr_Analysis/{Method3}_{way}_S{subject}.csv')
# data_c_cnn = pd.read_csv(f'./Corr_Analysis/{Method4}_{way}_S{subject}.csv')

data_itcca = data_itcca.to_numpy()
data_trca = data_trca.to_numpy()
# data_eegnet = data_eegnet.to_numpy()
# data_c_cnn = data_c_cnn.to_numpy()

print("data_itcca:", data_itcca)
print("data_trca:", data_trca)
# print("data_eegnet:", data_eegnet)
# print("data_c_cnn:", data_c_cnn)

targets = [9.25, 11.25, 13.25, 9.75, 11.75, 13.75,
           10.25, 12.25, 14.25, 10.75, 12.75, 14.75]

fig = plt.figure(figsize=(12, 7))   # 1200 * 900

for i in range(3):
    for j in range(4):
        n = i * 4 + j + 1
        ax1 = fig.add_subplot(3, 4, n)
        x = np.arange(0, 12)
        y1 = data_itcca[n - 1]
        y2 = data_trca[n - 1]
        # y3 = data_eegnet[n - 1]
        # y4 = data_c_cnn[n - 1]
        x_ticks_bound = [0, 5, 10]
        x_ticks_content = ['0', '5', '10']
        ax1.set_xticks(x_ticks_bound, x_ticks_content, fontproperties='Times New Roman', size=20)
        ax1.set_ylim(-0.4, 1.1, 0.25)
        y_ticks_bound = [(i - 1) * 0.25 for i in range(6)]
        y_ticks_content = [0 if y_ticks == 0 else '%.2f' % y_ticks for y_ticks in y_ticks_bound]
        ax1.set_yticks(y_ticks_bound, y_ticks_content, fontproperties='Times New Roman', size=20)

        # ax2 = ax1.twinx()
        ax1.plot(x, y1, label=f'{Method1}_{way}', color='blue')
        ax1.plot(x, y2, label=f'{Method2}_{way}', color='orange')
        # ax2.plot(x, y3, label=f'{Method3}', color='green', linestyle='--')
        # ax2.plot(x, y4, label=f'{Method4}', color='magenta', linestyle='--')

        # mark ground truth in red point
        point_x = n - 1
        index_y = np.argmax(np.array([y1[n - 1], y2[n - 1]]))
        delta_x, delta_y, point_size = 0.02, 0.01, 50
        if index_y == 0:
            ax1.scatter(point_x - delta_x, y1[n - 1] - delta_y, s=point_size, c='r')
        elif index_y == 1:
            ax1.scatter(point_x - delta_x, y2[n - 1] - delta_y, s=point_size, c='r')
        # elif index_y == 2:
        #     ax2.scatter(point_x - delta_x, y3[n - 1] - delta_y, s=point_size, c='r')
        # elif index_y == 3:
        #     ax2.scatter(point_x - delta_x, y4[n - 1] - delta_y, s=point_size, c='r')


lines1, labels1 = fig.axes[-2].get_legend_handles_labels()
# lines2, labels2 = fig.axes[-1].get_legend_handles_labels()
fig.legend(prop=font1, handles=lines1, labels=labels1)

plt.tight_layout()
plt.show()


