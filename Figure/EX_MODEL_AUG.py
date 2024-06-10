# 设计师:Pan YuDong
# 编写者:God's hand
# 时间:2022/7/7 19:33
import matplotlib
import matplotlib.pyplot as plt
'''Direction'''
# dataset = 'Direction'
# mean_itcca = [38.94, 61.34]
# var_itcca = [13.72, 22.73]
#
# mean_itcca_aug = [77.20, 86.11]
# var_itcca_aug = [17.03, 17.91]
#
# mean_trca = [61.47, 81.00]
# var_trca = [24.17, 21.72]
#
# mean_trca_aug = [77.13, 85.66]
# var_trca_aug = [17.91, 17.04]
#
# mean_eegnet = [39.79, 52.66]
# var_eegnet = [12.05, 19.00]
#
# mean_eegnet_aug = [69.42, 84.34]
# var_eegnet_aug = [19.82, 12.59]
#
#
# mean_ssvepnet = [63.94, 79.93]
# var_ssvepnet = [13.75, 21.32]
#
# mean_ssvepnet_aug = [78.73, 86.22]
# var_ssvepnet_aug = [16.81, 14.58]

'''Dial'''
dataset = 'Dial'
mean_itcca = [42.86, 67.61]
var_itcca = [19.11, 24.88]

mean_itcca_aug = [65.28, 85.00]
var_itcca_aug = [21.59, 16.28]

mean_trca = [66.44, 83.28]
var_trca = [21.95, 16.28]

mean_trca_aug = [66.67, 83.68]
var_trca_aug = [20.37, 16.90]

mean_eegnet = [35.63, 55.28]
var_eegnet = [12.60, 23.80]

mean_eegnet_aug = [60.12, 76.23]
var_eegnet_aug = [20.43, 21.12]


mean_ssvepnet = [79.23, 88.62]
var_ssvepnet = [19.83, 16.17]

mean_ssvepnet_aug = [69.92, 88.08]
var_ssvepnet_aug = [21.57, 12.15]


# set x axis element for bar
interval = 0.9
N = 2
a = [i * interval for i in range(N)]
b = [i * interval + 0.06 for i in range(N)]
c = [i * interval + 0.12 for i in range(N)]
d = [i * interval + 0.18 for i in range(N)]
e = [i * interval + 0.24 for i in range(N)]
f = [i * interval + 0.30 for i in range(N)]
g = [i * interval + 0.36 for i in range(N)]
h = [i * interval + 0.42 for i in range(N)]


# set size of figure
plt.figure(figsize=(16, 8), dpi=240)
matplotlib.rcParams['xtick.direction'] = 'in'
matplotlib.rcParams['ytick.direction'] = 'in'
font1 = {'family': 'Times New Roman',
'weight' : 'normal',
'size': 18}

# add grid for y axis
# plt.rc('axes', axisbelow=True)
# plt.grid(axis='y', linestyle='--', linewidth=1, alpha=0.4)

# plot the bars
plt.bar(a, mean_itcca, yerr=var_itcca, error_kw={'ecolor': '0.2', 'capsize': 6}, width=0.05, color="#cc0000", label='ITCCA')
plt.bar(b, mean_itcca_aug, yerr=var_itcca_aug, error_kw={'ecolor': '0.2', 'capsize': 6}, color="#ff0000", width=0.05, label='ITCCA_AUG')
plt.bar(c, mean_trca, yerr=var_trca, error_kw={'ecolor': '0.2', 'capsize': 6}, color="#ffcc33", width=0.05, label='TRCA')
plt.bar(d, mean_trca_aug, yerr=var_trca_aug, error_kw={'ecolor': '0.2', 'capsize': 6}, color="#ff9933", width=0.05, label='TRCA_AUG')
plt.bar(e, mean_eegnet, yerr=var_eegnet, error_kw={'ecolor': '0.2', 'capsize': 6}, color="#33ff66", width=0.05, label='EEGNet')
plt.bar(f, mean_eegnet_aug, yerr=var_eegnet_aug, error_kw={'ecolor': '0.2', 'capsize': 6}, color="#33ff33", width=0.05, label='EEGNet_AUG')
plt.bar(g, mean_ssvepnet, yerr=var_ssvepnet, error_kw={'ecolor': '0.2', 'capsize': 6}, color="#33ffff", width=0.05, label='SSVEPNet')
plt.bar(h, mean_ssvepnet_aug, yerr=var_ssvepnet_aug, error_kw={'ecolor': '0.2', 'capsize': 6}, color="#3333ff",width=0.05, label='SSVEPNet_AUG')


# set value range for axis x and axis y
plt.ylim(0, 108)

# set ticks for x axis and y axis
x_ticks_bound = [i * interval + 0.20 for i in range(N)]
x_ticks_content = ['IntraC-2vs8(0.5S)',  'IntraC-2vs8(1.0S)']
plt.xticks(x_ticks_bound, x_ticks_content, fontproperties='Times New Roman', size=18)


y_ticks_bound = [i * 10 for i in range(11)]
y_ticks_content = [str(i * 10) for i in range(11)]
plt.yticks(y_ticks_bound, y_ticks_content, fontproperties='Times New Roman', size=18)


# set label for data
plt.legend(prop=font1, framealpha=1.0, loc='lower center')
plt.ylabel('Accuracy(%)', fontproperties='Times New Roman', size=18)


ax = plt.gca()
ax.spines['bottom'].set_linewidth(2)  ### set the thickness of the bottom axis
ax.spines['left'].set_linewidth(2)   #### set the thickness of the left axis
ax.spines['right'].set_linewidth(2)  #### set the thickness of the right axis
ax.spines['top'].set_linewidth(2)    #### set the thickness of the top axis
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
plt.title(f'Results of Classification Accuracy Across All Subjects On {dataset}', fontsize=15)

plt.show()