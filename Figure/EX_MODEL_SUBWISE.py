# Designer:Ethan Pan
# Coder:God's hand
# Time:2024/2/22 17:58
import matplotlib
import matplotlib.pyplot as plt


method = 1

'''Direction dataset'''
interval = 0.9
shift = 0.35
N = 54
sub_space = 5

mean_itcca = [32.50, 40.75, 29.75, 67.50, 30.50, 41.00, 51.00, 26.75, 31.75, 39.00,
              81.25, 38.25, 26.75, 30.00, 32.00, 45.00, 43.00, 74.75, 33.75, 29.25,
              39.00, 24.50, 24.00, 30.00, 33.50, 28.00, 42.00, 31.25, 29.75, 63.25,
              65.50, 61.25, 30.25, 25.50, 30.00, 80.50, 54.75, 45.75, 39.25, 31.00,
              46.25, 34.25, 56.50, 51.75, 27.75, 32.25, 25.00, 34.75, 63.50, 44.00,
              26.00, 37.50, 33.75, 28.00]

mean_itcca_aug = [79.75, 76.75, 57.00, 94.75, 56.25, 86.00, 97.25, 71.00, 72.00, 75.00,
                  99.50, 89.25, 58.25, 83.50, 77.50, 92.25, 79.50, 96.00, 70.50, 54.50,
                  77.00, 57.75, 30.00, 58.00, 80.25, 60.00, 83.25, 68.25, 77.00, 97.25,
                  95.00, 97.25, 76.50, 32.50, 48.50, 99.75, 94.25, 89.25, 92.25, 68.25,
                  87.50, 72.75, 97.50, 93.00, 53.50, 70.25, 32.25, 88.00, 98.50, 90.00,
                  46.50, 86.50, 77.50, 45.25]

mean_eegnet = [43.25, 32.00, 28.75, 46.75, 29.75, 42.25, 45.00, 27.75, 45.50, 40.25,
               63.75, 44.25, 34.25, 34.50, 30.50, 44.50, 35.75, 56.50, 38.00, 33.75,
               46.50, 42.50, 33.50, 27.25, 46.75, 39.75, 41.25, 30.50, 45.50, 34.25,
               44.75, 44.25, 28.75, 31.25, 28.00, 61.75, 44.50, 35.50, 47.75, 28.25,
               37.00, 29.25, 62.50, 42.25, 45.75, 31.75, 29.50, 53.75, 61.25, 41.25,
               27.25, 38.00, 29.00, 30.00]

mean_eegnet_aug = [80.00, 76.75, 57.00, 94.75, 55.00, 83.75, 97.00, 69.25, 69.50, 75.75,
                  99.25, 88.75, 58.50, 82.50, 77.50, 92.00, 79.00, 96.25, 69.25, 54.50,
                  77.25, 57.25, 30.25, 57.25, 80.25, 59.25, 82.00, 68.00, 77.00, 97.25,
                  94.00, 97.25, 75.75, 33.75, 47.50, 99.25, 92.75, 88.75, 91.50, 67.50,
                  88.00, 72.50, 97.50, 92.00, 54.00, 69.50, 31.75, 86.00, 98.50, 88.75,
                  46.75, 85.75, 77.75, 46.00]


'''Dial dataset'''
# interval = 0.9
# shift = 0.29
# N = 10
# sub_space = 1
#
# mean_itcca = [20.28, 18.19, 36.53, 57.64, 54.86, 65.69, 36.11, 71.67, 45.14, 18.33]
# mean_itcca_aug = [45.14, 33.61, 67.78, 90.42, 97.08, 91.81, 82.64, 90.14, 82.78, 50.97]
#
# mean_eegnet = [23.77, 19.70, 30.57, 51.30, 29.40, 50.90, 45.83, 58.43, 25.57, 18.97]
# mean_eegnet_aug = [43.63, 34.37, 64.93, 90.57, 96.87, 91.10, 81.40, 89.63, 82.53, 49.77]


# set x axis element for bar
a = [i * interval for i in range(N)]
b = [i * interval + shift for i in range(N)]

# set size of figure
plt.figure(figsize=(20, 12), dpi=240)
matplotlib.rcParams['xtick.direction'] = 'in'
matplotlib.rcParams['ytick.direction'] = 'in'
font1 = {'family': 'Times New Roman',
'weight' : 'normal',
'size': 25}


# add grid for y axis
# plt.rc('axes', axisbelow=True)
# plt.grid(axis='y', linestyle='--', linewidth=1, alpha=0.4)

# plot the bars
mean_org = mean_itcca if method == 0 else mean_eegnet
mean_aug = mean_itcca_aug if method == 0 else mean_eegnet_aug

# color_org = "#99FFFF" if method == 0 else "#FFCC33"
# color_aug = "#99CCFF" if method == 0 else "#FF9933"

color_org = (255 / 255, 217 / 255, 102 / 255) if method == 0 else (112 / 255, 173 / 255, 71 / 255)
color_aug = (0 / 255, 112 / 255, 192 / 255) if method == 0 else (196 / 255, 89 / 255, 17 / 255)

label_org = "ITCCA" if method == 0 else "EEGNet"
label_aug = "ITCCA_AUG" if method == 0 else "EEGNet_AUG"

plt.bar(a, mean_org, width=shift + 0.01, color=color_org, label=label_org)
plt.bar(b, mean_aug, width=shift + 0.01, color=color_aug, label=label_aug)


# set value range for axis x and axis y
plt.ylim(0, 115)

# set ticks for x axis and y axis
x_ticks_bound = [i * interval + (shift + 0.01) / 2 for i in range(0, N, sub_space)]
x_ticks_content = [f'{i + 1}' for i in range(0, N, sub_space)]
plt.xticks(x_ticks_bound, x_ticks_content, fontproperties='Times New Roman', size=28)


y_ticks_bound = [i * 10 for i in range(11)]
y_ticks_content = [str(i * 10) for i in range(11)]
plt.yticks(y_ticks_bound, y_ticks_content, fontproperties='Times New Roman', size=28)


# set label for data
plt.legend(prop=font1, ncol=2, framealpha=1.0, loc='upper center')
plt.xlabel('Subject', fontproperties='Times New Roman', size=30)
plt.ylabel('Accuracy(%)', fontproperties='Times New Roman', size=30)


ax = plt.gca()
ax.spines['bottom'].set_linewidth(2)  ### set the thickness of the bottom axis
ax.spines['left'].set_linewidth(2)   #### set the thickness of the left axis
ax.spines['right'].set_linewidth(2)  #### set the thickness of the right axis
ax.spines['top'].set_linewidth(2)    #### set the thickness of the top axis
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.show()