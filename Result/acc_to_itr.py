# 设计师:Pan YuDong
# 编写者:God's hand
# 时间:2023/2/28 20:01
import math
import numpy as np
import pandas as pd

dataset = 'Direction'
method = 'EEGNet'
ratio = 3
UD = 0
win_size = 0.5
num_fold = 1
M = 4 if dataset == 'Direction' else 12

if ratio == -1:
    proportion = 'Training-Free'
elif ratio == 1:
    proportion = '8vs2'
    num_fold = 5
elif ratio == 2:
    proportion = '5vs5'
    num_fold = 2
elif ratio == 3:
     proportion = '2vs8'
     num_fold = 5
else:
    proportion = 'N-1vs1'

if UD == -1:
    val_way = 'Unsupervised'
elif UD == 0:
    val_way = 'PerSubject'
else:
    val_way = 'CrossSubject'

csv_file = f'./{dataset}/{method}/{proportion}_{val_way}_Classification_Result({str(win_size)}S).csv'

acc_df = pd.read_csv(csv_file)

print("acc_df:", acc_df)

def cal_itr(M, P, T):
    gaze_shift_time = 0.5
    T += gaze_shift_time
    if P < 0 or 1 < P:
        print('Accuracy need to be between 0 and 1.')
        exit()

    elif P < 1 / M:
        print('The ITR might be incorrect because the accuracy < chance level.')
        itr = 0

    elif P == 1:
        itr = math.log2(M) * 60 / T
    else:
        itr = (math.log2(M) + P * math.log2(P) + (1 - P) * math.log2((1 - P) / (M - 1))) * 60 / T
    return itr

final_itr_lst = []

for i in range(num_fold):
    fold_acc_lst = acc_df[f'Fold{i + 1}'].tolist()
    fold_itr_lst = [cal_itr(M, fold_acc_lst[i] / 100, win_size) for i in range(len(fold_acc_lst))]
    final_itr_lst.append(fold_itr_lst[:-1])

final_itr_lst = np.asarray(final_itr_lst)  # (5, 54)
final_mean_lst = np.mean(final_itr_lst, axis=0)
final_var_lst = np.std(final_itr_lst, ddof=1, axis=0)  # ddof: default——divide N(biased);1——divide N-1(unbiased)
final_mean_lst = np.append(final_mean_lst, np.mean(final_mean_lst, axis=0))
final_var_lst = np.append(final_var_lst, np.std(final_mean_lst, ddof=1, axis=0))

acc_df['ITR(bits/min)'] = [f'{mean:.2f}±{std:.2f}' for mean, std in zip(final_mean_lst, final_var_lst)]

acc_df.to_csv(csv_file, index=False)
print("acc_df:", acc_df)