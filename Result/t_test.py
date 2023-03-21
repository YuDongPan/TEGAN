# 设计师:Pan YuDong
# 编写者:God's hand
# 时间:2022/12/10 17:42
import numpy as np
import pandas as pd
from scipy import stats

win_size = 1.0
dataset = 'Direction'
method = 'TRCA'
control_group = f'{dataset}/{method}'
variable_group = f'{dataset}_AUG/{method}'
ratio = 3
UD = 0
metric = 0


if metric == 0:
    ttest_column = 'Mean±Std'
else:
    ttest_column = 'ITR(bits/min)'

if ratio == -1:
    proportion = 'Training-Free'
elif ratio == 1:
    proportion = '8vs2'
elif ratio == 2:
    proportion = '5vs5'
elif ratio == 3:
    proportion = '2vs8'
else:
    proportion = 'N-1vs1'

if UD == -1:
    val_way = 'Unsupervised'
elif UD == 0:
    val_way = 'PerSubject'
else:
    val_way = 'CrossSubject'


org_csv_file = f'./{control_group}/{proportion}_{val_way}_Classification_Result({str(win_size)}S).csv'
aug_csv_file = f'./{variable_group}/{proportion}_{val_way}_Classification_Result({str(win_size)}S).csv'

org_df = pd.read_csv(org_csv_file)
aug_df = pd.read_csv(aug_csv_file)

result_org = org_df[ttest_column].tolist()
result_aug = aug_df[ttest_column].tolist()

result_org = [float(result_org[i].split('±')[0]) for i in range(len(result_org) - 1)]
result_aug = [float(result_aug[i].split('±')[0]) for i in range(len(result_aug) - 1)]

result_org = np.asarray(result_org)
result_aug = np.asarray(result_aug)

print("result_org:", result_org)
print("result_aug:", result_aug)

t1, pval1 = stats.ttest_rel(result_org, result_aug)
t2, pval2 = stats.ttest_1samp(result_aug - result_org, 0)

print(f"t1:{t1}, pval1:{pval1}")
print(f"t2:{t2}, pval2:{pval2}")


