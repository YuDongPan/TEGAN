# 设计师:Pan YuDong
# 编写者:God's hand
# 时间:2022/11/25 21:49
import os
import shutil

for file in os.listdir('./'):
    if file.split('.')[-1] != 'pth':
        continue

    print("file:", file)
    splits = file.split('_')
    print("splits:", splits)
    folder = splits[-1]
    RX = folder[0] + folder[1]
    FX = folder[2] + folder[3]
    shutil.move(f'./{file}', f'./Target/{RX}/{FX}/{splits[0]}_{splits[1]}_{splits[2]}.pth')
