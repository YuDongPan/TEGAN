# 设计师:Pan YuDong
# 编写者:God's hand
# 时间:2022/10/13 12:58
import os
paper_list = {}  # paper without number

base_dir = 'D://Postgraduate//Research File//EEG Processing//Geometric Shapes Stimulus'

for paper in os.listdir(base_dir):
    segments = paper.split('.')

    if segments[-1] != 'pdf':
        continue

    if len(segments) > 2:  # if paper already has number, remove it
        paper_list[segments[1] + '.' + segments[2]] = paper
    else:
        paper_list[paper] = paper


print("paper_list:", paper_list)

clear_paper_list = sorted(paper_list.items(), key=lambda x:x[0], reverse=False)

print("clear_paper_list:", clear_paper_list)

for i, (clear_paper, paper) in enumerate(clear_paper_list):
    src_file = base_dir + f'//{paper}'
    tar_file = base_dir
    tar_file += f'//{i}.{clear_paper}'
    os.rename(src_file, tar_file)


print("paper:", paper_list)

