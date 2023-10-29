import os

import numpy as np

# train_data = open('raw_data_blank.txt', 'r', encoding='utf-8').readlines()
# new_train_data = open('./raw_data.txt','w', encoding='utf-8')
# for line in train_data:
#     tmp = ""
#     words = line.split('\t')[0]
#     label = line.split('\t')[1]
#     for word in words:
#         if word != " ":
#             tmp += word
#     new_train_data.write(tmp + "_!_" + label)


f_train = open('./train.txt', 'w', encoding='utf-8')
f_val = open('./val.txt', 'w', encoding='utf-8')
f_test = open('./test.txt', 'w', encoding='utf-8')
raw_data = open('./output.txt', 'r', encoding='utf-8').readlines()

num_samples  = len(raw_data)
idx = np.random.permutation(num_samples)
num_train = int(num_samples*0.7)
num_val = int(num_samples*0.2)
train_idx, val_idx, test_idx = idx[0:num_train], \
    idx[num_train:num_train + num_val], idx[num_train+num_val:]

# raw_data = np.array(raw_data)
# f_train.write(str(raw_data[train_idx]))
# f_val.write(str(raw_data[val_idx]))
# f_test.write(str(raw_data[test_idx]))
for i in train_idx:
    f_train.write(raw_data[i].strip('\n')+'\n')
f_train.close()

for i in val_idx:
    f_val.write(raw_data[i].strip('\n')+'\n')
f_val.close()

for i in test_idx:
    f_test.write(raw_data[i].strip('\n')+'\n')
f_test.close()