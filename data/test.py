from random import shuffle
with open('all.txt', 'r') as f:
    datas = f.readlines()
all = []
for i in range(len(datas)):
    data = datas[i].strip()
    if data != '':
        all.append(data)
    shuffle(all)

ratio = 0.8
with open('train.txt', 'w') as f:
    for i in range(len(all)):
        if i <= ratio * len(all):
            f.write(all[i] + '\n')

with open('test.txt', 'w') as f:
    for i in range(len(all)):
        if i > ratio * len(all):
            f.write(all[i] + '\n')





