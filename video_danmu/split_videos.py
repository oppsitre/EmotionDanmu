import numpy as np
ratios = 0.2
tmps = []
with open('../dataset/texts/video_class_train.txt', 'r') as f:
    for line in f:
        tmps.append(line.strip())
indexs = np.arange(len(tmps))
np.random.shuffle(indexs)
nlen = int(ratios * len(indexs))
print('index', indexs)
print('nlen', nlen)
indexs = indexs[:nlen]
with open('../dataset/texts/video_class_train_' + str(ratios) + '.txt', 'w') as f:
    for idx in indexs:
        f.write(tmps[idx] + '\n')
