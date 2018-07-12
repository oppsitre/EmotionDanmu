import numpy as np


with open('../result_dir/temporal_flows/PredictTest_0.8.csv', 'r') as f:
    res_temp = []
    for line in f:
        # print(line.strip().split(','))
        res_temp.append(line.strip().split(','))

with open('../result_dir/VGG/PredictTest_0.8.csv', 'r') as f:
    res_spat = []
    for line in f:
        # print(line.strip().split(',')s)
        res_spat.append(line.strip().split(','))


def average(a1, a2):
    a1 = np.array([float(x) for x in a1])
    a2 = np.array([float(x) for x in a2])
    a3 = a1 + a2
    return a3


with open('../result_dir/VGG_flows_average_fusion/PredictTest.csv', 'w') as f:
    print(','.join(res_temp[0]), file=f)
    for i in range(len(res_temp) - 1):
        label = res_temp[i+1][0]
        predt = average(res_temp[i + 1][2:], res_spat[i + 1][2:])
        res = [label, np.argmax(predt)]
        res.extend(predt)
        res = ','.join([str(x) for x in res])
        print(res, file=f)
