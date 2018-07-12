import multiprocessing as mp
import time
import pickle
import numpy as np
import copy
import os

results = []


def job(x):
    x = x + 1
    return x


def collect_results(result):
    results.append(result)


def multicore():
    d = np.random.random([80000, 3000])
    f = np.random.random([80000, 3000])
    d[0, 1] = 1

    x = [d, f]
    start_t = time.time()
    pool = mp.Pool()
    for i in range(len(x)):
        pool.apply_async(job, args=(copy.deepcopy(x[i]),))
        #os.system("taskset -p -c %d %d" % (i%os.cpu_count(), os.getpid()))
    pool.close()
    pool.join()
    end_t = time.time()
    print("BETA %f seconds" % (end_t - start_t))
    #
    result_new = []
    start_t = time.time()
    pool = mp.Pool()
    for i in range(len(x)):
        pool.apply_async(job, args=(copy.deepcopy(x[i]),))
        # result_new.append(res.get())
    pool.close()
    pool.join()
    end_t = time.time()
    print("BETA %f seconds" % (end_t - start_t))
    # pool = mp.Pool()
    # start_t = time.time()
    # for i in x:
    #     pool.apply_async(job, args=(pickle.loads(pickle.dumps(d,-1)), ), callback=collect_results)
    # pool.close()
    # pool.join()
    # end_t = time.time()
    # print("BETA2 %f seconds" %(end_t-start_t))

    # start_t = time.time()
    # rr = job(d)
    # end_t = time.time()
    # print("SINGLE %f seconds" %(end_t-start_t))
    # print(results)
    # print(result_new)
    #print([res.get() for res in multi_res])


if __name__ == '__main__':
    # multicore()
    import scipy.io as sio
    video_train = sio.loadmat('../DCCAE/video_train.mat')
    video_valid = sio.loadmat('../DCCAE/video_valid.mat')

    video_train = video_train['X1proj']
    video_valid = video_valid['XV1proj']
    data = np.vstack((video_train, video_valid))
    print(data.shape)
