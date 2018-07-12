from __future__ import division
import sys
import h5py
# import config
# append the upper dir into the system path
sys.path.append('../')
import numpy as np
import random
import math
from preprocess import Comment, Danmuku
import tensorflow as tf
import config


class Reader:

    def __init__(self, config=None, mode="debug", add_noise=config.add_noise):
        '''
        self.wid_tid2wtid: (word_id, topic_id) -> id of (word,topic)
        self.wtid2vec: the id of (word,topic) -> vector
        '''
        self.doc_vector_size = config.doc_vector_size
        print('doc_vector_size', self.doc_vector_size)
        self.n_part_danmu = config.n_part_danmu
        self.ratio_train = config.ratio_split[0]
        self.ratio_test = config.ratio_split[1]
        self.batch_size = config.batch_size
        self.modality = config.modality
        self.n_classes = config.n_classes
        self.mode = mode
        self.add_noise = add_noise
        self.videos_origin = []
        self.videos = []
        with open('../dataset/texts/video_class_1.txt', 'r', encoding = 'utf-8') as f:
            for line in f:
                line = line.strip().split('\t')
                self.videos.append([int(line[0]), int(line[-1])])
                self.videos_origin.append([int(line[0]), int(line[-1])])
        random.shuffle(self.videos)
        self.n_data = len(self.videos)
        print('n_data', self.n_data)
        self.n_train = int(self.n_data * self.ratio_train)
        self.n_test = self.n_data - self.n_train
        self.data_train = self.videos[:self.n_train]
        self.data_test = self.videos[self.n_test*(-1):]
        # self.data_train = self.read_data(config.file_data_train)
        # self.n_train = len(self.data_train)
        self.indexs_train = np.arange(len(self.data_train))
        np.random.shuffle(self.indexs_train)

        print('data_train', len(self.data_train))
        # self.data_train = self.data_train[self.indexs_train]
        # self.data_test = self.read_data(config.file_data_test)
        self.videos = self.data_train + self.data_test
        self.n_data = len(self.videos)
        self.n_train = len(self.data_train)
        self.n_test = len(self.data_test)
        self.indexs_train = self.indexs_train[:int(self.n_data * 0.25)]
        print('Train:', self.n_train, 'Test:', self.n_test)
        num_class = [0 for i in range(config.n_classes)]
        for x in self.videos:
            num_class[x[1]] += 1
        print('NUM_CLASS', num_class, flush=True)
        self.x_train = [x[0] for x in self.data_train]
        self.y_train = [x[1] for x in self.data_train]
        self.x_test = [x[0] for x in self.data_test]
        self.y_test = [x[1] for x in self.data_test]
        # self.x_all = [x[0] for x in self.videos_origin]
        # self.y_all = [x[1] for x in self.videos_origin]
        self.x_all = self.x_train + self.x_test
        self.y_all = self.y_train + self.y_test
        self.frame_num = config.frame_num
        self.n_input_frame = config.n_input_frame
        print('Read Doc2ID', flush=True)
        # [vid, type]->[did, clust_start, cluster_end, cluster_center]
        self.doc_id = self.doc_id_read(config.file_doc_id)
        self.frame_per_cluster = config.frame_per_cluster

        if config.train_status == 1 and config.unsupervised_network != 'no':
            print('config.train_status == 1 and config.unsupervised_network != no')
            self.features_video = np.load(
                config.unsupervised_network + '_' + config.file_feature_video)
            print('features_video.shape', self.features_video.shape)
            mean, std = np.mean(self.features_video, 0), np.std(
                self.features_video, 0)
            # print('video mean', mean, 'std', std)
            self.features_video = (self.features_video - mean) / std

            self.features_danmu = np.load(
                config.unsupervised_network + '_' + config.file_feature_danmu)
            print(self.features_danmu.shape)
            print(self.features_danmu)
            mean, std = np.mean(self.features_danmu, 0), np.std(
                self.features_danmu, 0)
            # print('danmu mean', mean, 'std', std)
            self.features_danmu = (self.features_danmu - mean) / std

            self.doc_embedding = self.doc_embedding_read(
                config.file_doc_embedding)  # did->doc_embedding
        else:
            print('config.train_status == 1 and config.unsupervised_network != no else')
            self.doc_embedding = self.doc_embedding_read(
                config.file_doc_embedding)  # did->doc_embedding
            self.frames = self.frames_read("../dataset/vgg_frame_fc7.h5")
            # self.frames_flows = self.frames_read("../dataset/vgg_flows_fc7.h5")
            # (h5py.File(, "r"))["fc_7"] # Read Frames
            print('Frame', self.frames.shape, type(self.frames))
            # print('Read Doc2ID', flush=True)
            # vid -> the index of vid in the h5 file
            self.vid2hid = self.vid2hid_read(
                '../dataset/texts/video_class_1.txt')

    def frames_read(self, filename):
        if config.train_status == 0 or config.unsupervised_network == 'no':
            with h5py.File(filename, 'r') as f:
                data = f['fc_7'][()]
            # data = (h5py.File(, "r"))["fc_7"]
            # shape = data.shape
            print('unsupervised_network is no')

            mean = np.mean(data, axis=(0, 1))
            std = np.std(data, axis=(0, 1))
            data = (data - mean) / std
            # std = np.zeros((shape[0] * shape[1], shape[2]))
            # print(mean.shape, std.shape)
            return data
        else:
            print('unsupervised_network is ', config.unsupervised_network)
            # data = np.load(config.unsupervised_network + '_' + config.file_feature_video)
            # import scipy.io as sio
            # video_train = sio.loadmat('../DCCAE/video_train.mat')
            # video_valid = sio.loadmat('../DCCAE/video_valid.mat')
            #
            # video_train = video_train['X1proj']
            # video_valid = video_valid['XV1proj']

            # data = np.vstack((video_train, video_valid))
            data = np.load(config.unsupervised_network +
                           '_' + config.file_feature_video)
            print('data.shape', data.shape)
            # print(type(video_train), type(train_valid))
            mean, std = np.mean(data, 0), np.std(data, 0)
            # print('video mean', mean, 'std', std)
            data = (data - mean) / std
            return data
        # for d in range(shape[2]):
            # print(data[:, :, d])

    def read_data(self, file_data):
        data = []
        with open(file_data, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip().split('\t')
                # if self.n_classes == 7:
                data.append([line[0], int(line[1])])
                # else:
                #     data.append([int(line[0]), int(line[-1])])
        return data

    def doc_embedding_read(self, file_doc_embedding):
        # if config.train_status == 0 or config.unsupervised_network == 'no':
        docs = {}
        vals = []
        mu, sigma = 0, 0.1
        with open(file_doc_embedding, 'r', encoding='utf-8') as f:
            idx = 0
            for line in f:
                l = line.strip().split(' ')
                docs[idx] = [float(i) for i in l]
                vals.append(docs[idx])
                idx += 1
        vals = np.array(vals)
        mean = np.mean(vals, 0)
        std = np.std(vals, 0)
        for key in docs.keys():
            docs[key] = (docs[key] - mean) / std

        return docs
        #
        # else:
        #     print('Doc Embedding Read Error')
        #     docs = {}
        #     vals = np.load(file_doc_embedding)
        #     mean = np.mean(vals, 0)
        #     std = np.std(vals, 0)
        #     idx = 0
        #     for val in vals:
        #         doc[idx] = (val - mean) / std
        #         idx += 1

    def doc_id_read(self, file_doc_id):
        doc2id = {}
        # if config.train_status == 0 or config.unsupervised_network == 'no':
        with open(file_doc_id, 'r', encoding='utf-8') as f:
            for line in f:
                l = line.strip().split('\t')
                doc2id[(l[0], l[1])] = l[2:]
        # else:
        #     idx = 0
        #     for vid in self.x_all:
        #         for i in range(self.frame_num):
        #             doc2id[(str(vid), str(i+1))] = idx
        #             idx += 1
        return doc2id

    def vid2hid_read(self, success_filename):
        """read the class of the video

        Args:
            success_filename: the path of the file that record the video_id extracted successfully
        Returns:
            vid2hid: vid -> the index of vid in the h5 file
        """
        vid2hid = {}
        if config.train_status == 0 or config.unsupervised_network == 'no':
            with open(success_filename, "r") as f:
                idx = 0
                for line in f:
                    l = line.strip().split('\t')
                    vid2hid[l[0]] = idx
                    idx += 1
            return vid2hid
        else:
            print('VID Read Error')

    def get_frames(self, index, data):
        """prepare the data batch that will be fed into the network

        cause the hdf file can only be taken in a strict ascending order.
        thus we take a continual examples for convenience

        Args:
            index: the set of id of the samples in this batch
            data: all video id in this type of data
        Returns:
            x_batch: a array whose shape should be [batch_size, frame_num, n_input_frame]
        """

        # k = np.random.randint(low=0, high=y.shape[0]-self.batch_size, size=[self.batch_size])
        x_batch = []
        if config.train_status == 0 or config.unsupervised_network == 'no':
            for i in range(len(index)):
                vid = data[index[i]]
                tmp = []
                for j in range(self.frame_num):
                    # by wangjialin 2017.6.6
                    # debug for overfitting
                    if self.mode == "debug":
                        selected_frame_seq = 0
                    else:
                        selected_frame_seq = np.random.randint(
                            self.frame_per_cluster)
                    tmp.append(
                        self.frames[self.vid2hid[vid], j * self.frame_per_cluster + selected_frame_seq, :])
                x_batch.append(tmp)
        else:
            # print('Frame Read Error')
            # else:
            for i in range(len(index)):
                vid = data[index[i]]
                tmp = []
                for j in range(self.frame_num):
                    # by wangjialin 2017.6.6
                    # debug for overfitting
                    # print(type(self.frames), self.frames.shape, self.doc_id[(vid, str(j+1))])
                    tmp.append(
                        self.frames[int(self.doc_id[(vid, str(j + 1))][0]), :])
                x_batch.append(tmp)
        return x_batch

    # def get_comment_embedding(self, index, data):
    #     docs = []
    #     for idx in index:
    #         vid = data[idx]
    #         did = self.doc_id[(str(vid), '0')]
    #         docs.append(np.array(self.doc_embedding[int(did[0])]))
    #     return docs

    def get_danmu_embedding(self, index, data):
        '''
        Args:
            index: the set of id of the samples in this batch
            data: all video id in this type of data
        Return:
            docs: the set of doc embedding, whose shape is [batch_size, n_part_danmu, doc_vector_size]
        '''
        # if config.train_status == 0 or config.unsupervised_network == 'no':
        docs = []
        mu, sigma = 0, 0.1
        for idx in index:
            dans = []
            for i in range(self.n_part_danmu):
                vid = data[idx]
                did = self.doc_id[(str(vid), str(i + 1))]
                tmp = self.doc_embedding[int(did[0])]
                # if self.add_noise is True:
                #     tmp += np.random.normal(mu, sigma, len(tmp))
                dans.append(tmp)
            docs.append(np.array(dans))
        return docs
        # else:
        #     print('Danmu Read Error')

    def next_batch(self):
        """
        @brief return a batch of train and target data
        @return video_batch_data: [batch_size, frame_num, n_input]
        @return danmu_batch_data:  [batch_size, n_part_danmu, n_input]
        @return target_data_batch: [batch_size, 1]
        """
        # print('N_train:', self.n_train, 'Batch_size:', self.batch_size)
        index = np.random.choice(
            np.arange(self.n_train), self.batch_size, replace=False)
        batch = []
        if config.train_status == 0 or config.unsupervised_network == 'no':
            print('train status = 0 or unsupervised_network is no')
            if self.modality[0] == 1:
                video_batch_data = self.get_frames(index, self.x_train)
                batch.append(np.array(video_batch_data))
            if self.modality[1] == 1:
                danmu_batch_data = self.get_danmu_embedding(
                    index, self.x_train)
                batch.append(np.array(danmu_batch_data))
        else:
            if self.modality[0] == 1:
                print('video modality read')
                video_batch_data = self.features_read(
                    self.features_video, self.indexs_train, self.x_train)
                batch.append(np.array(video_batch_data))
            if self.modality[1] == 1:
                print('text modality read')
                danmu_batch_data = self.features_read(
                    self.features_danmu, self.indexs_train, self.x_train)
                batch.append(np.array(danmu_batch_data))

        # y_train = np.zeros((self.batch_size, self.n_classes), dtype=float)
        y_train = np.zeros((len(self.indexs_train), self.n_classes), dtype=float)
        for i, v in enumerate(self.indexs_train):
        # for i, v in enumerate(index):
            y_train[i, self.y_train[v]] = 1.0
        batch.append(y_train)

        return batch

    def features_read(self, features, index, data):
        docs = []
        for idx in index:
            feats = []
            for i in range(self.n_part_danmu):
                vid = data[idx]
                did = self.doc_id[(str(vid), str(i + 1))]
                # print('did', did, 'vid', vid)
                feats.append(features[int(did[0]), :])
            docs.append(feats)

        return docs

    def get_test_data(self):
        index = np.arange(self.n_test)
        batch = []
        # if config.train_status == 0 or config.unsupervised_network == 'no':
        if self.modality[0] == 1:
            video_batch_data = self.get_frames(index, self.x_test)
            batch.append(np.array(video_batch_data))
        if self.modality[1] == 1:
            danmu_batch_data = self.get_danmu_embedding(index, self.x_test)
            batch.append(np.array(danmu_batch_data))
        else:
            if self.modality[0] == 1:
                video_batch_data = self.features_read(self.features_video, index, self.x_test)
                batch.append(np.array(video_batch_data))
            if self.modality[1] == 1:
                danmu_batch_data = self.features_read(self.features_danmu, index, self.x_test)
                batch.append(np.array(danmu_batch_data))

        y_test = np.zeros((self.n_test, self.n_classes), dtype=float)
        for i, v in enumerate(index):
            y_test[i, self.y_test[v]] = 1.0
        batch.append(y_test)
        return batch

    def get_all_data(self):
        batch = []
        x_all = []
        y_all = []
        with open('../dataset/texts/video_class_1.txt', 'r') as f:
            for line in f:
                l = line.strip().split('\t')
                x_all.append(l[0])
                y_all.append(int(l[1]))
        index = np.arange(len(x_all))
        if self.modality[0] == 1:
            video_batch_data = self.get_frames(index, x_all)
            batch.append(np.array(video_batch_data))
        if self.modality[1] == 1:
            danmu_batch_data = self.get_danmu_embedding(index, x_all)
            batch.append(np.array(danmu_batch_data))

        y_batch = np.zeros((self.n_data, self.n_classes), dtype=float)

        for i, v in enumerate(index):
            # print('i', i, 'v', v, flush=True)
            y_batch[i, y_all[v]] = 1.0
        batch.append(y_batch)
        return batch

# if __name__ == '__main__':
#     reader = Reader()
#     data = reader.next_batch()
