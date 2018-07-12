"""
Implementation of the collapsed Gibbs sampler for Sentiment-LDA, described in
Sentiment Analysis with Global Topics and Local Dependency (Li, Huang and Zhu)
single process : 471s
2core: 1.8 times speedup. 256s
3core:1.4 times speedup. 321s
"""
import sys
import io
import numpy as np
import re
import time
import math
import tqdm
from preprocess import *
import config
import pickle
import multiprocessing
import copy
import gc
global MAX_VOCAB_SIZE
MAX_VOCAB_SIZE = 10000


def sampleFromDirichlet(alpha):
    """
    Sample from a Dirichlet distribution
    alpha: Dirichlet distribution parameter (of length d)
    Returns:
    x: Vector (of length d) sampled from dirichlet distribution

    """
    return np.random.dirichlet(alpha)

def sampleFromCategorical(theta):
    """
    Samples from a categorical/multinoulli distribution
    theta: parameter (of length d)
    Returns:
    x: index ind (0 <= ind < d) based on probabilities in theta
    """
    theta = theta/np.sum(theta)
    return np.random.multinomial(1, theta).argmax()

def word_indices(wordOccuranceVec):
    """
    Turn a document vector of size vocab_size to a sequence
    of word indices. The word indices are between 0 and
    vocab_size-1. The sequence length is equal to the document length.
    """
    for idx in wordOccuranceVec.nonzero()[0]:
        for i in range(int(wordOccuranceVec[idx])):
            yield idx

def conditionalDistribution(d, v, n_d, n_dt, n_dts, n_ts, n_vts, numTopics, numSentiments, alpha, gamma, beta):
    """
    Calculates the (topic, sentiment) probability for word v in document d
    Returns:    a matrix (numTopics x numSentiments) storing the probabilities
    """
    probabilities_ts = np.ones((numTopics, numSentiments))
    firstFactor = (n_dt[d] + alpha) / \
        (n_d[d] + numTopics * alpha)
    secondFactor = (n_dts[d, :, :] + gamma) / \
        (n_dt[d, :] + numSentiments * gamma)[:, np.newaxis]
    thirdFactor = (n_vts[v, :, :] + beta) / \
        (n_ts + n_vts.shape[0] * beta)
    probabilities_ts *= firstFactor[:, np.newaxis]
    probabilities_ts *= secondFactor * thirdFactor
    probabilities_ts /= np.sum(probabilities_ts)
    return probabilities_ts

def update(ds, docs, topics, sentiments, priorSentiment, n_dt, n_d, n_dts, n_vts, n_ts, numTopics, numSentiments, alpha, gamma, beta):
    # print('DS', len(ds), len(docs), flush=True)
    for step in range(10):
        print('Step', step, flush=True)
        for di, d in enumerate(ds):
            # print('DI D', di, d, len(docs[di]), flush=True)
            # print('Docs',  docs[di])
    		# if len(docs[di]) == 0:
    		# 	return (topics, sentiments, n_dt, n_dts, n_vts, n_ts)
            for i, v in enumerate(docs[di]):
                # print('i,v', i, v, flush=True)
                t = topics[(d, i)]
                s = sentiments[(d, i)]
                n_dt[d, t] -= 1
                n_d[d] -= 1
                n_dts[d, t, s] -= 1
                n_vts[v, t, s] -= 1
                n_ts[t, s] -= 1

                probabilities_ts = conditionalDistribution(d, v, n_d, n_dt, n_dts, n_ts, n_vts, numTopics, numSentiments, alpha, gamma, beta)
                if v in priorSentiment:
                    s = priorSentiment[v]
                    t = sampleFromCategorical(probabilities_ts[:, s])
                else:
                    ind = sampleFromCategorical(probabilities_ts.flatten())
                    t, s = np.unravel_index(ind, probabilities_ts.shape)

                topics[(d, i)] = t
                sentiments[(d, i)] = s
                n_dt[d, t] += 1
                n_d[d] += 1
                n_dts[d, t, s] += 1
                n_vts[v, t, s] += 1
                n_ts[t, s] += 1

    print('Return update', flush=True)
    return (topics, sentiments, n_dt, n_dts, n_vts, n_ts)

class SentimentLDAGibbsSampler:

    def __init__(self, numTopics, alpha, beta, gamma, numSentiments=7):
        """
        numTopics: Number of topics in the model
        numSentiments: Number of sentiments (default 2)
        alpha: Hyperparameter for Dirichlet prior on topic distribution
        per document
        beta: Hyperparameter for Dirichlet prior on vocabulary distribution
        per (topic, sentiment) pair
        gamma:Hyperparameter for Dirichlet prior on sentiment distribution
        per (document, topic) pair
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.numTopics = numTopics
        self.numSentiments = numSentiments
        self.save_step = 1
        self.word2id = None
        self.id2word = None
        self.vocabSize = 0
        self.numDocs = 0
        self.word2fq = {}
        self.word2id = {}
        self.id2word = {}
        self.file_para = None
        self.file_wordmap = None

    def word_record(self, docs):
        global MAX_VOCAB_SIZE
        word2id = {}
        id2word = {}
        word2fq = {}
        for doc in docs:
            for w in doc:
                if w not in word2fq.keys():
                    word2fq[w] = 1
                else:
                    word2fq[w] = word2fq[w] + 1
        ll = {}
        emotion_dict = emotion_dict_read()
        for key in emotion_dict.keys():
            if key in word2fq.keys():
                ll[key] =  int(MAX_VOCAB_SIZE)

        print('word2fq size', len(word2fq.keys()))
        word2fq = sorted(word2fq.items(), key=lambda d: -d[1])

        with open('tmp/tmp_word', 'w', encoding='utf-8') as f:
            for it in word2fq:
                f.write(str(it[0]) + '\t' + str(it[1]) + '\n')

        error_num = 0
        with open('tmp/tmp_word', 'r', encoding='utf-8') as f:
            for l in f:
                l_tmp = l.strip().split('\t')
                if len(l_tmp) == 2:
                    ll[l_tmp[0]] =  int(l_tmp[1])
                else:
                    error_num += 1
                if len(ll.keys()) >= MAX_VOCAB_SIZE:
                    break
        print('len of ll', len(ll))
        print('Word Error Num:', error_num)
        word2fq = ll
        self.vocabSize = len(word2fq.keys())
        print('vocabSize', self.vocabSize)
        i = 0
        for w in word2fq.keys():
            word2id[w] = i
            id2word[i] = w
            i += 1

        print('word2fq', len(word2fq), flush=True)
        return word2id, id2word, word2fq

    def doc2mat(self, docs):
        '''
        change the words of documets into numbers.
        '''
        num = 0
        matrix = []
        for i, doc in enumerate(docs):
            tmp = []
            for w in doc:
                if w in self.word2id.keys():
                    tmp.append(self.word2id[w])
                    num += 1
            # if len(tmp) == 0:
            #     print('Length is 0', i, flush=True)
            matrix.append(tmp)
        # print('Word Num', num)
        return matrix

    def read_wordmap(self, file_wordmap):
        word2id = {}
        id2word = {}
        word2fq = {}
        with open(file_wordmap, 'r', encoding='utf-8') as f:
            l = f.readline()
            for l in f:
                # print(l, flush=True)
                word, idx, fq = l.strip().split('\t')
                word2id[word] = int(idx)
                word2fq[word] = int(fq)
                id2word[int(idx)] = word

        print('word2fq size', len(word2fq), flush=True)
        return word2id, id2word, word2fq

    def _initialize_(self, reviews, recover = False, file_assign = '../dataset/texts/topic_emotion_assign.txt', file_wordmap = '../dataset/texts/wordmap.txt'):
        """
        wordOccuranceMatrix: numDocs x vocabSize matrix encoding the
        bag of words representation of each document
        """
        self.numDocs = len(reviews)
        self.file_assign = file_assign
        self.file_wordmap = file_wordmap

        print('Get WordMap', flush=True)
        if recover is False:
            print('Recover is False', flush=True)
            self.word2id, self.id2word, self.word2fq = self.word_record(reviews)
            self.write_wordmap(file_wordmap)

        self.word2id, self.id2word, self.word2fq = self.read_wordmap(file_wordmap)
        self.wordmatrix = self.doc2mat(reviews)
        self.vocabSize = len(self.word2id.keys())
        print('VTS', self.vocabSize, self.numTopics, self.numSentiments)
        self.n_dt = np.zeros((self.numDocs, self.numTopics))
        self.n_dts = np.zeros((self.numDocs, self.numTopics, self.numSentiments))
        self.n_d = np.zeros((self.numDocs))
        self.n_vts = np.zeros((self.vocabSize, self.numTopics, self.numSentiments))
        self.n_ts = np.zeros((self.numTopics, self.numSentiments))
        self.topics = {}
        self.sentiments = {}
        self.priorSentiment = {}
        alphaVec = self.alpha * np.ones(self.numTopics)
        gammaVec = self.gamma * np.ones(self.numSentiments)

        emotion_dict = emotion_dict_read()
        # print(self.id2word[0], flush=True)
        for i in range(self.vocabSize):
            # print('i', i, flush=True)
            w = self.id2word[i]
            if w in emotion_dict.keys():
                self.priorSentiment[i] = emotion_dict[self.id2word[i]][0]

        if recover is False:
            d = 0
            for d in range(self.numDocs):
                topicDistribution = sampleFromDirichlet(alphaVec)
                sentimentDistribution = np.zeros(
                    (self.numTopics, self.numSentiments))
                for t in range(self.numTopics):
                    sentimentDistribution[t, :] = sampleFromDirichlet(gammaVec)
                for i, w in enumerate(self.wordmatrix[d]):
                    t = sampleFromCategorical(topicDistribution)
                    s = sampleFromCategorical(sentimentDistribution[t, :])
                    # print(d, i, flush=True)
                    # print(self.n_dt.shape, flush=True)
                    self.topics[(d, i)] = t
                    self.sentiments[(d, i)] = s
                    self.n_dt[d, t] += 1
                    self.n_dts[d, t, s] += 1
                    self.n_d[d] += 1
                    self.n_vts[w, t, s] += 1
                    self.n_ts[t, s] += 1
        else:
            print('Loading parameters....', flush=True)
            with open(file_assign, 'r', encoding='utf-8') as f:
                d = 0
                for l in f:
                    l = l.strip().split(' ')
                    if len(l) == 1 and len(l[0]) == 0:
                        d += 1
                        continue
                    # if d == 9528:
                    #     print('L', l, flush=True)
                    for i, wte in enumerate(l):
                        w, t, s = wte.strip().split(':')
                        t = int(t)
                        s = int(s)
                        w = int(w)

                        self.topics[(d, i)] = t
                        self.sentiments[(d, i)] = s
                        self.n_dt[d, t] += 1
                        # print('N_DTS', self.n_dts.shape, flush=True)
                        self.n_dts[d, t, s] += 1
                        self.n_d[d] += 1
                        self.n_vts[w, t, s] += 1
                        self.n_ts[t, s] += 1
                    d += 1
            print('Loaded parameters....', flush=True)

    def conditionalDistribution(self, d, v):
        """
        Calculates the (topic, sentiment) probability for word v in document d
        Returns:    a matrix (numTopics x numSentiments) storing the probabilities
        """
        probabilities_ts = np.ones((self.numTopics, self.numSentiments))
        firstFactor = (self.n_dt[d] + self.alpha) / \
            (self.n_d[d] + self.numTopics * self.alpha)
        firstFactor = (self.n_dt[d] + self.alpha)
        secondFactor = (self.n_dts[d, :, :] + self.gamma) / \
            (self.n_dt[d, :] + self.numSentiments * self.gamma)[:, np.newaxis]
        thirdFactor = (self.n_vts[v, :, :] + self.beta) / \
            (self.n_ts + self.n_vts.shape[0] * self.beta)
        probabilities_ts *= firstFactor[:, np.newaxis]
        probabilities_ts *= secondFactor * thirdFactor
        probabilities_ts /= np.sum(probabilities_ts)
        return probabilities_ts

    def getTopKWordsByLikelihood(self, K):
        """
        Returns top K discriminative words for topic t and sentiment s
        ie words v for which p(t, s | v) is maximum
        """
        pseudocounts = copy.deepcopy(self.n_vts)
        normalizer = np.sum(pseudocounts, (1, 2))
        pseudocounts /= normalizer[:, np.newaxis, np.newaxis]
        for t in range(self.numTopics):
            for s in range(self.numSentiments):
                topWordIndices = pseudocounts[:, t, s].argsort()[-1:-(K + 1):-1]
                vocab = self.vectorizer.get_feature_names()
                # print(t, s, [vocab[i] for i in topWordIndices])

    def getTopKWords(self, K):
        """
        Returns top K discriminative words for topic t and sentiment s
        ie words v for which p(v | t, s) is maximum
        """
        pseudocounts = copy.deepcopy(self.n_vts)
        normalizer = np.sum(pseudocounts, (0))
        pseudocounts /= normalizer[np.newaxis, :, :]
        for t in range(self.numTopics):
            for s in range(self.numSentiments):
                topWordIndices = pseudocounts[:, t, s].argsort()[-1:-(K + 1):-1]
                # vocab = self.vectorizer.get_feature_names()
                # print(t, s, [self.id2word[i] for i in topWordIndices])

    def is_coverged(self, step):
        # self.get_tassign()
        # print('Saving parameter...', flush=True)
        # # self.save_para(self.file_para)
        print('Parameter have been saved...', flush=True)
        parameter_old = None
        if step > self.save_step:
            parameter_old = np.load('emotion_lda.npy')
        parameter_save = []
        for d in range(self.numDocs):
            parameter_tmp = np.zeros((self.numTopics, self.numSentiments))
            for i, v in enumerate(self.wordmatrix[d]):
                t = self.topics[(d, i)]
                s = self.sentiments[(d, i)]
                self.n_dt[d, t] -= 1
                self.n_d[d] -= 1
                self.n_dts[d, t, s] -= 1
                self.n_vts[v, t, s] -= 1
                self.n_ts[t, s] -= 1
                parameter_tmp += self.conditionalDistribution(d, v)
                self.topics[(d, i)] = t
                self.sentiments[(d, i)] = s
                self.n_dt[d, t] += 1
                self.n_d[d] += 1
                self.n_dts[d, t, s] += 1
                self.n_vts[v, t, s] += 1
                self.n_ts[t, s] += 1
        parameter_tmp /= np.sum(parameter_tmp)
        parameter_save.append(parameter_tmp)
        parameter_save = np.array(parameter_save)
        print('Saving emotion_lda.npy', flush=True)
        np.save('emotion_lda.npy', np.array(parameter_save))
        if not(parameter_old is None):
            dif = np.sum(np.absolute(parameter_old - parameter_save))
            if dif < 0.001:
                print('Have coverage!')
                return 1
            else:
                print('Differnce',  dif)

    # def save_para(self, file_para):
    #     with open(file_para, 'w', encoding='utf-8') as f:
    #         para = {}
    #         para['n_dt'] = self.n_dt
    #         para['n_dts'] = self.n_dts
    #         para['n_d'] = self.n_d
    #         para['n_vts'] = self.n_vts
    #         para['n_ts'] = self.n_ts
    #         para['topics'] = self.topics
    #         para['sentiments'] = self.sentiments
    #         para['priorSentiment'] = self.priorSentiment
    #         json.dump(para, f)

    def single_update(self, d):
        for i, v in enumerate(self.wordmatrix[d]):
            t = self.topics[(d, i)]
            s = self.sentiments[(d, i)]
            self.n_dt[d, t] -= 1
            self.n_d[d] -= 1
            self.n_dts[d, t, s] -= 1
            self.n_vts[v, t, s] -= 1
            self.n_ts[t, s] -= 1

            probabilities_ts = self.conditionalDistribution(d, v)
            if v in self.priorSentiment:
                s = self.priorSentiment[v]
                t = sampleFromCategorical(probabilities_ts[:, s])
            else:
                ind = sampleFromCategorical(probabilities_ts.flatten())
                t, s = np.unravel_index(ind, probabilities_ts.shape)

            self.topics[(d, i)] = t
            self.sentiments[(d, i)] = s
            self.n_dt[d, t] += 1
            self.n_d[d] += 1
            self.n_dts[d, t, s] += 1
            self.n_vts[v, t, s] += 1
            self.n_ts[t, s] += 1

    def update_parameter(self, dss, paras):
        n_dt_tmp = np.zeros((self.numDocs, self.numTopics))
        n_dts_tmp = np.zeros((self.numDocs, self.numTopics, self.numSentiments))
        # n_d_tmp = np.zeros((self.numDocs))
        n_vts_tmp = np.zeros((self.vocabSize, self.numTopics, self.numSentiments))
        n_ts_tmp = np.zeros((self.numTopics, self.numSentiments))

        for di, ds in enumerate(dss):
            # print('di, ds', di, len(ds), flush=True)
            # print('len of para', len(paras), flush=True)
            topics = paras[di][0]
            sentiments = paras[di][1]

            n_dt = paras[di][2]
            n_dt_tmp += n_dt - self.n_dt

            n_dts = paras[di][3]
            n_dts_tmp += n_dts - self.n_dts

            n_vts = paras[di][4]
            n_vts_tmp += n_vts - self.n_vts

            n_ts = paras[di][5]
            n_ts_tmp += n_ts - self.n_ts

            # for key in self.topics.keys():
            #     if key[0] == 9528:
            #         print('self.topics', key, flush=True)
            #
            # for key in topics.keys():
            #     if key[0] == 9528:
            #         print('topics', key, flush=True)

            # print('Doc', self.wordmatrix[9528], flush=True)
            for i, d in enumerate(ds):
                # print(self.wordmatrix[d])
				# if len()
                for j, v in enumerate(self.wordmatrix[d]):
                    self.topics[(d,j)] = topics[(d,j)]
                    self.sentiments[(d,j)] = sentiments[(d,j)]

        self.n_dt += n_dt_tmp
        self.n_dts += n_dts_tmp
        self.n_vts += n_vts_tmp
        self.n_ts += n_ts_tmp
        del paras
        del n_dt_tmp
        del n_vts_tmp
        del n_ts_tmp
        gc.collect()

    def collect_results(self, result):
        print('result', len(result))
        self.results.append(result)

    def run(self, reviews, maxIters=30, recover = False):
        """
        Runs Gibbs sampler for sentiment-LDA, deepcopy is a very slow method
        """
        self._initialize_(reviews, recover)
        # numDocs, vocabSize = self.wordOccuranceMatrix.shape
        time_start = time.time()
        print('Updating...', flush=True)
        for iteration in tqdm.tqdm(range(maxIters), mininterval=6):
            # for d in range(self.numDocs):
            #     self.single_update(d)
            n_procs = 5
            pool = multiprocessing.Pool(processes=n_procs)
            n_per_core = self.numDocs//n_procs
            print('n_per_core', n_per_core, flush=True)
            #print('Start paralle', flush = True)
            self.results = []
            dss = []
            #print("There are %d docs to process" % self.numDocs)
            start_t = time.time()

            results = []
            for d in range(n_procs):
                if d  == n_procs-1:
                    ds =  [kk for kk in range(d*n_per_core, self.numDocs)]
                else:
                    ds = [kk for kk in range(d*n_per_core, (d+1)*n_per_core)]
                dss.append(ds)
                docs = [self.wordmatrix[dd] for dd in ds]

                pp = pool.apply_async(update, args=(ds, docs, copy.deepcopy(self.topics), copy.deepcopy(self.sentiments), \
                                 copy.deepcopy(self.priorSentiment), copy.deepcopy(self.n_dt), copy.deepcopy(self.n_d), \
                                 copy.deepcopy(self.n_dts), copy.deepcopy(self.n_vts), copy.deepcopy(self.n_ts), \
                                 copy.copy(self.numTopics), copy.copy(self.numSentiments), copy.deepcopy(self.alpha),\
                                 copy.deepcopy(self.gamma), copy.deepcopy(self.beta)))
                results.append(pp)
            pool.close()
            pool.join()
            self.results = []
            for res in results:
                self.results.append(res.get())
            self.update_parameter(dss, self.results)
            # end_t = time.time()
            # t1 = end_t-start_t
            # print("One multiprocessing iteration consume %f seconds" % t1)
            # print('paralle stop', flush = True)

            #
            # print("Single Process Start")
            # start_t = time.time()
            # for i in range(self.numDocs):
            #     self.single_update(i)
            # end_t = time.time()
            # t2 = end_t - start_t
            # print("Single Process Consume %f seconds" % t2)
            # print("use %d cores, speed %f times" %(n_procs, t2/t1))

            if (iteration + 1) % self.save_step == 0:
                # self.get_wordmap()
                print('Get Assign')
                self.get_tassign()
                self.document_topic_emotion_distribution()
                self.word_topic_emotion_distribution()
                if self.is_coverged(iteration + 1):
                    print('finish!')
                    break
                time_end = time.time()
                print("Finshed iteration %d of %d Time Cost is %d" % (iteration + 1, maxIters, math.ceil(time_end - time_start)), flush=True)
                time_start = time_end

    def write_wordmap(self, file_wordmap):
        with open(file_wordmap, 'w', encoding = 'utf-8') as f:
            f.write(str(self.vocabSize) + '\n')
            for w in self.word2id.keys():
                f.write(w + '\t' + str(self.word2id[w]) + '\t' + str(self.word2fq[w]) + '\n')

    def get_tassign(self):
        with open('../dataset/texts/topic_emotion_assign.txt', 'w', encoding = 'utf-8') as f:
            for d in range(self.numDocs):
                doc_str = ''
                for i, v in enumerate(self.wordmatrix[d]):
                    t = self.topics[(d, i)]
                    s = self.sentiments[(d, i)]
                    doc_str += str(v) + ':' + str(t) + ':' + str(s) + ' '
                f.write(doc_str[:-1] + '\n')

    def document_topic_emotion_distribution(self):
        '''
        Calculates the topic distribution and emotion distribution of each domcument respectively.
        '''
        print('Get document_distribution')
        with open('../dataset/texts/document_distribution.txt', 'w', encoding = 'utf-8') as f:
            for d in range(self.numDocs):
                f.write(str(d) + '\n')
                probabilities_ts = np.zeros((self.numTopics, self.numSentiments))
                for i, v in enumerate(self.wordmatrix[d]):
                    t = self.topics[(d, i)]
                    s = self.sentiments[(d, i)]
                    self.n_dt[d, t] -= 1
                    self.n_d[d] -= 1
                    self.n_dts[d, t, s] -= 1
                    self.n_vts[v, t, s] -= 1
                    self.n_ts[t, s] -= 1
                    probabilities_ts += self.conditionalDistribution(d, v)
                    self.topics[(d, i)] = t
                    self.sentiments[(d, i)] = s
                    self.n_dt[d, t] += 1
                    self.n_d[d] += 1
                    self.n_dts[d, t, s] += 1
                    self.n_vts[v, t, s] += 1
                    self.n_ts[t, s] += 1
                if np.sum(probabilities_ts) != 0:
                    probabilities_ts /= np.sum(probabilities_ts)

                # print('np.sum(probabilities_ts, 0).shape', np.sum(probabilities_ts, 0).shape)
                # print('np.sum(probabilities_ts, 1).shape', np.sum(probabilities_ts, 1).shape)
                ss = ''
                for it in np.sum(probabilities_ts, 0):
                    ss = ss + str(it) + ' '
                f.write(ss + '\n')
                ss = ''
                for it in np.sum(probabilities_ts, 1):
                    ss = ss + str(it) + ' '
                # f.write(ss + '\n')
                f.write(ss + '\n')

    def word_topic_emotion_distribution(self):
        print('Get word_distribution')
        with open('../dataset/texts/word_distribution.txt', 'w', encoding = 'utf-8') as f:
            pro = {}
            for i in range(self.vocabSize):
                pro[i] = np.zeros((self.numTopics, self.numSentiments))
            for d in range(self.numDocs):
                for i, v in enumerate(self.wordmatrix[d]):
                    t = self.topics[(d, i)]
                    s = self.sentiments[(d, i)]
                    self.n_dt[d, t] -= 1
                    self.n_d[d] -= 1
                    self.n_dts[d, t, s] -= 1
                    self.n_vts[v, t, s] -= 1
                    self.n_ts[t, s] -= 1
                    pro[v] += self.conditionalDistribution(d, v)
                    self.topics[(d, i)] = t
                    self.sentiments[(d, i)] = s
                    self.n_dt[d, t] += 1
                    self.n_d[d] += 1
                    self.n_dts[d, t, s] += 1
                    self.n_vts[v, t, s] += 1
                    self.n_ts[t, s] += 1
            for i in range(self.vocabSize):
                f.write(str(i) + '\t' + str(self.id2word[i]) + '\n')
                if np.sum(pro[i]) > 0:
                    pro[i] = pro[i] / np.sum(pro[i])
                # f.write(str(np.sum(pro[i], 0)) + '\n')
                # f.write(str(np.sum(pro[i], 1)) + '\n')
                ss = ''
                for it in np.sum(pro[i], 0):
                    ss = ss + str(it) + ' '
                # ss = ss[:-1]
                f.write(ss + '\n')
                ss = ''
                for it in np.sum(pro[i], 1):
                    ss = ss + str(it) + ' '
                # ss = ss[:-1]
                # f.write(ss + '\n')
                f.write(ss + '\n')

if __name__ == '__main__':
    # word_segment()
    reviews = []
    with open('../dataset/texts/tokenisation.txt' , 'r', encoding = 'utf-8') as f:
        idx = 0
        for line in f:
            line = line.strip().split('\t')
            reviews.append(line[2:])
            # if len(line[2:]) == 0:
                # print(idx, line, end='\n', file=sys.stdout, flush=True)
            idx += 1
    print('Initialize...', end='\n', file=sys.stdout, flush=True)
    sampler = SentimentLDAGibbsSampler(config.topic_number, config.lda_alpha, config.lda_beta, config.lda_gamma)
    print('Emotion-LDA Running',flush=True)
    sampler.run(reviews, 1000, recover = False)
    #
    # sampler.getTopKWords(25)
    # sampler.get_wordmap()
    # sampler.get_tassign()
