"""
Implementation of the collapsed Gibbs sampler for Sentiment-LDA, described in
Sentiment Analysis with Global Topics and Local Dependency (Li, Huang and Zhu)
"""
import sys
# sys.stdout.flush()
import io
# sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import numpy as np
import re
import time
import math
from preprocess import *
import config
import multiprocessing
import copy
# from AC_algo_py3_jieba_server import word_segment
MAX_VOCAB_SIZE = 50000


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

def update(ds, docs, topic, sentiments, priorSentiment, n_dt, n_d, n_dts, n_vts, n_ts, numTopics, numSentiments, alpha, gamma, beta):
    for di, d in enumerate(ds):
        for i, v in enumerate(docs[di]):
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

        return (topic, sentiments, n_dt, n_dts, n_vts, n_ts)

class SentimentLDAGibbsSampler:

    def __init__(self, numTopics, alpha, beta, gamma, numSentiments=21):
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

    # def processSingleReview(self, review, d=None):
    #     """
    #     Convert a raw review to a string of words, including Tokensization and Filtering stopwords
    #     """
    #     expressions = expressions_read()
    #     stops = stopwords_read()
    #     words = sentence_process(expression_segment(review, expressions), stops)
    #     # print(words)
    #     return(words)

    def word_record(self, docs):
        word2id = {}
        id2word = {}
        word2fq = {}
        # word2fq2 = {}
        for doc in docs:
            for w in doc:
                if w not in word2fq.keys():
                    word2fq[w] = 1
                else:
                    word2fq[w] = word2fq[w] + 1
        word2fq = sorted(word2fq.items(), key=lambda d: -d[1])
        word2fq = dict(word2fq[:MAX_VOCAB_SIZE])
        self.vocabSize = len(word2fq)
        i = 0
        for doc in docs:
            for w in doc:
                if w in word2fq.keys():
                    if w not in word2id.keys():
                        word2id[w] = i
                        id2word[i] = w
                        # print(w, word2id[w], word2fq[w], id2word[i], flush=True)
                        i += 1
        # print(word2fq, flush = True)
        print('word2fq', len(word2fq), flush=True)
        return word2id, id2word, word2fq

    def doc2mat(self, docs):
        '''
        change the words of documets into numbers.
        '''
        self.word2id, self.id2word, self.word2fq = self.word_record(docs)
        print('Get WordMap', flush=True)
        self.get_wordmap()
        matrix = []
        for i, doc in enumerate(docs):
            tmp = []
            for w in doc:
                if w in self.word2fq.keys():
                    tmp.append(self.word2id[w])
            matrix.append(tmp)
        return matrix

    def processReviews(self, reviews, saveAs=None, saveOverride=False):
        time_start = time.time()
        # for review in reviews:
        #     if((i + 1) % 5000 == 0):
        #         print("Review %d of %d" % (i + 1, len(reviews)))
        #         time_end = time.time()
        #         print('Time Cost is ', math.ceil((time_end - time_start)))
        #         time_start = time_end
        #     processed_reviews.append(self.processSingleReview(review, i))
        #     i += 1
        #     # if i == 100:
        #     #     break
        self.numDocs = len(reviews)
        wordOccurenceMatrix = self.doc2mat(reviews)
        return wordOccurenceMatrix

    def _initialize_(self, reviews, saveAs=None, saveOverride=False):
        """
        wordOccuranceMatrix: numDocs x vocabSize matrix encoding the
        bag of words representation of each document
        """
        self.wordmatrix = self.processReviews(reviews, saveAs, saveOverride)
        # numDocs, vocabSize = self.wordOccuranceMatrix.shape

        # Pseudocounts
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
        for i in range(self.vocabSize):
            w = self.id2word[i]
            if w in emotion_dict.keys():
                self.priorSentiment[i] = emotion_dict[self.id2word[i]][0]
        for d in range(self.numDocs):
            topicDistribution = sampleFromDirichlet(alphaVec)
            sentimentDistribution = np.zeros(
                (self.numTopics, self.numSentiments))
            for t in range(self.numTopics):
                sentimentDistribution[t, :] = sampleFromDirichlet(gammaVec)
            for i, w in enumerate(self.wordmatrix[d]):
                t = sampleFromCategorical(topicDistribution)
                s = sampleFromCategorical(sentimentDistribution[t, :])
                self.topics[(d, i)] = t
                self.sentiments[(d, i)] = s
                self.n_dt[d, t] += 1
                self.n_dts[d, t, s] += 1
                self.n_d[d] += 1
                self.n_vts[w, t, s] += 1
                self.n_ts[t, s] += 1

    def conditionalDistribution(self, d, v):
        """
        Calculates the (topic, sentiment) probability for word v in document d
        Returns:    a matrix (numTopics x numSentiments) storing the probabilities
        """
        probabilities_ts = np.ones((self.numTopics, self.numSentiments))
        firstFactor = (self.n_dt[d] + self.alpha) / \
            (self.n_d[d] + self.numTopics * self.alpha)
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
        parameter_old = None
        if step > self.save_step:
            parameter_old = np.load('emotion_lda.npy')
        parameter_save = []
        for d in range(self.numDocs):
            for i, v in enumerate(self.wordmatrix[d]):
                t = self.topics[(d, i)]
                s = self.sentiments[(d, i)]
                self.n_dt[d, t] -= 1
                self.n_d[d] -= 1
                self.n_dts[d, t, s] -= 1
                self.n_vts[v, t, s] -= 1
                self.n_ts[t, s] -= 1
                probabilities_ts = self.conditionalDistribution(d, v)
                parameter_save.append(probabilities_ts)
                self.topics[(d, i)] = t
                self.sentiments[(d, i)] = s
                self.n_dt[d, t] += 1
                self.n_d[d] += 1
                self.n_dts[d, t, s] += 1
                self.n_vts[v, t, s] += 1
                self.n_ts[t, s] += 1
        parameter_save = np.array(parameter_save)
        np.save('emotion_lda.npy', np.array(parameter_save))
        if parameter_old != None:
            dif = np.sum(np.absolute(parameter_old - parameter_save))
            if dif < 1:
                print('Have coverage!')
                return 1
            else:
                print('Differnce',  dif)



    def update_parameter(self, dss, paras):
        n_dt_tmp = np.zeros((self.numDocs, self.numTopics))
        n_dts_tmp = np.zeros((self.numDocs, self.numTopics, self.numSentiments))
        # n_d_tmp = np.zeros((self.numDocs))
        n_vts_tmp = np.zeros((self.vocabSize, self.numTopics, self.numSentiments))
        n_ts_tmp = np.zeros((self.numTopics, self.numSentiments))
        for di, ds in enumerate(dss):
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

            for i, d in enumerate(ds):
                for j, v in enumerate(self.wordmatrix[d]):
                    self.topics[(d,j)] = topics[(d,j)]
                    self.sentiments[(d,j)] = sentiments[(d,j)]

        self.n_dt += n_dt_tmp
        self.n_dts += n_dts_tmp
        self.n_vts += n_vts_tmp
        self.n_ts += n_ts_tmp

    def update(self, d):
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

    def run(self, reviews, maxIters=30, saveAs=None, saveOverride=False):
        """
        Runs Gibbs sampler for sentiment-LDA
        """
        self._initialize_(reviews, saveAs, saveOverride)
        # numDocs, vocabSize = self.wordOccuranceMatrix.shape
        time_start = time.time()
        print('Updating...', flush=True)



        # step = 2500
        for iteration in range(maxIters):
            # pool = multiprocessing.Pool(processes=10)
            # print('Start paralle', flush = True)
            results = []
            dss = []
            for d in range(self.numDocs):
                self.update(d)
            #     ds = range(d, min(self.numDocs, d + step), 1)
            #     print(d, len(ds), flush=True)
            #     dss.append(ds)
            #     docs = [self.wordmatrix[dd] for dd in ds]
            #     result = pool.apply_async(update, args=(ds, docs, copy.deepcopy(self.topics), copy.deepcopy(self.sentiments), copy.deepcopy(self.priorSentiment), copy.deepcopy(self.n_dt), copy.deepcopy(self.n_d), copy.deepcopy(self.n_dts), copy.deepcopy(self.n_vts), copy.deepcopy(self.n_ts), copy.copy(self.numTopics), copy.copy(self.numSentiments), copy.deepcopy(self.alpha), copy.deepcopy(self.gamma), copy.deepcopy(self.beta), ))
            #     results.append(result)
            #     # print('Document', d)
            #     # self.update(d)
            # print('allocation finish', flush = True)
            # pool.close()
            # pool.join()
            # res = []
            # for result in results:
            #     res.append(result.get())
            # print('paralle stop', flush = True)
            # self.update_parameter()


            if (iteration + 1) % self.save_step == 0:
                # self.get_wordmap()
                print('Get Assing')
                self.get_tassign()
                # self.document_topic_emotion_distribution()
                # self.word_topic_emotion_distribution()
                # if self.is_coverged(iteration + 1):
                #     print('finish!')
                #     break
                time_end = time.time()
                print("Finshed iteration %d of %d Time Cost is %d" % (iteration + 1, maxIters, math.ceil(time_end - time_start)), flush=True)
                time_start = time_end

    def get_wordmap(self):
        with open('../dataset/texts/wordmap.txt', 'w', encoding = 'utf-8') as f:
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
        with open('document_distribution.txt', 'w', encoding = 'utf-8') as f:
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
                f.write(str(np.sum(probabilities_ts, 0)) + '\n')
                f.write(str(np.sum(probabilities_ts, 1)) + '\n')

    def word_topic_emotion_distribution(self):
        with open('word_distribution.txt', 'w', encoding = 'utf-8') as f:
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
                f.write(str(np.sum(pro[i], 0)) + '\n')
                f.write(str(np.sum(pro[i], 1)) + '\n')

if __name__ == '__main__':
    # word_segment()
    reviews = []
    with open('tokenisation.txt' , 'r', encoding = 'utf-8') as f:
        idx = 0
        for line in f:
            line = line.strip().split('\t')
            reviews.append(line[2:])
            if len(line[2:]) == 0:
                print(idx, line, end='\n', file=sys.stdout, flush=True)
            idx += 1
    print('Initialize...', end='\n', file=sys.stdout, flush=True)
    sampler = SentimentLDAGibbsSampler(config.topic_number, 2.5, 0.1, 0.3)
    print('Emotion-LDA Running',flush=True)
    sampler.run(reviews, 2000, None, True)
    #
    # sampler.getTopKWords(25)
    # sampler.get_wordmap()
    # sampler.get_tassign()
