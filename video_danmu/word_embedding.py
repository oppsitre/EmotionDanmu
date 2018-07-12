import gensim
import config
import numpy as np
import multiprocessing
import math
import json

global wid2vec
global idf
global docs


def idf_count(docs):
    idf = {}
    for doc in docs:
        tmp = {}
        for w in doc:
            tmp[w] = 1
        for key in tmp.keys():
            if key in idf.keys():
                idf[key] += 1
            else:
                idf[key] = 1
    for key in idf.keys():
        idf[key] = math.log(float(len(docs)) / idf[key])
    return idf


def load_wordmap(file_wordmap='../dataset/texts/wordmap_word.txt', split_symbol='\t'):
    word2id = {}
    id2word = {}
    with open(file_wordmap, 'r') as f:
        num_words = int((f.readline()).strip(split_symbol))
        print('Num of words', num_words)
        for line in f:
            ws = line.strip().split(split_symbol)
            # print(ws)
            ws[1] = int(ws[1])
            word2id[ws[0]] = ws[1]
            id2word[ws[1]] = ws[0]

    return id2word, word2id


def process(file_assign='../dataset/texts/topic_emotion_assign_lda.txt', file_docs='tmp/train.txt'):
    fw = open(file_docs, 'w')
    with open(file_assign, 'r') as f:
        lid = -1
        docs = []
        for line in f:
            lid += 1
            if lid % 2 == 0:
                continue
            doc = []
            l = line.strip().split(' ')
            if not(len(l) == 1 and len(l[0]) == 0):
                for ws in l:
                    # print(ws)
                    w, t, e = ws.strip().split(':')
                    doc.append(w)
            docs.append(doc)
            fw.write(' '.join(doc) + '\n')
    fw.close()


def vector_read(vector_file):
    wid2vec = {}
    with open(vector_file, 'r') as f:
        idx = 0
        for line in f:
            l = line.strip().split(' ')
            wid2vec[l[0]] = [float(i) for i in l[1:]]
            idx += 1
    return wid2vec


def doc_embedding_calc(i):
    global wid2vec
    global idf
    global docs
    print('Document', i)
    doc_embedding = [0.0 for k in range(config.word_vector_size)]
    if len(docs[i]) == 0:
        return doc_embedding
    tf = {}
    for w in docs[i]:
        if w in wid2vec:
            if w in tf.keys():
                tf_val = tf[w]
            else:
                tf_val = 0.0
                for w_ in docs[i]:
                    if w == w_:
                        tf_val += 1
            tf_val /= len(docs[i])
            tf[w] = tf_val
            tf_idf = tf[w] * idf[w]
            doc_embedding = [doc_embedding[j] + tf_idf * wid2vec[w][j]
                             for j in range(config.word_vector_size)]
    return doc_embedding


def docs_embedding_calc(doc_embedding_file, docs_file, vector_file='tmp/vector'):
    global wid2vec
    # wid2vec = gensim.models.Word2Vec.load(vector_file)
    # wid2vec = wid2vec.wv
    wid2vec = vector_read(vector_file)
    numDocs = len(docs)
    with open(doc_embedding_file, 'w') as f:
        cpus = 100
        idx = 0
        pool = multiprocessing.Pool(processes=cpus)
        while idx < numDocs - 1:
            results = pool.map(doc_embedding_calc, [
                               (i + idx) for i in range(min(numDocs - idx, cpus))])
            for res in results:
                f.write(' '.join([str(j) for j in res]) + '\n')
            idx += cpus


if __name__ == '__main__':
    # wordmapfile = '../../JST_py/emotion/wordmap.txt'
    # tassignfile = '../../JST_py/emotion/99900.tassign'
    # wordmapfile = '../emotion-detection-from-text/output/wordmap.tsv'
    tassignfile = '../emotion-detection-from-text/lda_output/100000.tassign'
    # id2word, word2id = load_wordmap(wordmapfile, split_symbol='\t')
    print('Process')
    process(file_assign=tassignfile)
    sentences = gensim.models.word2vec.LineSentence("tmp/train.txt")
    w = gensim.models.Word2Vec(
        sentences, size=config.word_vector_size, workers=100)
    w.save("tmp/vector")
    print('Save the word vector')
    w.save_word_vectors(
        "../emotion-detection-from-text/output/word_vector.txt")

    print('Calcuate Doc Embedding')
    global idf
    global docs
    docs_file = 'tmp/train.txt'
    docs = []
    with open(docs_file, 'r') as f:
        for line in f:
            doc = []
            if len(line) > 0:
                for w in line.strip().split(' '):
                    doc.append(w)
            docs.append(doc)
    idf = idf_count(docs)
    docs_embedding_calc('../emotion-detection-from-text/output/doc_embedding_we.txt',
                        'tmp/train.txt', "../emotion-detection-from-text/output/word_vector.txt")
    # vec = np.load('tmp/vectors.syn0.npy')
