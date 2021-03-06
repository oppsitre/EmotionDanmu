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
        for ws in doc:
            # w = tuple(ws)
            w = ws
            tmp[w] = 1
        for key in tmp.keys():
            if key in idf.keys():
                idf[key] += 1
            else:
                idf[key] = 1
    for key in idf.keys():
        idf[key] = math.log(float(len(docs)) / (idf[key] + 1))
    return idf

def load_wordmap(file_wordmap, split_symbol):
    word2id = {}
    id2word = {}
    with open(file_wordmap, 'r') as f:
        num_words = int((f.readline()).strip())
        print('Num of words', num_words)
        for line in f:
            ws = line.strip().split(split_symbol)
            ws[1] = int(ws[1])
            word2id[ws[0]] = ws[1]
            id2word[ws[1]] = ws[0]

    return id2word, word2id

def process(file_assign, file_docs):
    fw = open(file_docs, 'w')
    with open(file_assign, 'r') as f:
        docs = []
        for line in f:
            doc = []
            l = line.strip().split(' ')
            if not(len(l) == 1 and len(l[0]) == 0):
                for ws in l:
                    w, t, e = ws.strip().split(':')
                    doc.append(w)
            docs.append(doc)
            fw.write(' '.join(doc) + '\n')
    fw.close()

def docs_read(file_read):
    with open(file_read, 'r') as f:
        docs = []
        for line in f:
            doc = []
            l = line.strip().split(' ')
            if not(len(l) == 1 and len(l[0]) == 0):
                for ws in l:
                    tmp = ws.strip().split(':')
                    if len(tmp) == 3:
                        tup = (tmp[0], tmp[1], tmp[2])
                    elif len(tmp) == 2:
                        tup = (tmp[0], tmp[1])
                    elif len(tmp) == 1:
                        tup = tmp[0]
                    else:
                        print('Doc Read Error')
                    doc.append(tup)
            docs.append(doc)
    return docs

def twe_2_process(docs, file_docs):
    content2id = {}
    id2content = {}
    content_num = 0
    id_docs = []
    with open(file_docs, 'w') as f:
        for doc in docs:
            id_doc = []
            for tup in doc:
                if tup not in content2id.keys():
                    content2id[tup] = content_num
                    id2content[content_num] = tup
                    content_num += 1
                id_doc.append(content2id[tup])
            f.write(' '.join([str(i) for i in id_doc]) + '\n')

def doc_embedding_calc(i):
    global wid2vec
    global idf
    global docs
    doc_embedding = [0.0 for k in range(config.word_vector_size)]

    if len(docs[i]) == 0:
        return doc_embedding
    tf = {}
    for ws in docs[i]:
        # w = tuple(ws)
        # print(ws)
        w = ws
        if w in wid2vec:
            # print('Yes')
            if w in tf.keys():
                tf_val = tf[w]
            else:
                tf_val = 0.0
                for ws_ in docs[i]:
                    w_ = ws_
                    if w == w_:
                        tf_val += 1
            tf_val /= len(docs[i])
            tf[w] = tf_val
            tf_idf = tf[w] * idf[w]
            # print('tf_idf', tf_idf)
            # print(len(wid2vec[w[0]]))
            doc_embedding = [doc_embedding[j] + tf_idf * wid2vec[w][j] for j in range(config.word_vector_size)]
    print('Document', i, len(doc_embedding))
    return doc_embedding

def docs_embedding_calc(doc_embedding_file, vector_file = 'tmp/vector'):
    global wid2vec
    global docs
    wid2vec = gensim.models.Word2Vec.load(vector_file)
    wid2vec = wid2vec.wv
    numDocs = len(docs)
    with open(doc_embedding_file, 'w') as f:
        cpus = 300
        idx = 0
        pool = multiprocessing.Pool(processes=cpus)
        while idx < numDocs - 1:
            results = pool.map(doc_embedding_calc, [(i + idx) for i in range(min(numDocs - idx, cpus))])
            for res in results:
                f.write(' '.join([str(j) for j in res]) + '\n')
            idx += cpus

if __name__ == '__main__':
    global idf
    global docs
    print('Now Process')
    # process()
    docs = docs_read('../dataset/texts/topic_emotion_assign_tw.txt')
    twe_2_process(docs, "tmp/train_tw2.txt")
    sentences = gensim.models.word2vec.LineSentence("tmp/train_tw2.txt")
    w = gensim.models.Word2Vec(sentences, size=config.word_vector_size, workers=100, min_count = 5)
    w.save("tmp/vector")
    id2word, word2id = load_wordmap('../dataset/texts/wordmap.txt', split_symbol = ' ')
    print('Save the word vector')
    # with open('output/word_vector.txt', 'w') as f:
    #     for key in word2id.keys():
    #         key = str(key)
    #         lst = [key]
    #         if key in w.wv:
    #             vec = w.wv[key]
    #             lst.extend(vec.tolist())
    #             lst = [str(j) for j in lst]
    #             f.write(' '.join(lst) + '\n')

    print('Calcuate Doc Embedding')

    docs = []
    with open('tmp/train_tw2.txt', 'r', encoding='utf-8') as f:
        for line in f:
            doc = []
            if len(line.strip()) > 0:
                for ws in line.strip().split(' '):
                    doc.append(ws)
            docs.append(doc)
    idf = idf_count(docs)
    docs_embedding_calc('../dataset/texts/doc_embedding_tw2_py3.txt', 'tmp/vector')
