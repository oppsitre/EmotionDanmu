# import preprocess
import numpy as np
file_doc_id = '../dataset/texts/doc2id.txt'
topic_emotion_file = '../../JST_py/data/documents.txt'
file_emotion_dict = '../../JST_py/data/emotion_dict.txt'
# wordmap = '../dataset/texts/wordmap.txt'
def read_doc():
    docs = []
    with open(topic_emotion_file, 'r') as f:
        for line in f:
            doc = []
            l = line.strip().split(' ')
            if not (len(l) == 1 and len(l[0]) == 0):
                for ws in l:
                    # w, t, e = ws.strip().split(':')
                    doc.append(ws)
            docs.append(doc)
    return docs

def doc_id_read(file_doc_id = file_doc_id):
    doc2id = {}
    id2doc = {}
    with open(file_doc_id, 'r', encoding = 'utf-8') as f:
        for line in f:
            l = line.strip().split('\t')
            doc2id[(l[0],l[1])] = l[2:]
    return doc2id

def doc_aggregate(docs, doc2id, did):
    doc = []
    for i in range(10):
        # l = doc2id[(str(did), str(i+1))]
        doc.extend(docs[int(doc2id[(str(did), str(i+1))][0])])
    return doc

def load_wordmap(wordmap):
    id2word = {}
    word2id = {}
    idx = 0
    with open(wordmap, 'r') as f:
        f.readline()
        for l in f:
            try:
                word, idx, fq = l.strip().split('\t')
            except:
                print('Error', l)
            idx = int(idx)
            id2word[idx] = word
            word2id[word] = idx
    return id2word, word2id

global num


def predict(doc, emotion_dict):
    global num
    ep = np.array([0 for i in range(7)])
    for w in doc:
        if w in emotion_dict.keys():
            ep += emotion_dict[w]

    if sum(ep) != 0:
        ep = ep / np.sum(ep)
        # ep = [i / np.sum(ep) for i in ep]
    return np.argmax(ep), [str(i) for i in ep]
    
def emotion_dict_read():
    emotion_dict = {}
    with open(file_emotion_dict, 'r') as f:
        for line in f:
            l = line.strip().split('\t')
            emotion_dict[l[0]] = [int(i) for i in l[1:]]

    return emotion_dict

if __name__ == '__main__':
    global num
    num = 0
    emotion_dict = emotion_dict_read()
    print(len(emotion_dict.keys()))
    docs = read_doc()
    doc2id = doc_id_read()
    # id2word, word2id = load_wordmap(wordmap)
    fw = open('../lexicon_method/PredictTrain.csv', 'w')
    fw.write('Label,Prediction,Probabilities\n')
    with open('../dataset/texts/video_class_1.txt', 'r') as f:
        idx = 0
        for line in f:
            l = line.strip().split('\t')
            doc = doc_aggregate(docs, doc2id, l[0])
            y, pro = predict(doc, emotion_dict)
            idx += 1
            s = [l[1], str(y)]
            s.extend(pro)
            # print(','.join(s))
            fw.write(','.join(s) + '\n')
