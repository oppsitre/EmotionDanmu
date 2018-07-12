import numpy as np
import preprocess
file_doc_id = '../dataset/texts/doc2id.txt'

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
        tmp = docs[int(doc2id[(str(did), str(i+1))][0])]
        # print()
        doc += tmp
    return doc

def predict(doc, idx):
    global num
    ep = np.zeros((7,))
    for w in doc:
        ep[int(w[1])] += 1
    ep = ep / np.sum(ep)
    return np.argmax(ep), [str(i) for i in ep]

if __name__ == '__main__':
    # emtion2id = {'PA': [0, 0], 'PE': [0, 1], 'PD': [1, 2], 'PH': [1, 3], 'PG': [1, 4], 'PB': [1, 5], 'PK': [1, 6],
    #  			 'NA': [2, 7], 'NB': [3, 8], 'NJ': [3, 9], 'NH': [3, 10], 'PF':[3, 11], 'NI':[4, 12], 'NC':[4, 13],
    # 			 'NG': [4, 14], 'NE':[5, 15], 'ND': [5, 16], 'NN': [5, 17], 'NK': [5, 18], 'NL': [5, 19], 'PC': [6, 20]
    # }
    # ff = []
    docs = []
    with open('../../JST_py/data/99900.tassign', 'r') as f:
        lid = -1
        for line in f:
            lid += 1
            if lid % 2 == 0:
                continue
            l = line.strip().split()
            doc = []
            for ws in l:
                # print(ws)
                w, e, t = ws.strip().split(':')
                doc.append((w, e))
            docs.append(doc)

    # docs = {}
    # i = 0
    # while i+2 < len(ff):
    #     # print(ff[i+2].split(' '))
    #     docs[ff[i]] = np.array([float(j) for j in ff[i+1].split(' ')])
    #     i += 3

    doc2id = doc_id_read()
    fw = open('../emotionlda_predict/PredictTrain.csv', 'w')
    fw.write('Label,Prediction,Probabilities\n')
    with open('../dataset/texts/video_class_1.txt', 'r') as f:
        idx = 0
        for line in f:
            l = line.strip().split('\t')
            doc = doc_aggregate(docs, doc2id, l[0])
            y, pro = predict(doc, idx)
            idx += 1
            s = [l[1], str(y)]
            s.extend(pro)
            # print(','.join(s))
            fw.write(','.join(s) + '\n')
