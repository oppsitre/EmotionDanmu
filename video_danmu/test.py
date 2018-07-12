import json
import config
import numpy as np
from preprocess import Danmuku
def change():
    docs = []
    with open('../dataset/texts/topic_emotion_assign.txt', 'r') as f:
        for line in f:
            doc = []
            # print(len(line), line, flush=True)
            l = line.strip().split(' ')
            if not(len(l) == 1 and len(l[0]) == 0):
                for ws in l:
                    # print(ws)
                    w, e, t = ws.strip().split(':')
                    # print(w, e, t, flush=True)
                    doc.append((w,e,t))
            docs.append(doc)

    print('Len', len(docs))

    with open('../dataset/texts/words.txt', 'w') as f:
        for doc in docs:
            ws = []
            for tup in doc:
                ws.append(tup[0])
            f.write(' '.join(ws) + '\n')

def load_wordmap(file_wordmap = '../dataset/texts/wordmap_word.txt', split_symbol = '\t'):
    word2id = {}
    id2word = {}
    with open(file_wordmap, 'r') as f:
        num_words = int((f.readline()).strip())
        for line in f:
            ws = line.strip().split(split_symbol)
            # ws[1] = int(ws[1])
            word2id[ws[0]] = ws[1]
            id2word[ws[1]] = ws[0]

    return word2id, id2word

emtion2id = {'PA': [0, 0], 'PE': [0, 1], 'PD': [1, 2], 'PH': [1, 3], 'PG': [1, 4], 'PB': [1, 5], 'PK': [1, 6],
 			 'NA': [2, 7], 'NB': [3, 8], 'NJ': [3, 9], 'NH': [3, 10], 'PF':[3, 11], 'NI':[4, 12], 'NC':[4, 13],
			 'NG': [4, 14], 'NE':[5, 15], 'ND': [5, 16], 'NN': [5, 17], 'NK': [5, 18], 'NL': [5, 19], 'PC': [6, 20]
}

emotion_name = [('快乐', 'PA'), ('安心', 'PE'), ('尊敬', 'PD'),
('赞扬', 'PH'), ('相信', 'PG'), ('喜爱', 'PB'), ('祝愿', 'PK'),
('愤怒', 'NA'), ('悲伤', 'NB'), ('失望','NJ'), ('疚','NH'), ('思','PF'), ('慌', 'NI'), ('恐惧', 'NC'), ('羞','NG'), ('烦闷','NE'), \
('憎恶','ND'), ('贬责','NN'), ('妒忌','NK'), ('怀疑', 'NL'), ('惊奇', 'PC')]
def emotion_dict_read(files = ['../dictionary/word/emotion_dictionary/word_emotion_dict.csv', '../dictionary/word/emotion_dictionary/expression_all.txt']):
    emotion_dict = {}
    with open(files[0], 'r', encoding='utf-8') as f:
        lid = 1
        for l in f:
            l = l.strip().split('\t')
            if len(l) != 4:
                print(lid, l)
            lid += 1
            for key in emtion2id.keys():
                if l[3].find(key) != -1:
                    emotion_dict[l[0]] = emtion2id[key]
                    break

    print('Len 1:', len(emotion_dict.keys()))
    with open(files[1], 'r', encoding='utf-8') as f:
        for l in f:
            l = l.strip().split('\t')
            for key in emtion2id.keys():
                if l[4].find(key) != -1:
                    emotion_dict[l[0]] = emtion2id[key]
                    break

    print('Len 2:', len(emotion_dict.keys()))
    return emotion_dict

def add_topic_emotion():
    slda_word2id, slda_id2word = load_wordmap()
    lda_word2id, lda_id2word = load_wordmap('../dataset/texts/wordmap.txt', split_symbol = ' ')
    emotion_dict = emotion_dict_read()
    docs = []
    with open('../dataset/texts/model-18600.tassign', 'r') as f:
        for line in f:
            doc = []
            l = line.strip().split(' ')
            for tup in l:
                w, t = tup.split(':')
                if lda_id2word[w] == '###':
                    break
                w_raw = slda_id2word[lda_id2word[w]]
                if w_raw in emotion_dict.keys():
                    e = emotion_dict[w_raw][0]
                else:
                    e = 7
                doc.append(':'.join([w, t]))
            docs.append(doc)


    with open('../dataset/texts/topic_emotion_assign_tw.txt', 'w') as f:
        for doc in docs:
            f.write(' '.join(doc) + '\n')

def del_empty():
    with open('../dataset/texts/words.txt', 'r') as f:
        docs = []
        for line in f:
            doc = []
            l = line.strip().split(' ')
            # print(line.strip(), '###', l)
            if len(l) == 1 and len(l[0]) == 0:
                doc = '###'
            else:
                doc = line.strip()
            docs.append(doc)

    with open('../dataset/texts/words_###.txt', 'w') as f:
        for doc in docs:
            f.write(doc + '\n')

# def add_empty_symbol():

def calc_mean_std():
    docs = {}
    vals = []
    file_doc_embedding = config.file_doc_embedding
    with open(file_doc_embedding, 'r', encoding = 'utf-8') as f:
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
        tmp = docs[key]
        print('Before', tmp, flush=True)
        tmp = (tmp - mean) / std
        print('After', tmp, flush=True)
        docs[key] = tmp
    print(mean)
    print(std)

def word_record(docs):
    MAX_VOCAB_SIZE = 10000
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
    vocabSize = len(word2fq.keys())
    print('vocabSize', vocabSize)
    i = 0
    for w in word2fq.keys():
        word2id[w] = i
        id2word[i] = w
        i += 1

    print('word2fq', len(word2fq), flush=True)
    return word2id, id2word, word2fq

if __name__ == '__main__':
    # import random
    # videos = []
    # with open('../dataset/texts/video_class_1.txt', 'r') as f:
    #     for line in f:
    #         l = line.strip().split('\t')
    #         videos.append(l)
    # print(len(videos))
    # random.shuffle(videos)
    # videos_train = videos[:3244]
    # videos_test = videos[3244:]
    # print(len(videos_train), len(videos_test))
    #
    # with open('../dataset/texts/video_class_train.txt', 'w') as f:
    #     for video in videos_train:
    #         f.write('\t'.join(video) + '\n')
    #
    # with open('../dataset/texts/video_class_test.txt', 'w') as f:
    #     for video in videos_test:
    #         f.write('\t'.join(video) + '\n')
    # word2id, id2word = load_wordmap('../dataset/texts/wordmap_word.txt')
    # emotion_dict = emotion_dict_read()
    # new_dict = {}
    # print(len(emotion_dict.keys()))
    # with open('../dataset/texts/tokenisation.txt', 'r') as f:
    #     for line in f:
    #         l = line.strip().split()
    #         for w in l:
    #             # print(w)
    #             if w in emotion_dict.keys():
    #                 new_dict[w] = 1
    #     print(len(new_dict.keys()))
    # with open('../dataset/texts/tokenisation.txt', 'r') as f:
    #     docs = []
    #     for line in f:
    #         docs.append(line.strip().split('\t'))
    #
    # word2id, id2word, word2fq = word_record(docs)
    # with open('../dataset/texts/wordmap_word.txt', 'w') as f:
    #     f.write(str(len(word2id.keys())) + '\n')
    #     for key in word2id.keys():
    #         f.write(key + '\t' + str(word2id[key]) + '\t' + str(word2fq[key]) + '\n')
    #
    # with open('../dataset/texts/documents.txt', 'w') as f:
    #     did = 0
    #     for doc in docs:
    #         tmp_doc = []
    #         for w in doc:
    #             if w in word2id.keys():
    #                 tmp_doc.append(word2id[w])
    #         if len(tmp_doc) == 0:
    #             tmp_doc.append(len(word2id.keys()))
    #         tmp_doc = [did] + tmp_doc
    #         tmp_doc = [str(i) for i in tmp_doc]
    #         f.write(' '.join(tmp_doc) + '\n\n')
    #         did += 1
    # print(len(emotion_dict.keys()))
    # ll = []
    # with open('../dictionary/word/emotion_dictionary/word_emotion_dict.csv', 'r') as f:
    #     for line in f:
    #         l = line.strip().split('\t')
    #         ll.append(l)
    #
    # with open('../dictionary/word/emotion_dictionary/tmp.csv', 'w') as f:
    # for l in ll:
    #     if len(l) == 5:
    #         new_l = [l[0], l[1], l[2], l[4]]
    #         # f.write('\t'.join(new_l) + '\n')
    #     else:
    #         new_l = l
    #         # f.write('\t'.join(l) + '\n')
    #     finded = False
    #     for tup in emotion_name:
    #         if new_l[3].find(tup[0]) != -1 and new_l[3].find(tup[1]) != -1:
    #             finded = True
    #             break
    #     if finded == False:
    #         print(finded, new_l)
    #
    #
    # with open('../dataset/texts/emotion_dict.txt', 'w') as f:
    #     for key in word2id.keys():
    #         if key in emotion_dict.keys():
    #             e = ['0'] * 7
    #             e[emotion_dict[key][0]] = '1'
    #             s = [word2id[key]] + e
    #             f.write('\t'.join(s) + '\n')
    #
    #
    # with open('../dataset/texts/tokenisation.txt', 'r') as f:
    #     docs = []
    #     lid = 0
    #     for line in f:
    #         doc = []
    #         l = line.strip().split('\t')
    #         for w in l:
    #             if w in word2id.keys():
    #                 doc.append(word2id[w])
    #         if len(doc) == 0:
    #             doc.append(str(len(word2id.keys())))
    #         # print(doc)
    #         docs.append(doc)
    # failed_video = []
    # with open('../Video_Extract/failed_video_id.txt', 'r') as f:
    #     for line in f:
    #         l = line.strip()
    #         failed_video.append(l)
    #
    # video_train = []
    # with open('../dataset/texts/video_class_train.txt', 'r') as f:
    #     for line in f:
    #         l = line.strip().split('\t')
    #         if l[0] not in failed_video:
    #             video_train.append(l)
    # print('train', len(video_train))
    # video_test = []
    # with open('../dataset/texts/video_class_test.txt', 'r') as f:
    #     for line in f:
    #         l = line.strip().split('\t')
    #         if l[0] not in failed_video:
    #             video_test.append(l)
    # print('test', len(video_test))
    #
    # with open('../dataset/texts/video_class_train_tmp.txt', 'w') as f:
    #     for v in video_train:
    #         f.write('\t'.join(v) + '\n')
    # with open('../dataset/texts/video_class_test_tmp.txt', 'w') as f:
    #     for v in video_test:
    #         f.write('\t'.join(v) + '\n')
    # for key in emotion_disct.keys():
    #     print(key, emotion_dict[key])
    # add_topic_emotion()
    # del_empty()
    # with open('../dataset/texts/doc_embedding_wts.txt') as f:
    #     for line in f:
    #         l = line.strip().split(' ')
    #         print(len(l))
    # calc_mean_std()
    # ll = []
    # num = 0
    # with open('../dataset/texts/doc2id.txt', 'r') as f:
    #     for line in f:
    #         l = line.strip().split('\t')
    #         if l[0] not in ll:
    #             num += 1
    #             ll.append(l[0])
    # print(num)
    # num = 0
    # videos = []
    # with open('../dataset/texts/video_class_1.txt', 'r') as f:
    #     for line in f:
    #         l = line.strip().split('\t')
    #         if l[0] in videos:
    #             num += 1
    #         videos.append(l[0])
    # print(num)
    # with open(config.file_doc_embedding, 'r') as f:
    #     docs = []
    #     for line in f:
    #         l = line.strip().split(' ')
    #         docs.append(l)
    #     print(len(docs), len(docs[0]))
    videos = []
    with open('../dataset/texts/video_class_train.txt', 'r') as f:
        for line in f:
            l = line.strip().split('\t')
            videos.append(l[0])
    with open('../dataset/texts/video_class_test.txt', 'r') as f:
        for line in f:
            l = line.strip().split('\t')
            videos.append(l[0])
    print(len(videos))
    lens = []
    for aid in videos:
        dan = Danmuku(aid = aid)
        lens.append(dan.video_length)
        print(aid, lens[-1], flush=True)
    print(min(lens), max(lens), sum(lens) / len(lens))
