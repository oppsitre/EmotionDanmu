from Aho_Corasick import *
from hanziconv import HanziConv
import jieba
import codecs
import os
import multiprocessing as mp
import math

# paths
dict_path = "../dictionary"
data_path = "."
#########################################################################

def load_expressions(txt_path):
	#load all expressions in "expressions_all.txt"
	with codecs.open(txt_path+'/expressions_all.txt', encoding='utf-8') as f:
	    expressions_all = f.read().split('\n')
	return [k for k in list(set(expressions_all)) if k!=u""]

def load_stopwords(txt_path):
	#load stopwords in "stopwords_new.txt"
	with codecs.open(txt_path+'/stopwords_new.txt', encoding='utf-8') as f:
	    stopwords_new = f.read().split('\n')
	return [k for k in list(set(stopwords_new)) if k!=u""] + [' ', '\t'] #space and tab were added to stopword list

def load_dicts_jieba(txts_path):
	# This function will load jieba_dict_lower.txt
	# and also ALL txt files under "extra dict" and "hanlp"
	# the folder structure as follow
	'''
	txts_path
		├──jieba_dict_lower.txt
		└──extra dict
			└──hanlp
	'''
	# load all dictionaries under folder 'extra dicts/hanlp' into jieba
	hanlp_dicts = [i for i in os.listdir(txts_path+'/extra dict/hanlp') if i.split('.')[-1]=='txt']
	for i in hanlp_dicts:
	    jieba.load_userdict(txts_path+'/extra dict/hanlp/'+i)

	# load all dictionaries under folder 'extra dicts' into jieba
	extra_dicts = [i for i in os.listdir(txts_path+'/extra dict') if i.split('.')[-1]=='txt']
	for i in extra_dicts:
	    jieba.load_userdict(txts_path+'/extra dict/'+i)

    # reloaded the jieba dictionary with lowercase
	jieba.load_userdict(txts_path+'/jieba_dict_lower.txt')

	return hanlp_dicts, extra_dicts

def load_data(txt_path):
    #load all texts in "texts.txt"
    with codecs.open('../dataset/texts/texts.txt', encoding='utf-8') as f:
        data = f.read().split('\n')

    # get texts from file 'texts.txt' directly
    vid = [data[i].split('\t')[0] for i in range(0,len(data),2) if data[i]!='']
    cluster = [data[i].split('\t')[1] for i in range(0,len(data),2) if data[i]!='']
    textss = [data[i] for i in range(1,len(data),2)]
    print('VID length:', len(vid), 'Cluster length:', len(cluster), 'Texts Length', len(textss))
    # for t in textss:
    #     if type(t) is None:
    #         print(type(t))
    return vid, cluster, textss

def clean_stopwords(token_list, stopwords):
    return [i for i in token_list if i not in stopwords]

def expression_process(text):
    # print(text)
    strings = acp.expression_extract(text)
    res=[]
    for i in strings:
        if i[1] is 'str':
            string = HanziConv.toSimplified(i[0]).lower()
            res += clean_stopwords(list(jieba.cut(string)), stopwords_new)
        else:
            res += [i[0]]
    return res


def get_all_process(texts):
    # print(texts)
    # print(len(texts))
    return [expression_process(i) for i in texts]


#text = '(°∀°)ﾉ°∀°)ﾉ震惊了……UP主让我忘记原版怎么唱了……(╯°口°)╯(ﾟДﾟ≡ﾟдﾟ)!?'
#f = expression_process(text)
#g = get_all_process(texts[:1])
expressions_all = load_expressions(dict_path)
stopwords_new = load_stopwords(dict_path)
hanlp_dicts, extra_dicts = load_dicts_jieba(dict_path)
vidID, clusterID, texts = load_data(data_path)
########################################################################

# build the acp
acp = acmation()
for i in expressions_all:
    acp.insert(i)
acp.ac_automation()


def word_segment():


    from time import time
    start = time()

    n_instance = 13
    batch_size = math.ceil(len(texts)/n_instance)
    texts_start = list(range(0, len(texts), batch_size)) + [len(texts)]
    para_arg = [texts[i:j] for i,j in list(zip(texts_start[:-1], texts_start[1:]))]

    with mp.Pool(processes=n_instance) as pool:
        results = pool.map(get_all_process, para_arg)

    end=time()
    print(end-start)

    print('len of instance:', len(results))

    results = [item for sublist in results for item in sublist]
    print('len of res:', len(results))

    data = list(zip(vidID, clusterID, results))

    with codecs.open('../dataset/texts/tokenisation.txt', 'w', encoding='utf-8') as fw:
        for i,j,k in list(zip(vidID, clusterID, results)):
            fw.writelines(i + '\t' + j + '\t')
            fw.writelines('\t'.join(k) +' \n')

if __name__ == '__main__':
    word_segment()
