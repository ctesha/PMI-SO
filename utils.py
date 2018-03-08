# coding: utf-8
'''
Author: Leyi Wang (leyiwang.cn@gmail.com)
Date: Last updated on 2016-04-08 by Leyi Wang
'''

import os, re, sys, random, math, subprocess
from nltk.stem import WordNetLemmatizer

is_win32 = (sys.platform == 'win32')
########### Global Parameters ###########
if is_win32:
    TOOL_PATH = 'F:\\NJUST\\Toolkits'
    LIBLINEAR_LEARN_EXE = TOOL_PATH + '\\liblinear-2.1\\windows\\train.exe'
    LIBLINEAR_CLASSIFY_EXE = TOOL_PATH + '\\liblinear-2.1\\windows\\predict.exe'

else:
    TOOL_PATH = '/home/lywang/Toolkits'
    LIBLINEAR_LEARN_EXE = TOOL_PATH + '/liblinear-2.1/train'
    LIBLINEAR_CLASSIFY_EXE = TOOL_PATH + '/liblinear-2.1/predict'

########## PosTag Reviews ##########
def read_text(fname_list, samp_tag):
    '''text format 2: one class one file, docs are sperated by samp_tag
    '''
    doc_class_list = []
    doc_str_list = []
    for fname in fname_list: # for fname in sorted(fname_list):
        # print 'Reading', fname
        doc_str = open(fname, 'r').read().decode('utf8', 'ignore').encode('utf8','ignore')
        patn = '<' + samp_tag + '>(.*?)</' + samp_tag + '>'
        str_list_one_class = re.findall(patn, doc_str, re.S)
        class_label = os.path.basename(fname)
        doc_str_list.extend(str_list_one_class)
        doc_class_list.extend([class_label] * len(str_list_one_class))
    doc_str_list = [x.strip() for x in doc_str_list]
    return doc_str_list, doc_class_list

########## Feature Extraction Fuctions ##########
def get_doc_terms_list(doc_str_list):
    res = []
    import collections
    term_freq_dict = dict(collections.Counter([word for doc in doc_str_list for word in doc.split()]).most_common())
    res = [[w for w in doc.split() if term_freq_dict[w] > 5] for doc in doc_str_list]
    return res

def get_class_set(doc_class_list):
    class_set = sorted(list(set(doc_class_list)))
    return class_set

def get_term_set(doc_terms_list):
    term_set = set()
    for doc_terms in doc_terms_list:
        term_set.update(doc_terms)
    return sorted(list(term_set))

def stat_df_term(term_set, doc_terms_list):
    '''
    df_term is a dict
    '''
    df_term = {}.fromkeys(term_set, 0)
    for doc_terms in doc_terms_list:
        for term in set(doc_terms):
            if df_term.has_key(term):
                df_term[term] += 1
    return df_term

def stat_df_class(class_set, doc_class_list):
    '''
    df_class is a list
    '''
    df_class = [doc_class_list.count(x) for x in class_set]
    return df_class

def stat_df_term_class(term_set, class_set, doc_terms_list, doc_class_list):
    '''
    df_term_class is a dict-list

    '''
    class_id_dict = dict(zip(class_set, range(len(class_set))))
    df_term_class = {}
    for term in term_set:
        df_term_class[term] = [0]*len(class_set)
    for k in range(len(doc_class_list)):
        class_label = doc_class_list[k]
        class_id = class_id_dict[class_label]
        doc_terms = doc_terms_list[k]
        for term in set(doc_terms):
            if df_term_class.has_key(term):
                df_term_class[term][class_id] += 1
    return df_term_class

def feature_selection_mi(df_class, df_term_class):
    term_set = df_term_class.keys()
    term_score_dict = {}.fromkeys(term_set)
    for term in term_set:
        df_list = df_term_class[term]
        class_set_size = len(df_list)
        cap_n = sum(df_class)
        score_list = []
        for class_id in range(class_set_size):
            cap_a = df_list[class_id]
            cap_b = sum(df_list) - cap_a
            cap_c = df_class[class_id] - cap_a
            p_c_t = (cap_a + 1.0) / (cap_a + cap_b + class_set_size)
            p_c = float(cap_a+cap_c) / cap_n
            score = math.log(p_c_t / p_c)
            score_list.append(score)
        term_score_dict[term] = score_list[1] - score_list[0]
    term_score_list = term_score_dict.items()
    term_score_list.sort(key=lambda x: -x[1])
    term_set_fs = [x[0] for x in term_score_list]
    return term_set_fs, term_score_dict
