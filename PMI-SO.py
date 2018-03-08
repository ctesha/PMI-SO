#coding=utf8
'''
  Title: PMI-SO
  Author: Leyi Wang
  Date: Last update 2018-03-08
  Email: leyiwang.cn@gmail.com
'''
import os
import utils

FNAME_LIST = ['negative', 'positive']
SAMP_TAG = 'review_text'

def save_score_list( term_score_dict, term_set_fs, fname_score):
    score_str = '\n'.join( (str(term) + '\t' + str(term_score_dict[term])) for term in term_set_fs)
    open(fname_score,'w').write(score_str)

def build_fs_dict(token_date_dir, result_dir, fs_percent=1):
    PMI_SO = result_dir + os.sep + 'PMI-SO'
    print 'Reading text...'
    doc_str_list_token, doc_class_list_token = utils.read_text([token_date_dir + os.sep + x for x in FNAME_LIST], SAMP_TAG)
    print 'End Reading'
    doc_terms_list_train = utils.get_doc_terms_list(doc_str_list_token)
    class_set = utils.get_class_set(doc_class_list_token)
    term_set = utils.get_term_set(doc_terms_list_train)
    
    print 'PMI-SO Sentiment Lexicon Construction...'
    df_term = utils.stat_df_term(term_set, doc_terms_list_train)
    df_class = utils.stat_df_class(class_set, doc_class_list_token)
    df_term_class = utils.stat_df_term_class(term_set, class_set, doc_terms_list_train, doc_class_list_token)
    term_set_fs, term_score_list = utils.feature_selection_mi(df_class, df_term_class)
    save_score_list(term_score_list, term_set_fs, PMI_SO)

if __name__=='__main__':
    token_dir, result_dir = 'data' + os.sep + 'train', 'result'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    build_fs_dict(token_dir, result_dir)
    print 'Done!'
