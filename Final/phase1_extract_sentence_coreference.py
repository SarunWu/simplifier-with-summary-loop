import utils.utils_misc as utils_misc
import os

# check free GPU
"""freer_gpu = str(utils_misc.get_freer_gpu())
os.environ["CUDA_VISIBLE_DEVICES"] = ""+str(freer_gpu)
print("Using GPU "+str(freer_gpu))"""


from sklearn.feature_extraction.text import TfidfVectorizer
from utils.eval_rouge import cal_rouge
from nltk.tokenize import word_tokenize

import numpy as np
import pandas as pd
import nltk
#nltk.download('punkt') # one time execution
import re
from nltk.tokenize import sent_tokenize, RegexpTokenizer
import networkx as nx
from heapq import nlargest

import pickle, time
from tqdm import tqdm
from datetime import datetime, timedelta
ts = time.time()
import sentence_transformers as sbert 

# sbmodel = sbert.SentenceTransformer(model_name_or_path='all-MiniLM-L6-v2', device='cuda')
sbmodel = sbert.SentenceTransformer(model_name_or_path='/home/alexjcortes/.cache/torch/sentence_transformers/sentence-transformers_all-MiniLM-L6-v2/', device='cuda')

ts = time.time()

import spacy
from spacy import displacy
nlp = spacy.load("en_core_web_sm") #Original from Jacky
# nlp = spacy.load("en_core_web_md")

from utils.utils_coref import IntersectionStrategy
coref = IntersectionStrategy()
from utils.stop_word import STOP_WORDS
import wandb

remove_list = [".", ",", "'", "\"", '``', "'", ',', '-', '`', "''"]

# change --> can be sepated into many file for faster process
# e.g. one file may process 20,000 contents, then set the st_idx and end_idx accordingly 
# otherwise
name_to_save = "t0"
name_to_save_loop = name_to_save+".pkl"
name_to_save_final = "final_"+name_to_save+".pkl"

# change --> can be sparated according to above 
st_idx = 0
end_idx = 5

total_news_list = []

# change
# load currently process file (pickle). If none, a blank pkl file should be created first
with open ("/home/jacky/research_boost/final_thesis/"+name_to_save_loop, "rb") as f:
    total_news_list = pickle.load(f)

cur_idx = len(total_news_list)

# change
# load dataset to be extracted (pickle)
with open ("/home/jacky/research_boost/data/train_10news.pkl", "rb") as f:
    news_list = pickle.load(f)

news_list = news_list[st_idx:end_idx+1]

num_test = len(news_list)
#num_test = 15

# some content cannot be process and it makes the program crash. Include that content's idx here to skip it
skip_idx = []

wandb.init(project="extract_sentence")
wandb.run.name = name_to_save
wandb.run.save()


def find_tfidf(content):
    vectorizer = TfidfVectorizer(stop_words=STOP_WORDS, ngram_range=(1, 1))
    x = vectorizer.fit_transform(content)
    vectorizer.get_feature_names()
    df = pd.DataFrame(x.toarray(),  columns=vectorizer.get_feature_names())
    df = df.stack().reset_index()
    df = df.rename(columns={0:'tfidf', 'level_0': 'document', 'level_1': 'term'})
    df2 = df.sort_values(by=['document','tfidf'], ascending=[True,False]).groupby(['document']).head(10)
    tf_term = df2['term'].tolist()
    return tf_term

def find_tfidf_idx(idx):
    content = [news_list[idx]['content']]
    vectorizer = TfidfVectorizer(stop_words=STOP_WORDS, ngram_range=(1, 1))
    x = vectorizer.fit_transform(content)
    vectorizer.get_feature_names()
    df = pd.DataFrame(x.toarray(),  columns=vectorizer.get_feature_names())
    df = df.stack().reset_index()
    df = df.rename(columns={0:'tfidf', 'level_0': 'document', 'level_1': 'term'})
    df2 = df.sort_values(by=['document','tfidf'], ascending=[True,False]).groupby(['document']).head(10)
    tf_term = df2['term'].tolist()
    return tf_term

def filter_and_sort_score(score_dict, num_sent=11):
    res = nlargest(num_sent, score_dict, key = score_dict.get)
    res.sort()
    #print(res)
    # return sorted list
    return res

def make_sentence(sent_list, sorted_score_dict):
    """
    sent_list : list of tokenized sentence of each article
    sorted_score_dict that we want to make sentence
    """
    #len_sent_list = len(sent_list)
    final_sent_list = []
    for i in sorted_score_dict:
        #print(i)
        if i <= len_sent_list-1:
            final_sent_list.append(sent_list[i])
    # join into string
    #result = ' '.join(final_sent_list).replace(',', ' ')
    result = ' '.join(final_sent_list)
    return result


def make_sentence_coref(sent_list, sorted_score_dict):
    """
    sent_list : list of tokenized sentence of each article
    sorted_score_dict that we want to make sentence
    """
    #len_sent_list = len(sent_list)
    #print(len_sent_list)
    final_sent_list = first3_sent
    for i in sorted_score_dict:
        #print(i)
        #if i <= len_sent_list-1:
        final_sent_list.append(sent_list[i])

    # join into string
    #result = ' '.join(final_sent_list).replace(',', ' ')
    result = ' '.join(final_sent_list)
    return result


def lemma_sent(select_sent_coref):
    select_sent_coref_lemma = []
    for i in range(len(select_sent_coref)):
        text = select_sent_coref[i]
        doc = nlp(text)
        token_lemma = [token.lemma_ for token in doc]
        select_sent_coref_lemma.append(token_lemma)
    return select_sent_coref_lemma

def remove_stop_word(sentence):
    for word in STOP_WORDS:
        if word in sentence:
            sentence.remove(word)

    for puct in remove_list:
        if puct in sentence:
            sentence.remove(puct)
    return sentence

def lemma_keyword(kw_list):
    kw_lemma = []
    for word in kw_list:
        doc = nlp(word)
        token_lemma = [token.lemma_ for token in doc]
        for i in token_lemma:
            kw_lemma.append(i)
    return kw_lemma



print("\n\n Start extract sentence")


for idx in range(len(news_list)):

    idx = idx + cur_idx

    print("\n\n")
    print("processing == ", idx)
    print("Writing story {} of {}; {:.2f} percent done. Time spent: {}".format(idx, num_test, float(idx)*100.0/float(num_test), timedelta(seconds=time.time() - ts)))

    print("\n\n")

    if idx == num_test:
        break

    if idx in skip_idx:
        news_dict = dict()
        content = news_list[idx]['content']
        ref_summary = news_list[idx]['summary']

        news_dict["content"] = content
        news_dict["ref_summary"] = ref_summary
        news_dict["tr_content"] = content
        total_news_list.append(news_dict)
        continue

    # save # change
    if (idx !=0) and (idx % 100 == 0):
        print("Saving story {} of {}; {:.2f} percent done. Time spent: {}".format(idx, num_test, float(idx)*100.0/float(num_test), timedelta(seconds=time.time() - ts)))
        # save to file # change
        with open("/home/jacky/research_boost/final_thesis/"+name_to_save_loop, 'wb') as f:
            pickle.dump(total_news_list, f)
        print("saved file : ", name_to_save_loop, " -- idx -- ", idx)

    text = news_list[idx]['content']
    if len(text) == 0:
        continue
    
    # tokenize content into sentences
    sent_list = sent_tokenize(text)

    len_sent_list = len(sent_list)
    first3_sent = sent_list[:3]
    last_sent = sent_list[-1:]

    # the rest of sentences
    select_sent = sent_list[3:len_sent_list-1] 
    # join selected sent into string first
    select_text = ' '.join(select_sent)

    # co-reference selected sentences
    select_text_coref = coref.resolve_coreferences(select_text)
    # tokenized co-ref content into sent
    select_sent_coref = sent_tokenize(select_text_coref)

    select_sent_coref_list = select_sent_coref

    # lower case sentence
    select_sent_coref = [sentence.lower() for sentence in select_sent_coref]

    # lemma
    select_sent_coref_lemma = lemma_sent(select_sent_coref=select_sent_coref)

    # remove stop word in sent after lemma
    select_sent_coref_lemma = [remove_stop_word(sent) for sent in select_sent_coref_lemma]

    # find key word and lemma them
    tf = find_tfidf_idx(idx)
    kw_lemma_list = lemma_keyword(tf)

    # initial matrix for kw co-occurrence score
    len_sent = len(select_sent_coref_lemma)
    sim_mat = np.zeros([len_sent, len_sent])

    # cal score for each pair base on kw co-occurence
    for i in range(len_sent):
        for j in range(len_sent):
            if i != j:
                text1 = select_sent_coref_lemma[i]
                text2 = select_sent_coref_lemma[j]
                for kw in kw_lemma_list:
                    if (kw in text1) and (kw in text2):                
                        sim_mat[i][j] +=1 

    # create graph
    nx_graph = nx.from_numpy_array(sim_mat)

    # pagerank score
    textrank_score = nx.pagerank(nx_graph)

    # sort sentences
    sort_sent = filter_and_sort_score(score_dict=textrank_score, num_sent=round(len_sent/2) - 3)

    # make article with coref
    final_sent_coref = make_sentence_coref(sent_list=select_sent_coref_list,sorted_score_dict=sort_sent)

    # shift sorted list by adding 3 from the beginning (coz we spare 3 sentences to be added already)
    sort_sent = [x+3 for x in sort_sent]
    # add first 3 sent and re-sort
    sort_sent.append(0)
    sort_sent.append(1)
    sort_sent.append(2)
    sort_sent.sort()
    
    # combine sent into final string
    final_sent = make_sentence(sent_list=sent_list, sorted_score_dict=sort_sent)
    
    # combine into dict
    news_dict = dict()
    content = news_list[idx]['content']
    ref_summary = news_list[idx]['summary']

    news_dict["content"] = content
    news_dict["ref_summary"] = ref_summary
    news_dict["tr_content"] = final_sent
    news_dict["tr_coref"] = final_sent_coref

    total_news_list.append(news_dict)


# change
# save to file
# with open("/home/jacky/research_boost/final_thesis/"+name_to_save_final, 'wb') as f: # ORIGINAL JACKY PATH
with open("/Jacky\'s\ code/extracted_dataset/"+name_to_save_final, 'wb') as f: # MY PATH
    pickle.dump(total_news_list, f)
print("saved file : ", name_to_save_final)

print("Total time used")
print(str(timedelta(seconds=time.time() - ts)))

