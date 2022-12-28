STOP_WORDS = set(
    """
a about above across after afterwards again against all almost alone along
already also although always am among amongst amount an and another any anyhow
anyone anything anyway anywhere are around as at
back be became because become becomes becoming been before beforehand behind
being below beside besides between beyond both bottom but by
call can cannot ca could
did do does doing done down due during
each eight either eleven else elsewhere empty enough even ever every
everyone everything everywhere except
few fifteen fifty first five for former formerly forty four from front full
further
get give go
had has have he hence her here hereafter hereby herein hereupon hers herself
him himself his how however hundred
i if in indeed into is it its itself
keep
last latter latterly least less
just
made make many may me meanwhile might mine more moreover most mostly move much
must my myself
name namely neither never nevertheless next nine no nobody none noone nor not
nothing now nowhere
of off often on once one only onto or other others otherwise our ours ourselves
out over own
part per perhaps please put
quite
rather re really regarding
same say see seem seemed seeming seems serious several she should show side
since six sixty so some somehow someone something sometime sometimes somewhere
still such
take ten than that the their them themselves then thence there thereafter
thereby therefore therein thereupon these they third this those though three
through throughout thru thus to together too top toward towards twelve twenty
two
under until up unless upon us used using
various very very via was we well were what whatever when whence whenever where
whereafter whereas whereby wherein whereupon wherever whether which while
whither who whoever whole whom whose why will with within without would
yet you your yours yourself yourselves
cnn ll ve lrb rrb -PRON-
""".split()
)

contractions = ["n't", "'d", "'ll", "'m", "'re", "'s", "'ve"]
STOP_WORDS.update(contractions)

for apostrophe in ["‘", "’"]:
    for stopword in contractions:
        STOP_WORDS.add(stopword.replace("'", apostrophe))

remove_list = [".", ",", "'", "\"", '``', "'", ',', '-', '`', "''"]

import nltk
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('punkt')
from nltk.tokenize import sent_tokenize
import networkx as nx
from heapq import nlargest
import time
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

ts = time.time()

import spacy  # 2.3.8 with numpy==1.21
from spacy import displacy

nlp = spacy.load("en_core_web_sm")

import allennlp

from .intersection_strategy import IntersectionStrategy

coref = IntersectionStrategy()


def find_tfidf(content):
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 1))
    x = vectorizer.fit_transform(content)
    vectorizer.get_feature_names_out()
    df = pd.DataFrame(x.toarray(), columns=vectorizer.get_feature_names_out())
    df = df.stack().reset_index()
    df = df.rename(columns={0: 'tfidf', 'level_0': 'document', 'level_1': 'term'})
    df2 = df.sort_values(by=['document', 'tfidf'], ascending=[True, False]).groupby(['document']).head(10)
    tf_term = df2['term'].tolist()
    return tf_term


def find_tfidf_idx(content):
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 1))
    x = vectorizer.fit_transform(content)
    vectorizer.get_feature_names_out()
    df = pd.DataFrame(x.toarray(), columns=vectorizer.get_feature_names_out())
    df = df.stack().reset_index()
    df = df.rename(columns={0: 'tfidf', 'level_0': 'document', 'level_1': 'term'})
    df2 = df.sort_values(by=['document', 'tfidf'], ascending=[True, False]).groupby(['document']).head(10)
    tf_term = df2['term'].tolist()
    return tf_term


def find_tfidf2term(content):
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(2, 2))
    x = vectorizer.fit_transform(content)
    vectorizer.get_feature_names_out()
    df = pd.DataFrame(x.toarray(), columns=vectorizer.get_feature_names_out())
    df = df.stack().reset_index()
    df = df.rename(columns={0: 'tfidf', 'level_0': 'document', 'level_1': 'term'})
    df2 = df.sort_values(by=['document', 'tfidf'], ascending=[True, False]).groupby(['document']).head(20)
    tf_term = df2['term'].tolist()
    return tf_term


def showtext_ent(summary):
    doc = nlp(summary)
    displacy.serve(doc, style="ent")


def filter_and_sort_score(score_dict, num_sent=11):
    res = nlargest(num_sent, score_dict, key=score_dict.get)
    res.sort()
    # print(res)
    # return sorted list
    return res


def make_sentence(sent_list, sorted_score_dict):
    """
    sent_list : list of tokenized sentence of each article
    sorted_score_dict that we want to make sentence
    """
    # len_sent_list = len(sent_list)
    # print(len_sent_list)
    final_sent_list = []
    for i in sorted_score_dict:
        # print(i)
        if i <= len(sent_list) - 1:
            final_sent_list.append(sent_list[i])

    # join into string
    # result = ' '.join(final_sent_list).replace(',', ' ')
    result = ' '.join(final_sent_list)
    return result


def make_sentence_coref(sent_list, sorted_score_dict, first3_sent):
    """
    sent_list : list of tokenized sentence of each article
    sorted_score_dict that we want to make sentence
    """
    # len_sent_list = len(sent_list)
    # print(len_sent_list)
    final_sent_list = first3_sent
    for i in sorted_score_dict:
        final_sent_list.append(sent_list[i])

    # join into string
    # result = ' '.join(final_sent_list).replace(',', ' ')
    result = ' '.join(final_sent_list)
    return result


# Lemmatize sentence
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


ner_lst = nlp.pipe_labels['ner']


# kw_lemma_list = lemma_keyword(tf)

def lemma_keyword(kw_list):
    kw_lemma = []
    for word in kw_list:
        doc = nlp(word)
        token_lemma = [token.lemma_ for token in doc]
        for i in token_lemma:
            kw_lemma.append(i)
    return kw_lemma


def generate_summary(text):
    # tokenize content into sentences
    sent_list = sent_tokenize(text)
    len_sent_list = len(sent_list)
    first3_sent = sent_list[:3]
    last_sent = sent_list[-1:]

    # the rest of sentences
    select_sent = sent_list[3:len_sent_list - 1]

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
    tf = find_tfidf([text])
    kw_lemma_list = lemma_keyword(tf)

    # initial matrix for kw co-occurrence score
    len_sent = len(select_sent_coref_lemma)
    sim_mat = np.zeros([len_sent, len_sent])

    # cal score for each pair base on kw co-occurrence
    for i in range(len_sent):
        for j in range(len_sent):
            if i != j:
                text1 = select_sent_coref_lemma[i]
                text2 = select_sent_coref_lemma[j]
                for kw in kw_lemma_list:
                    if (kw in text1) and (kw in text2):
                        sim_mat[i][j] += 1
    print("*" * 50)
    print("sim_mat: ", sim_mat)

    # create graph
    nx_graph = nx.from_numpy_array(sim_mat)
    print("*" * 50)
    print("nx_graph: ", nx_graph)

    # pagerank score
    textrank_score = nx.pagerank(nx_graph)
    print("*" * 50)
    print("textrank_score: ", textrank_score)

    # sort sentences
    sort_sent = filter_and_sort_score(score_dict=textrank_score, num_sent=round(len_sent / 2) - 3)

    # make article with coref
    final_sent_coref = make_sentence_coref(sent_list=select_sent_coref_list, sorted_score_dict=sort_sent,
                                           first3_sent=first3_sent)
    print("final_sent_coref: ", final_sent_coref)

    # shift sorted list by adding 3 from the beginning (coz we spare 3 sentences to be added already)
    sort_sent = [i + 3 for i in sort_sent]

    # add first 3 sent and re-sort
    sort_sent.append(0)
    sort_sent.append(1)
    sort_sent.append(2)
    sort_sent.sort()

    # combine sent into final string
    final_sent = make_sentence(sent_list=sent_list, sorted_score_dict=sort_sent)

    return final_sent, final_sent_coref

# tf = find_tfidf(content)

# %%
