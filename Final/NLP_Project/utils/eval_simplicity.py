import textstat
from wordfreq import zipf_frequency
import time
from datetime import datetime, timedelta
from tqdm import tqdm
import numpy as np
import nltk
from varname import nameof, argname
from bert_score import BERTScorer
from nltk.tokenize import sent_tokenize, RegexpTokenizer


#S-BERT
from sentence_transformers import SentenceTransformer, util
sbertmodel = SentenceTransformer('all-MiniLM-L6-v2')

# SBERT Cosine Similarity
def cal_SBERT_cosine_score(ref_text, pred_text):
    ts = time.time()

    total_score = dict()
    total_score['sbert'] = []
    pred_text_emb = []
    ref_text_emb = []

    # hyps = sampled, refs = referenced
    for hyps, refs in tqdm(zip(pred_text, ref_text)):
        emb1 = sbertmodel.encode(hyps)
        emb2 = sbertmodel.encode(refs)
        pred_text_emb.append(emb1)
        ref_text_emb.append(emb2)

    for emb1, emb2 in tqdm(zip(pred_text_emb, ref_text_emb)):
        cos_sim_score = util.cos_sim(emb1, emb2)
        score = cos_sim_score.item()
        total_score['sbert'].append(score)

    total_score['sbert'] = np.array(total_score['sbert'])

    score = total_score['sbert'].mean()

    print("S-BERT cosine similarity Score")
    print("the higher, the better \n")
    print("S-BERT cosine similarity Score between text = {:.3f}".format(score))
    print()
    print("time spent in calculation:{}".format(timedelta(seconds=time.time() - ts)))

    return total_score

# Word Frequency, the higher, the easier (score between 0-8)
class cal_wordfreq:
    def __init__(self):
        self.stopws = set(nltk.corpus.stopwords.words("english") + ["might", "would", "``", "-", "--"])

    def word_score_func(self, w):
        return zipf_frequency(w, 'en', wordlist="large")

    def is_good_word(self, w):
        if "'" in w:
            return False
        if len(w) > 30 or len(w) == 1:
            return False
        if w.lower() in self.stopws:
            return False
        if all(c.isdigit() for c in w):
            return False
        return True

    def cal_score(self, text):
        words1 = nltk.tokenize.word_tokenize(text)
        words1 = set([w.lower() for w in words1 if self.is_good_word(w)])
        words1_zipfs = [{"w": w, "zipf": self.word_score_func(w)} for w in words1]
        words1_zipfs = sorted(words1_zipfs, key=lambda x: x['zipf'])

        if len(words1_zipfs) == 0:
            words1_avg_zipfs = 0.0
        else:
            words1_avg_zipfs = np.mean([x['zipf'] for x in words1_zipfs])

        return words1_avg_zipfs

    def cal_score_median(self, text):
        words1 = nltk.tokenize.word_tokenize(text)
        words1 = set([w.lower() for w in words1 if self.is_good_word(w)])
        words1_zipfs = [{"w": w, "zipf": self.word_score_func(w)} for w in words1]
        words1_zipfs = sorted(words1_zipfs, key=lambda x: x['zipf'])

        if len(words1_zipfs) == 0:
            words1_avg_zipfs = 0.0
        else:
            words1_avg_zipfs = np.median([x['zipf'] for x in words1_zipfs])

        return words1_avg_zipfs

    def score_three_list(self, ref_sum, ori_sum, sim_sum, show_all=False):
        ts = time.time()
        total_score = dict()
        total_score['reference'] = []
        total_score['original'] = []
        total_score['simplify'] = []

        for text in ref_sum:
            score = self.cal_score(text)
            total_score['reference'].append(score)

        for text in ori_sum:
            score = self.cal_score(text)
            total_score['original'].append(score)

        for text in sim_sum:
            score = self.cal_score(text)
            total_score['simplify'].append(score)

        total_score['reference'] = np.array(total_score['reference'])
        total_score['original'] = np.array(total_score['original'])
        total_score['simplify'] = np.array(total_score['simplify'])

        ref_score = total_score['reference'].mean()
        ori_score = total_score['original'].mean()
        sim_score = total_score['simplify'].mean()

        print("Word Frequency")
        print("the higher, the easier \n")
        print("Reference summary = {:.3f}".format(ref_score))
        print("Original summary = {:.3f}".format(ori_score))
        print("Simplified summary = {:.3f}".format(sim_score))
        #print("Diff between simplified - original = {:.2f}".format(sim_score - ori_score))
        #print()
        #print("time spent in calculation:{}".format(timedelta(seconds=time.time() - ts)))

        return total_score


    def cal_score_median(self, text):
        words1 = nltk.tokenize.word_tokenize(text)
        words1 = set([w.lower() for w in words1 if self.is_good_word(w)])
        words1_zipfs = [{"w": w, "zipf": self.word_score_func(w)} for w in words1]
        words1_zipfs = sorted(words1_zipfs, key=lambda x: x['zipf'])

        if len(words1_zipfs) == 0:
            words1_avg_zipfs = 0.0
        else:
            words1_avg_zipfs = np.median([x['zipf'] for x in words1_zipfs])

        return words1_avg_zipfs

    def score(self, text_list, title="", printtop=True):
        ts = time.time()
        total_score = dict()
        total_score['reference'] = []

        for text in text_list:
            score = self.cal_score(text)
            total_score['reference'].append(score)

        total_score['reference'] = np.array(total_score['reference'])

        score = total_score['reference'].mean()

        if printtop:
            print("Word Frequency -- the higher, the easier \n")
        if title == "":
            print("score of {} = {:.3f}".format(argname("text_list"), score))
        else:
            print("score of {} = {:.3f}".format(title, score))
        return (score, total_score)



    def score_median(self, ref_sum, ori_sum, sim_sum, show_all=False):
        ts = time.time()
        total_score = dict()
        total_score['reference'] = []
        total_score['original'] = []
        total_score['simplify'] = []

        for text in ref_sum:
            score = self.cal_score_median(text)
            total_score['reference'].append(score)

        for text in ori_sum:
            score = self.cal_score_median(text)
            total_score['original'].append(score)

        for text in sim_sum:
            score = self.cal_score_median(text)
            total_score['simplify'].append(score)

        total_score['reference'] = np.array(total_score['reference'])
        total_score['original'] = np.array(total_score['original'])
        total_score['simplify'] = np.array(total_score['simplify'])

        ref_score = np.median(total_score['reference'])
        ori_score = np.median(total_score['original'])
        sim_score = np.median(total_score['simplify'])

        print("Word Frequency *** median ***")
        print("the higher, the easier \n")
        if show_all:
            print("Reference summary = {:.3f}".format(ref_score))
        print("Original summary = {:.3f}".format(ori_score))
        print("Simplified summary = {:.3f}".format(sim_score))
        print("Diff between simplified - original = {:.3f}".format(sim_score - ori_score))
        #print()
        #print("time spent in calculation:{}".format(timedelta(seconds=time.time() - ts)))

        return total_score


def cal_word_count(text_list, title="", printtop=True):
    ts = time.time()
    total_score = []
    remove_word = ["", " ",]

    for text in text_list:
        text = text.strip().split(' ')
        text = [word for word in text if word not in remove_word]
        total_score.append(len(text))

    score = np.mean(total_score)

    if printtop:
        print("Word Count \n")
    if title == "":
        print("Average number of words of {} = {:.3f}".format(argname("text_list"), score))
    else:
        print("Average number of words of {} = {:.3f}".format(title, score))
    return (score, total_score)


# Eval input from Coreference 
def embed_sent(text):
    remove_list = [".", ",", "'", "\"", '``', "'", ',', '-', '`', "''"]
    
    sent_list = []
    total_sent_emb = []
    sent_list = sent_tokenize(text)
    
    for puct in remove_list:
        if puct in sent_list:
            sent_list.remove(puct)
    
    total_sent_emb = [sbertmodel.encode(sent) for sent in sent_list]
    return total_sent_emb
    
# embed each sentence by sentence
def cosine_by_sentence_all_separate(text_list1, text_list2):
    final_score = []
    for text1, text2 in tqdm(zip(text_list1, text_list2)):

        text_emb1 = embed_sent(text1)
        text_emb2 = embed_sent(text2)
        score_list = []

        for i in range(len(text_emb1)):
            for j in range(len(text_emb2)):
                emb1 = text_emb1[i]
                emb2 = text_emb2[j]
                cos_sim_score = util.cos_sim(emb1, emb2)
                score = cos_sim_score.item()
                score_list.append(score)
        score_list = np.array(score_list)
        total_score = score_list.mean()
        final_score.append(total_score)
        
    final_score = np.array(final_score)
    avg_score = final_score.mean()
    return avg_score, final_score

# embed ref in one shot
def cosine_by_sentence_ref_oneshot(content, reference):
    final_score = []
    for text1, text2 in tqdm(zip(content, reference)):

        text_emb1 = embed_sent(text1)
        text_emb2 = sbertmodel.encode(text2)
        
        score_list = []

        for i in range(len(text_emb1)):
            emb1 = text_emb1[i]
            emb2 = text_emb2
            cos_sim_score = util.cos_sim(emb1, emb2)
            score = cos_sim_score.item()
            score_list.append(score)
        score_list = np.array(score_list)
        total_score = score_list.mean()
        final_score.append(total_score)
        
    final_score = np.array(final_score)
    avg_score = final_score.mean()
    return avg_score, final_score