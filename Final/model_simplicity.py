import torch, os, sys, time
import nltk
import spacy
import utils.utils_textstatjk as dc
from bert_score import BERTScorer
from wordfreq import zipf_frequency
import numpy as np
import sentence_transformers as sbert 
import textstat


dale_chall = dc.textstatistics()
sp = spacy.load("en_core_web_sm")
#bert_scorer = BERTScorer(lang="en", rescale_with_baseline=True)
#bert_scorer = BERTScorer(lang='en', model_type='bert-base-uncased', num_layers=8, nthreads=8, batch_size=256, rescale_with_baseline=True)
#bert_scorer_deberta_large_mnli = BERTScorer(lang="en", rescale_with_baseline=True, model_type="microsoft/deberta-large-mnli")


STOP_WORDS = set(["'", ".", "!", "?", ",", '"', '-', 'we', 'our', 'you', 'he', 'him', 'she', 'her', 'it', "it's", 'its', 'they', 'their', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'a', 'an', 'the', 'and', 'or', 'as', 'of', 'at', 'by', 'to', 'not', 'so', "'s", "in", "for", "with", "on"])

class DiffWordReward:

    def __init__(self):
        self.stop_words = "STOP_WORDS"  
    
    def count_diffword(self, doc):
        doc = sp(doc)
        diffdoc = [] 
        #print(len(doc))
        for i in range(len(doc)):
            word = doc[i].text
            wordlemma = doc[i].lemma_.lower()
            wordpos = doc[i].pos_
            wordner = doc[i].ent_iob_
            #print(word, wordlemma, wordpos, wordner)

            # check if token is NER
            if doc[i].ent_iob_ != "O":
                continue

            # diff word
            if dale_chall.dcis_difficult_word(wordlemma):
                diffdoc.append(word)
                #print(i, diffdoc)
        num_diff_word = float(len(diffdoc))
        #print("Difficult words\n")
        #print(diffdoc)
        #print(num_diff_word)
        return num_diff_word


    def score(self, summaries, bodies, bodies_tokenized=None, lengths=None, extra=None):
        
        referenced_summaries = bodies
        final_score = []
        
        # store number of diff word
        sampled_diff_word = []
        referenced_diff_word = []

        # count diff word for each summary
        for sampled, referenced in zip(summaries, referenced_summaries):
            sampled_word = self.count_diffword(sampled)
            referenced_word = self.count_diffword(referenced)
            
            # add to list
            sampled_diff_word.append(sampled_word)
            referenced_diff_word.append(referenced_word)
            
            #print("sampled_diff_word", sampled_diff_word)
            #print("referenced_diff_word", referenced_diff_word)
        
        # calculate score
        for sampled, referenced in zip(sampled_diff_word, referenced_diff_word):
            if sampled > referenced:
                score = -1.0
            elif (sampled == 0) and (referenced == 0):
                score = 0.0
            elif (sampled == referenced) and (referenced != 0):
                score = -1.0
            else:
                score = float(1 - (sampled / referenced))

            final_score.append(score)
            
        return final_score, None

        #print("sampled_diff_word", sampled_diff_word)
        #print("referenced_diff_word", referenced_diff_word)
        #print("final_score", final_score)
        


"""
            print(sampled)
            print("sampled_word", sampled_word)
            print("sampled_diff_word", sampled_diff_word)
            print("\n")
            
            print(referenced)
            print("referenced_word", referenced_word)
            print("referenced_diff_word", referenced_diff_word)
            print("\n\n\n")
"""

class DiffWordPenalty:

    def __init__(self):
        self.stop_words = "STOP_WORDS"
        
    def count_diffword(self, doc):
        doc = sp(doc)
        diffdoc = [] 
        #print(len(doc))
        for i in range(len(doc)):
            word = doc[i].text
            wordlemma = doc[i].lemma_.lower()
            wordpos = doc[i].pos_
            wordner = doc[i].ent_iob_
            #print(word, wordlemma, wordpos, wordner)

            # check if token is NER
            if doc[i].ent_iob_ != "O":
                continue

            # diff word
            if dale_chall.dcis_difficult_word(wordlemma):
                diffdoc.append(word)
                #print(i, diffdoc)
        num_diff_word = float(len(diffdoc))
        #print("Difficult words\n")
        #print(diffdoc)
        #print(num_diff_word)
        return num_diff_word


    def score(self, summaries, bodies, bodies_tokenized=None, lengths=None, extra=None):
        
        referenced_summaries = bodies
        final_score = []
        
        # store number of diff word
        sampled_diff_word = []
        referenced_diff_word = []

        # count diff word for each summary
        for sampled, referenced in zip(summaries, referenced_summaries):
            sampled_word = self.count_diffword(sampled)
            referenced_word = self.count_diffword(referenced)
            
            # add to list
            sampled_diff_word.append(sampled_word)
            referenced_diff_word.append(referenced_word)
            
            #print("sampled_diff_word", sampled_diff_word)
            #print("referenced_diff_word", referenced_diff_word)
        
        # calculate score
        for sampled, referenced in zip(sampled_diff_word, referenced_diff_word):
            if sampled > referenced:
                score = 1.0
            elif (referenced == 0) and (sampled == 0): #should we change this one?
                score = 0.0
            else:
                score = 0.0
            final_score.append(score)
        
        return final_score, None



        #print("sampled_diff_word", sampled_diff_word)
        #print("referenced_diff_word", referenced_diff_word)
        #print("final_score", final_score)

class DallchallReward:

    def __init__(self):
        self.stop_words = "STOP_WORDS"

    def score(self, summaries, bodies, bodies_tokenized=None, lengths=None, extra=None):
        
        referenced_summaries = bodies
        
        final_score = []
        
        # store readability score
        sampled_readability = []
        referenced_readability = []

        # calculate readability score for each summary
        for sampled, referenced in zip(summaries, referenced_summaries):
            sam_read = float(dale_chall.dale_chall_readability_score(sampled))
            ref_read = float(dale_chall.dale_chall_readability_score(referenced))
            
            # add to list
            sampled_readability.append(sam_read)
            referenced_readability.append(ref_read)
            
            #print("sampled_diff_word", sampled_diff_word)
            #print("referenced_diff_word", referenced_diff_word)
        
        # calculate score
        for sampled, referenced in zip(sampled_readability, referenced_readability):
            score = float((referenced - sampled) / 2)

            if score > 1.0:
                score = 1.0
            elif score < -1.0:
                score = -1.0

            final_score.append(score)
            
        return final_score, None

        #print("sampled_readability", sampled_readability)
        #print("referenced_readability", referenced_readability)
        #print("final_score", final_score)
        

class DallchallPenalty:

    def __init__(self):
        self.stop_words = "STOP_WORDS"

    def score(self, summaries, bodies, bodies_tokenized=None, lengths=None, extra=None):
        
        referenced_summaries = bodies
        
        final_score = []
        
        # store readability score
        sampled_readability = []
        referenced_readability = []

        # count diff word for each summary
        for sampled, referenced in zip(summaries, referenced_summaries):
            sam_read = float(dale_chall.dale_chall_readability_score(sampled))
            ref_read = float(dale_chall.dale_chall_readability_score(referenced))
            
            # add to list
            sampled_readability.append(sam_read)
            referenced_readability.append(ref_read)
            
            #print("sampled_diff_word", sampled_diff_word)
            #print("referenced_diff_word", referenced_diff_word)
        
        # calculate score
        for sampled, referenced in zip(sampled_readability, referenced_readability):
            if sampled > referenced:
                score = 1.0
            else:
                score = 0.0
            final_score.append(score)
            
        return final_score, None

        #print("sampled_readability", sampled_readability)
        #print("referenced_readability", referenced_readability)
        #print("final_score", final_score)


class SentenceBERTCosine:

    def __init__(self, threshold=0.0):
        self.stop_words = "STOP_WORDS"
        self.model = sbert.SentenceTransformer(model_name_or_path='all-MiniLM-L6-v2', device='cuda')
        self.threshold = threshold

    def score(self, summaries, bodies, bodies_tokenized=None, lengths=None, extra=None):
        
        referenced_summaries = bodies
        final_score = []

        # hyps = sampled, refs = referenced
        for hyps, refs in zip(summaries, referenced_summaries):
            #input to function must be list
            emb1 = self.model.encode(hyps)
            emb2 = self.model.encode(refs)
            cos_similarity = sbert.util.cos_sim(emb1, emb2)
            score = cos_similarity.item()

            if (self.threshold != 0.0) and (score < self.threshold):
                score = 0.0

            final_score.append(score)
        return final_score, None


class SemanticBERTScore:

    def __init__(self, threshold=0.0):
        self.stop_words = "STOP_WORDS"
        self.threshold = threshold
        self.bert_scorer = BERTScorer(lang='en', model_type='bert-base-uncased', num_layers=8, nthreads=8, rescale_with_baseline=True)

    def score(self, summaries, bodies, bodies_tokenized=None, lengths=None, extra=None):
        
        referenced_summaries = bodies
        final_score = []

        # hyps = sampled, refs = referenced
        for hyps, refs in zip(summaries, referenced_summaries):
            #input to function must be list
            P, R, F1 = self.bert_scorer.score([hyps], [refs])
            score = F1.item()

            if (self.threshold != 0.0) and (score < self.threshold):
                score = 0.0
            
            final_score.append(score)
        return final_score, None



"""
            print("\n\nhyps = ", hyps)
            print("\n\nhyps type = ", type(hyps))
            
            print("\nrefs = ", refs)
            print("\n\nrefs type = ", type(refs))

            print("\n\nP = ", P)
            print("R = ", R)
            print("F1 = ", F1)
            print("score = ", score)
        
        print("final_score = ", final_score)
"""


class WordFreqScore:
    def __init__(self, target_shift=0.4, word_change_ratio=0.1):
        self.target_shift = target_shift
        self.word_change_ratio = word_change_ratio # Number of words that we expect to be swapped
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

    def vocab_shift_score(self, ref_sum, gen_sum, printing=False):
        words1 = nltk.tokenize.word_tokenize(ref_sum)
        # print("words1 = ", words1)
        words2 = nltk.tokenize.word_tokenize(gen_sum)
        # print("words2 = ", words2)

        words1 = set([w.lower() for w in words1 if self.is_good_word(w)])
        # printprint("words1 = ", words1)
        words2 = set([w.lower() for w in words2 if self.is_good_word(w)])
        #print("words2 = ", words2)

        words1_zipfs = [{"w": w, "zipf": self.word_score_func(w)} for w in words1]
        #print("words1_zipfs = ", words1_zipfs)
        words2_zipfs = [{"w": w, "zipf": self.word_score_func(w)} for w in words2]
        #print("words2_zipfs = ", words2_zipfs)

        words1_zipfs = sorted(words1_zipfs, key=lambda x: x['zipf'])
        words2_zipfs = sorted(words2_zipfs, key=lambda x: x['zipf'])
        #print("\n\n")
        #print("words1_zipfs = ", words1_zipfs)
        #print("words2_zipfs = ", words2_zipfs)

        if len(words1_zipfs) == 0:
            words1_avg_zipfs = 0.0
        else:
            words1_avg_zipfs = np.mean([x['zipf'] for x in words1_zipfs])

        if len(words2_zipfs) == 0:
            words2_avg_zipfs = 0.0
        else:
            words2_avg_zipfs = np.mean([x['zipf'] for x in words2_zipfs])
        
        if printing:
            #print("Desired # word swaps: %d" % (target_n_words))
            print("[Avg Zipf: %.3f] words1 :" % (words1_avg_zipfs), words1)
            print("\n\n")
            print("[Avg Zipf: %.3f] words2 :" % (words2_avg_zipfs), words2)

        return words1_avg_zipfs, words2_avg_zipfs
    
    
    def shift_to_score(self, diff_zipfs, target_shift, right_slope=0.25):
        
        if diff_zipfs <= target_shift:
            score = diff_zipfs / (target_shift+0.001)
        else:
            score = 1.0 - right_slope * (diff_zipfs - target_shift) / (target_shift+0.001)
        return np.clip(score, 0.0, 1.0)

    
    def score(self, summaries, bodies, bodies_tokenized=None, lengths=None, extra=None):
        
        referenced_summaries = bodies
        final_score = []
        
        for sampled, referenced in zip(summaries, referenced_summaries):
            ref_zipfs, gen_zipfs = self.vocab_shift_score(ref_sum=referenced, gen_sum=sampled, printing=False)
            diff_zipfs = gen_zipfs - ref_zipfs
            #score = self.shift_to_score(diff_zipfs=diff_zipfs, target_shift=self.target_shift)
            score = float(diff_zipfs/self.target_shift)
            
            #if score > 1.0:
            #    score = 1.0
            #elif score < 0.0:
            #    score = 0.0
            score = float(np.clip(score, 0.0, 1.0))
            final_score.append(score)
            
        return final_score, None



class WordFreqScorePenalty:
    def __init__(self, target_shift=0.4, word_change_ratio=0.1):
        self.target_shift = target_shift
        self.word_change_ratio = word_change_ratio # Number of words that we expect to be swapped
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

    def vocab_shift_score(self, ref_sum, gen_sum, printing=False):
        words1 = nltk.tokenize.word_tokenize(ref_sum)
        # print("words1 = ", words1)
        words2 = nltk.tokenize.word_tokenize(gen_sum)
        # print("words2 = ", words2)

        words1 = set([w.lower() for w in words1 if self.is_good_word(w)])
        # printprint("words1 = ", words1)
        words2 = set([w.lower() for w in words2 if self.is_good_word(w)])
        #print("words2 = ", words2)

        words1_zipfs = [{"w": w, "zipf": self.word_score_func(w)} for w in words1]
        #print("words1_zipfs = ", words1_zipfs)
        words2_zipfs = [{"w": w, "zipf": self.word_score_func(w)} for w in words2]
        #print("words2_zipfs = ", words2_zipfs)

        words1_zipfs = sorted(words1_zipfs, key=lambda x: x['zipf'])
        words2_zipfs = sorted(words2_zipfs, key=lambda x: x['zipf'])
        #print("\n\n")
        #print("words1_zipfs = ", words1_zipfs)
        #print("words2_zipfs = ", words2_zipfs)

        if len(words1_zipfs) == 0:
            words1_avg_zipfs = 0.0
        else:
            words1_avg_zipfs = np.mean([x['zipf'] for x in words1_zipfs])

        if len(words2_zipfs) == 0:
            words2_avg_zipfs = 0.0
        else:
            words2_avg_zipfs = np.mean([x['zipf'] for x in words2_zipfs])
        
        if printing:
            #print("Desired # word swaps: %d" % (target_n_words))
            print("[Avg Zipf: %.3f] words1 :" % (words1_avg_zipfs), words1)
            print("\n\n")
            print("[Avg Zipf: %.3f] words2 :" % (words2_avg_zipfs), words2)

        return words1_avg_zipfs, words2_avg_zipfs
    
    
    def shift_to_score(self, diff_zipfs, target_shift, right_slope=0.25):
        
        if diff_zipfs <= target_shift:
            score = diff_zipfs / (target_shift+0.001)
        else:
            score = 1.0 - right_slope * (diff_zipfs - target_shift) / (target_shift+0.001)
        return float(np.clip(score, 0.0, 1.0))

    
    def score(self, summaries, bodies, bodies_tokenized=None, lengths=None, extra=None):
        
        referenced_summaries = bodies
        final_score = []
        
        for sampled, referenced in zip(summaries, referenced_summaries):
            ref_zipfs, gen_zipfs = self.vocab_shift_score(ref_sum=referenced, gen_sum=sampled, printing=False)
            diff_zipfs = gen_zipfs - ref_zipfs
            score = self.shift_to_score(diff_zipfs=diff_zipfs, target_shift=self.target_shift)
            #score = float(diff_zipfs/self.target_shift)
            
            #if score > 1.0:
            #    score = 1.0
            #elif score < 0.0:
            #    score = 0.0
            #score = float(np.clip(score, 0.0, 1.0))
            final_score.append(score)
            
        return final_score, None


class FKGLScore:
    def __init__(self):
        self.stopws = set(nltk.corpus.stopwords.words("english") + ["might", "would", "``", "-", "--"])

    def fkgl_score(self, text1, text2):
        score1 = textstat.flesch_kincaid_grade(text1)
        score2 = textstat.flesch_kincaid_grade(text2)
        return score1, score2
    
    def score(self, summaries, bodies, bodies_tokenized=None, lengths=None, extra=None):
        
        referenced_summaries = bodies
        final_score = []
        
        for sampled, referenced in zip(summaries, referenced_summaries):
            # find FKGL of each summary and calculate the difference
            ref_grade, gen_grade = self.fkgl_score(text1=referenced, text2=sampled)
            diff_grade = ref_grade - gen_grade

            # target is the reference (Phase1 summary) - threshold_grade which is based on FKGL of ref in CNNDM
            threshold_grade = 6.60

            if ref_grade > threshold_grade:
                target_shift = float(ref_grade - threshold_grade)
            else:
                target_shift = float(0.60)

            score = float(diff_grade/target_shift)
            score = float(np.clip(score, 0.0, 1.0))
            final_score.append(score)
            
        return final_score, None