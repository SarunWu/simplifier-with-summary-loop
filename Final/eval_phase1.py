import utils.utils_misc as utils_misc
import os

# check free GPU
freer_gpu = str(utils_misc.get_freer_gpu())
os.environ["CUDA_VISIBLE_DEVICES"] = ""+str(freer_gpu)
print("Using GPU "+str(freer_gpu))
 

from tqdm import tqdm
import pickle, torch, time
from datetime import datetime, timedelta
import pandas as pd

from utils.eval_rouge import cal_rouge
from utils.eval_simplicity import cal_word_count, cosine_by_sentence_ref_oneshot
import wandb

ts = time.time()

# Load data in pkl file to be evalauted
print( " ==================== " )
print("Loading Data")
file_path = "/home/jacky/research_boost/main/models/textrank/" 
name_to_open =  "extracted_data_11490news.pkl" 
name_to_save = name_to_open+"_phase1_" 

with open (file_path+name_to_open, "rb") as f:
    news_list = pickle.load(f)
num_test = len(news_list)

wandb.init(project="gen for eval")
wandb.run.name = "eval_"+name_to_save
wandb.run.save()


ref_list = [] 
sim_list = []

for i in range(len(news_list)):
    ref_sum = news_list[i]["ref_summary"]
    sim_sum = news_list[i]["sim_summary"]
    ref_list.append(ref_sum)
    sim_list.append(sim_sum)


print("Start to eval")
print( " ==================== " )

#ROUGE
rouge_score2 = cal_rouge(gold_list=ref_list, cand_list=sim_list)
print()
print( " ==================== " )

#Avg Word Count
wc_score3 = cal_word_count(text_list=sim_list, title="", printtop=False)
print()
print( " ==================== " )

# S-Bert Cosine for long sentence
sb_long, sb_long2 = cosine_by_sentence_ref_oneshot(content=sim_list, reference=ref_list)
print("SBERT Long Sentence =  ", sb_long)
print()
print( " ==================== " )
 
score_dict = dict()
score_dict['SBERT_LongSent'] = sb_long
score_dict['rouge1'] = rouge_score2['rouge1'].mean()
score_dict['rouge2'] = rouge_score2['rouge2'].mean()
score_dict['rougeL'] = rouge_score2['rougeL'].mean()
score_dict['AvgWordCount'] = wc_score3[0]


df = pd.DataFrame(score_dict, index=[0])

name_to_save_csv = "score_"+name_to_save+str(num_test)+"news.csv"
df.to_csv(file_path+name_to_save_csv, index=False)

print()
print("saved file : ", name_to_save_csv)
print("DONE! evaluation summary")
print("Total time used")
print(str(timedelta(seconds=time.time() - ts)))