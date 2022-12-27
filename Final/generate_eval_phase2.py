import utils.utils_misc as utils_misc
import os

# check free GPU
freer_gpu = str(utils_misc.get_freer_gpu())
os.environ["CUDA_VISIBLE_DEVICES"] = ""+str(freer_gpu)
print("Using GPU "+str(freer_gpu))


from model_generator import GeneTransformer
from tqdm import tqdm
import pickle, torch, time
from datetime import datetime, timedelta
import pandas as pd

from utils.eval_rouge import cal_rouge
from utils.eval_simplicity import cal_wordfreq, cal_word_count, cal_SBERT_cosine_score
import wandb

# load model for simplified summary
model_name_sim = "Final_p1_summarizer_ckpt.bin"
file_path = "/home/jacky/research_boost/final_thesis/models/" 
name_to_save = "test_final_thesis" 
sum_output_len = 25

wandb.init(project="gen for eval")
wandb.run.name = name_to_save
wandb.run.save()

generator_sim = GeneTransformer(device="cuda")
generator_sim.reload(file_path+"{}".format(model_name_sim))
generator_sim.eval()

# load test dataset (pickle)
with open ("/home/jacky/research_boost/data/test_extracted_data.pkl", "rb") as f:
    news_list = pickle.load(f)

num_test = len(news_list)
name_to_save_pkl = "summary_"+name_to_save+str(num_test)+"news.pkl"

print("model for simplified summary -> ", model_name_sim, '\n')
print("number of articls to process -> ", num_test, '\n')
print("name_to_save -> ", name_to_save, '\n')

total_news_list = []

ts = time.time()

# generate summary
for idx, news in tqdm(enumerate(news_list)):
    if idx == num_test:
        break
        
    # progress printing
    if (idx !=0) and (idx % 100 == 0):
        print("Writing story {} of {}; {:.2f} percent done. Time spent: {}".format(idx, num_test, float(idx)*100.0/float(num_test), timedelta(seconds=time.time() - ts)))
        
    news_dict = dict()
    ref_summary = news['ref_summary']

    # generate simplified summary
    sim_summary = generator_sim.decode([news['tr_content']], max_output_length=sum_output_len, beam_size=1, return_scores=False, sample=False)

    if sim_summary == "":
        sim_summary = "None"
    else:
        sim_summary = sim_summary[0][0]

    # combine all into dict
    news_dict["ref_summary"] = ref_summary
    news_dict["sim_summary"] = sim_summary

    # append to a global list
    total_news_list.append(news_dict)

print()
print( " ==================== " )
print("DONE! simplified summary")

# save to file
with open(file_path+name_to_save_pkl, 'wb') as f:
    pickle.dump(total_news_list, f)
print("saved file : ", name_to_save_pkl)

print("Total time used")
print(str(timedelta(seconds=time.time() - ts)))


print()
print( " ==================== " )
print("Start to eval")

ref_list = []
sim_list = []

for i in range(len(total_news_list)):
    ref_sum = total_news_list[i]["ref_summary"]
    sim_sum = total_news_list[i]["sim_summary"]
    ref_list.append(ref_sum)
    sim_list.append(sim_sum)

#S-BERT cosine
score = cal_SBERT_cosine_score(ref_text=ref_list, pred_text=sim_list)
print()
print( " ==================== " )

#Word freq
cal_wf = cal_wordfreq()
wordfreq_score3 = cal_wf.score(text_list=sim_list, title="", printtop=True)
print()
print( " ==================== " )

#ROUGE
rouge_score2 = cal_rouge(gold_list=ref_list, cand_list=sim_list)
print()
print( " ==================== " )

#Avg Word Count
wc_score3 = cal_word_count(text_list=sim_list, title="", printtop=False)
print()
print( " ==================== " )

score_dict = dict()
score_dict['WF'] = wordfreq_score3[0]
score_dict['SBERT'] = score['sbert'].mean()
score_dict['AvgWordCount'] = wc_score3[0]
score_dict['rouge1'] = rouge_score2['rouge1'].mean()
score_dict['rouge2'] = rouge_score2['rouge2'].mean()
score_dict['rougeL'] = rouge_score2['rougeL'].mean()

df = pd.DataFrame(score_dict, index=[0])

name_to_save_csv = "score_"+name_to_save+str(num_test)+"news.csv"
df.to_csv(file_path+name_to_save_csv, index=False)

print()
print("saved file : ", name_to_save_csv)
print("DONE! evaluation summary")
print("Total time used")
print(str(timedelta(seconds=time.time() - ts)))