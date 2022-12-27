# import utils.utils_misc as utils_misc
import os

# check free GPU
# freer_gpu = str(utils_misc.get_freer_gpu())
# os.environ["CUDA_VISIBLE_DEVICES"] = ""+str(freer_gpu)
# print("Using GPU "+str(freer_gpu))


from model_generator import GeneTransformer
from tqdm import tqdm
import pickle, torch, time
from datetime import datetime, timedelta
from utils.eval_rouge import cal_rouge 


# load model for original summary
"""model_name_original = "summarizer_train0214_reference_ckpt.bin"
generator_original = GeneTransformer(device="cuda") # Initialize the generator
generator_original.reload("models/run_finished/{}".format(model_name_original))
generator_original.eval()"""

#to run:  python /home/alexjcortes/jacky_code/NLP_Project/generate_summary_for_eval50tokens.py

# load model for simplified summary
model_name_sim = "summarizer_summarizer_coverage_50token_0615_ckpt.bin"
generator_sim = GeneTransformer(device="cuda")
generator_sim.reload("/Final/NLP_Project/{}".format(model_name_sim))
generator_sim.eval()


# load test dataset (pickle)
with open ("/Final/NLP_Project/test_dataset.pkl", "rb") as f:
    news_list = pickle.load(f)

num_test = len(news_list)
name_to_save = "coverage_40token_0615_"+str(num_test)+"news.pkl"

#print("model for original summary -> ", model_name_original, '\n')
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
    content = news['content']
    ref_summary = news['summary']
    
    #print("original summary")
    # generate original summary
    #ori_summary = generator_original.decode([news['content']], max_output_length=25, beam_size=1, return_scores=False, sample=False)
    #if ori_summary == "":
    #    ori_summary = "None"
    #else:
    #    ori_summary = ori_summary[0][0]

    #print("simplified summary")
    # generate simplified summary
    sim_summary = generator_sim.decode([news['content']], max_output_length=40, beam_size=1, return_scores=False, sample=False)
    if sim_summary == "":
        sim_summary = "None"
    else:
        sim_summary = sim_summary[0][0]

    # combine all into dict
    news_dict["content"] = content
    news_dict["ref_summary"] = ref_summary
    #news_dict["ori_summary"] = ori_summary
    news_dict["sim_summary"] = sim_summary
    
    # append to a global list
    total_news_list.append(news_dict)


print()
print( " ==================== " )
print("DONE! generate original and simplified summary")

# save to file
with open("/Final/NLP_Project/saved/"+name_to_save, 'wb') as f:
    pickle.dump(total_news_list, f)
print("saved file : ", name_to_save)

print("Total time used")
print(str(timedelta(seconds=time.time() - ts)))