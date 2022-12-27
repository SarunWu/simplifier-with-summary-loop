import os, utils.utils_misc

# freer_gpu = str(utils.utils_misc.get_freer_gpu())
# os.environ["CUDA_VISIBLE_DEVICES"] = ""+str(freer_gpu)

from torch.utils.data import DataLoader, RandomSampler
import torch, sys, time, argparse, numpy as np
from utils.utils_dataset import SQLDataset, HDF5Dataset
from transformers.optimization import AdamW
from model_generator import GeneTransformer

from datetime import datetime, timedelta
from utils.utils_logplot import LogPlot
import utils.utils_tokenizer
 
from model_coverage import KeywordCoverage
from model_guardrails import PatternPenalty, LengthPenalty, RepeatPenalty
import threading, queue
import wandb

user = os.getlogin()

parser = argparse.ArgumentParser()
parser.add_argument("--experiment", type=str, required=True, help="Experiment name. Will be used to save a model file and a log file.")
parser.add_argument("--dataset_file", type=str, required=True, help="Which dataset file to use. Can be full path or the root folder will be attached.")
parser.add_argument("--initial_summarizer", type=str, required=True, help="Specify the name of initial summarizer to start.")

# parser.add_argument("--root_folder", type=str, default="/home/jacky/research_boost/final_thesis/")
parser.add_argument("--root_folder", type=str, default="/home/alexjcortes/jacky_code/")
parser.add_argument("--train_batch_size", type=int, default=4, help="Training batch size.")
parser.add_argument("--n_epochs", type=int, default=4, help="Number of epochs to run over the data.")
parser.add_argument("--optim_every", type=int, default=4, help="Optimize every x backprops. A multiplier to the true batch size.")
parser.add_argument("--max_output_length", type=int, default=25, help="Maximum output length. Saves time if the sequences are short.")
parser.add_argument("--save_log_every", type=int, default=60, help="Number of seconds between any two saves.")
parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
parser.add_argument('--fp16', action='store_true', help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
parser.add_argument("--ckpt_every", type=int, default=600, help="If 0, checkpointing is not used. Otherwise, checkpointing is done very x seconds.")
parser.add_argument("--ckpt_lookback", type=int, default=300, help="When checkpointing, will consider the avg total score of the last x samples.")

args = parser.parse_args()


models_folder = args.root_folder + "models/"
log_folder = args.root_folder + "logs/"

# Select initial Summarizer 
summarizer_model_start = os.path.join(models_folder, args.initial_summarizer)

print("\n\nThis is phrase 1 training")
print("\nThe initial summerizer is: ", args.initial_summarizer)
print("\nIt will be loaded from : ", summarizer_model_start)
print("\n\n*** Carefully check the initial model ***")
time.sleep(3)

# if args.device == "cuda":
#     print("Working on GPU "+str(freer_gpu))
# print("---------------\n")

total_score_history = []
best_ckpt_score = None
learning_rate = 2e-5
n_epochs = args.n_epochs

ckpt_every = args.ckpt_every
ckpt_lookback = int((args.ckpt_lookback+args.train_batch_size-1)/args.train_batch_size)
print("consider avg total score of last [ ", ckpt_lookback, " ] samples for checkpoint")


ckpt_file = os.path.join(models_folder, "summarizer_"+args.experiment+"_ckpt.bin")
ckpt_optimizer_file = os.path.join(models_folder, "summarizer_optimizer_"+args.experiment+"_ckpt.bin")
print("Model check point file name: ", ckpt_file)
print("Optimizer check point file name: ", ckpt_optimizer_file)
print("---------------\n")


bert_tokenizer = utils.utils_tokenizer.BERTCacheTokenizer()
print("bert_tokenizer loaded")
print("---------------\n")

summarizer = GeneTransformer(max_output_length=args.max_output_length, device=args.device, tokenizer_type='gpt2', starter_model=summarizer_model_start)
print("Summarizer loaded")
print("---------------\n")

def collate_func(inps):
    if ".db" in args.dataset_file:
        return [a['body'] for a in inps]
    else:
        return [inp[0].decode() for inp in inps]

param_optimizer = list(summarizer.model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
print("Parameters loaded")
print("---------------\n")

logplot_file = os.path.join(log_folder, "summarizer_%s.log" % (args.experiment))
logplot = LogPlot(logplot_file)

time_save = time.time()
time_ckpt = time.time()

if args.fp16:
    try:
        from apex import amp
    except ImportError:
        raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
    summarizer.model, optimizer = amp.initialize(summarizer.model, optimizer, opt_level="O1") # For now O1. See details at https://nvidia.github.io/apex/amp.html


# (Optional) Reload checkpoint
# print("  <<< Loading previous checkpoint >>>  ")
# previous_summarizer_ckpt = os.path.join(models_folder, "p1_summarizer_COREF_0806_reload2_28per_ckpt.bin")
# previous_optimizer_ckpt =  os.path.join(models_folder, "p1_summarizer_optimizer_COREF_0806_reload2_28per_ckpt.bin")
# summarizer.model.load_state_dict(torch.load(previous_summarizer_ckpt))
# optimizer.load_state_dict(torch.load(previous_optimizer_ckpt))
# print("previous_summarizer_ckpt --> ", previous_summarizer_ckpt)
# print("previous_optimizer_ckpt --> ", previous_optimizer_ckpt)


print("Loading scorers")
coverage_model_file = os.path.join(models_folder, "bert_coverage.bin")
fluency_news_model_file = os.path.join(models_folder, "fluency_news_bs32.bin")

scorers = [
          {"name": "coverage", "importance": 10.0, "sign": 1.0, "model": KeywordCoverage(args.device, model_file=coverage_model_file, n_kws=9)}, # keyword_model_file=coverage_keyword_model_file,
          {"name": "fluency", "importance": 2.0, "sign": 1.0, "model": GeneTransformer(max_output_length=args.max_output_length, device=args.device, starter_model=fluency_news_model_file)},
          {"name": "patpen", "importance": 5.0, "sign": -1.0, "model": PatternPenalty()},
          {"name": "lengthpen", "importance": 2.0, "sign": -1.0, "model": LengthPenalty(args.max_output_length)},
          {"name": "reppen", "importance": 2.0, "sign": -1.0, "model": RepeatPenalty()},
           ]

print("All scorers loaded")
print("---------------\n")

def background_tokenizer(bodies, out_queue):
    out_queue.put([bert_tokenizer.encode(body) for body in bodies])

my_queue = queue.Queue()

if ".db" in args.dataset_file:
    all_dataset = SQLDataset(args.dataset_file)
else:
    all_dataset = HDF5Dataset(args.dataset_file, collection_name="name")

dataset = all_dataset
dataloader = DataLoader(dataset=dataset, batch_size=args.train_batch_size, sampler=RandomSampler(dataset), drop_last=True, collate_fn=collate_func)
total_batch = (len(dataset)) // args.train_batch_size

print("Dataset size: ", len(dataset))
print("Batch size: ", args.train_batch_size)
print("Total batch: ", total_batch)
print("---------------\n")

wandb.init(project="p1_coref")
wandb.config.update({"learning_rate": learning_rate, "ckpt_file": ckpt_file, "ckpt_optimizer_file":ckpt_optimizer_file, "len_dataset": len(dataset),
                    "total_batch": total_batch, "start_time": datetime.now(), "ckpt_lookback_sample": ckpt_lookback})
wandb.config.update(args)
wandb.run.name = args.experiment
wandb.run.save()


# Function for Factual Consistency from Robin's Thesis (We don't use this in our thesis)
def not_trigger_faith(log_obj, total_sampled_scores, start_training):
    if  (time.time() - start_training) < 86400: # 86400=second/day, trigger after 1-day of training
        return True

    # only triggers when summary quality is good enough
    current_total_score = torch.mean(total_sampled_scores).item()
    if len(log_obj) == 0 or current_total_score < 4:
        return True
    else:
        return ( log_obj["reppen_score"] > 0 or
                 log_obj["patpen_score"] > 0 or
                 log_obj["lengthpen_score"] == 1 or
                 log_obj['fluency_score']  < 0.3 or
                 log_obj['coverage_score'] < 0.3
                )

print("Started training")
print("---------------\n")

start_training = time.time()

for epi in range(n_epochs):
    print("=================== EPOCH",epi, "===================")
    for ib, documents in enumerate(dataloader):
        Timer = {}

        T1 = time.time()
        log_obj = {}

        bodies = [" ".join(doc.split(" ")[:300]) for doc in documents if len(doc) > 0 ]

        # We run tokenization in the background, as it is BERT tokenization only used after the summarizer has run. Saves about 5% of time.
        thread1 = threading.Thread(target = background_tokenizer, args = (bodies, my_queue))
        # bodies_bert_tokenized = [bert_tokenizer.enncode(body) for body in bodies] # This is the not background version
        thread1.start()

        T2 = time.time()
        Timer["preprocessing_starting"] = T2-T1

        # T1b = time.time()
        sampled_summaries, sampled_logprobs, sampled_tokens, input_past, sampled_end_idxs = summarizer.decode_batch(bodies, max_output_length=args.max_output_length, return_logprobs=True, sample=True)

        T3 = time.time()
        Timer["generator_sampled"] = T3-T2
        with torch.no_grad():
            argmax_summaries, argmax_end_idxs = summarizer.decode_batch(bodies, max_output_length=args.max_output_length, input_past=input_past)
        T4 = time.time()
        Timer["generator_argmax"] = T4-T3

        selected_logprobs = torch.sum(sampled_logprobs, dim=1)
        batch_size, seq_length = sampled_logprobs.shape

        # We join it here, saying the tokenization that's been running in the background should be done by now.
        thread1.join()
        bodies_bert_tokenized = my_queue.get()

        scores_track = {}
        total_sampled_scores = torch.FloatTensor([0.0] * batch_size).to(args.device)
        total_argmax_scores = torch.FloatTensor([0.0] * batch_size).to(args.device)
        for scorer in scorers:
            #if scorer['name'] == "faith" and not_trigger_faith(log_obj, total_sampled_scores, start_training):
            #    continue # when summary quality is low, not worth to cal faithfulness, usually 0
            T = time.time()

            sampled_scores, extra = scorer['model'].score(sampled_summaries, bodies, bodies_tokenized=bodies_bert_tokenized, extra=None, lengths=sampled_end_idxs)
            sampled_scores = torch.FloatTensor(sampled_scores).to(args.device)

            argmax_scores, _ = scorer['model'].score(argmax_summaries, bodies, bodies_tokenized=bodies_bert_tokenized, extra=extra, lengths=argmax_end_idxs)
            argmax_scores  = torch.FloatTensor(argmax_scores).to(args.device)

            Timer["scores_"+scorer['name']] = time.time()-T
            total_sampled_scores += (scorer['sign'])*(scorer['importance'])*sampled_scores
            total_argmax_scores  += (scorer['sign'])*(scorer['importance'])*argmax_scores
            log_obj[scorer['name']+"_score"] = sampled_scores.mean().item()
            scores_track[scorer['name']+"_scores"] = sampled_scores

        T5 = time.time()
        Timer['all_scores'] = T5-T4
        Loss = torch.mean((total_argmax_scores - total_sampled_scores) * selected_logprobs)

        if args.fp16:
            with amp.scale_loss(Loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            Loss.backward()

        T6 = time.time()
        Timer['backward'] = T6-T5

        if ib%args.optim_every == 0:
            optimizer.step()
            optimizer.zero_grad()

        T7 = time.time()
        Timer['optim'] = T7-T6

        # log_obj['summary_nwords'] = int(np.mean([summ.count(" ")+1 for summ in sampled_summaries]))
        avg_total = total_sampled_scores.mean().item()

        total_score_history.append(avg_total)
        log_obj['summary_nwords'] = int(np.mean(sampled_end_idxs))
        log_obj['loss'] = Loss.item()
        log_obj['total_score'] = avg_total
        log_obj['count'] = batch_size
        logplot.cache(log_obj, prefix="T_")

        Tfinal = time.time()
        Timer['total'] = Tfinal - T1

        total_runtime = str(timedelta(seconds=time.time() - start_training))
        # print(Timer)

        if (time.time()-time_save > args.save_log_every):
            print("\n\n==========================================")
            print("Epoch: ", epi, "\tbatch ", ib, " from ", total_batch, " >>> %.2f %%" % ((ib+1)/total_batch*100))
            print("-----------")
            print(bodies[0])
            print("-----------")
            print(sampled_summaries[0])
            print("-----------")
            print("Total score:", total_sampled_scores[0].item())
            for scorer in scorers:
                #if scorer['name'] == "faith" and (scorer['name']+"_scores") not in scores_track:
                #    continue
                print(scorer['name']+" score:", scores_track[scorer['name']+"_scores"][0].item())
            print("-----------")
            print("Total time used: ", total_runtime)
            print("-----------")

            logplot.save(total_runtime=total_runtime, printing=True)
            # print(Timer)

            time_save = time.time()
            print("==========================================")

        if ckpt_every > 0 and len(total_score_history) > ckpt_lookback:
            current_score = np.mean(total_score_history[-ckpt_lookback:])

            if time.time()-time_ckpt > ckpt_every:
                revert_ckpt = best_ckpt_score is not None and current_score < min(1.2*best_ckpt_score, 0.8*best_ckpt_score) # Could be negative or positive
                print("\n\n================================== CKPT TIME, "+str(datetime.now())+" =================================")
                print("Epoch: ", epi, "\tbatch ", ib, " from ", total_batch, " >>> %.2f %%" % ((ib+1)/total_batch*100))
                print("Previous best:", best_ckpt_score)
                print("Current Score:", current_score)
                print("[CKPT] Am I reverting?", ("yes" if revert_ckpt else "no! BEST CKPT"))
                if revert_ckpt:
                    summarizer.model.load_state_dict(torch.load(ckpt_file))
                    optimizer.load_state_dict(torch.load(ckpt_optimizer_file))
                time_ckpt = time.time()
                print("==============================================================================")
    
            if best_ckpt_score is None or current_score > best_ckpt_score:
                print("[CKPT] Saved new best score: %.3f at %s" % (current_score, "["+str(datetime.now())+"]"))
                best_ckpt_score = current_score
                torch.save(summarizer.model.state_dict(), ckpt_file)
                torch.save(optimizer.state_dict(), ckpt_optimizer_file)


end_time = time.time()
print("training done!")
print("checkpoint saved succesfully")
print("Total time used")
print(str(timedelta(seconds=end_time - start_training)))

wandb.alert(
    title=f"Coverage Phase -- {args.experiment} finished at {datetime.now()}", 
    text=f"Coverage Phase {args.experiment} finished at {datetime.now()}"
)
