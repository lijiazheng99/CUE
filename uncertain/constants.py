import os
import subprocess as sp
import numpy as np
import regex as re

DATAFOLDER = ""
CACHEFOLDER = ""
os.environ["TRANSFORMERS_CACHE"] = CACHEFOLDER
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

from datasets import load_metric

def __gpu_auto_select(num_gpu:int=1):
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    try:
        memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    except:
        print("Error: Please check the avalibility of 'nvidia-smi' command")
    else:
        memory_free_values = {str(i):int(x.split()[0]) for i, x in enumerate(memory_free_info)}
        if num_gpu > len(memory_free_values):
            print(f"Error: Number of desired gpus: {num_gpu} larger than total avaliable gpus {len(memory_free_values)}.")
            raise Exception
        ranked_gpus = list(dict(sorted(memory_free_values.items(), key=lambda item: item[1], reverse=True)).keys())[:num_gpu]
        if len(ranked_gpus) > 1:
            result = ','.join(ranked_gpus)
        else:
            result = ranked_gpus[0]
        return result, 'cuda'

def get_path(foldername:str):
    return os.path.join(DATAFOLDER,foldername)


def find_in_path(foldername:str):
    return_folders = []
    folders = os.listdir(DATAFOLDER)
    pattern = r"-\d{4}-\d{4}-\d{1}"
    if os.path.exists(os.path.join(DATAFOLDER,foldername)):
        return [os.path.join(DATAFOLDER,foldername)]
    else:
        pattern = f"{foldername}" + r"-\d{4}-\d{4}-\d{1}"
        for each in folders:
            if re.match(pattern, each):
                return_folders.append(os.path.join(DATAFOLDER,each))
    print('Folders to test:', return_folders)
    return return_folders

def __checkpoint_num(foldername:str):
    return int(foldername.split('-')[1])

def sort_checkpoints(foldername:str):
    num_checkpoints = []
    append_checkpoints = []
    for folder in os.listdir(foldername):
        if 'best' in folder:
            append_checkpoints += [folder]
        elif 'checkpoint' in folder:
            num_checkpoints += [folder]
    num_checkpoints = sorted([folder for folder in num_checkpoints if 'checkpoint' in folder], key=__checkpoint_num)
    num_checkpoints.extend(append_checkpoints)
    return num_checkpoints

def get_finetuned_checkpoint(foldername:str):
    folder_path = get_path(foldername)
    return os.path.join(folder_path, sort_checkpoints(folder_path)[-1])

from datetime import datetime
now = datetime.now()
time = now.strftime("%m%d-%H%M")
accuracy = load_metric("accuracy", experiment_id = time)
f1_score = load_metric("f1", experiment_id = time)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    output = {}
    output.update(accuracy.compute(predictions=predictions, references=labels))
    if type(labels)==type(np.ndarray(1)) or len(set(labels))>2:
        output.update(f1_score.compute(predictions=predictions, references=labels, average='macro'))
    else:
        output.update(f1_score.compute(predictions=predictions, references=labels, average='binary'))
    return output

GPU_NUM, DEVICE = __gpu_auto_select(1)
print(f"***** Selected CUDA device: {GPU_NUM} *****")
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_NUM
