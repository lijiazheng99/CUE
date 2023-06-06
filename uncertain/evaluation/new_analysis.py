import os
import numpy as np
import pandas as pd

from transformers import TrainingArguments, Trainer

from transformers.trainer_callback import PrinterCallback
from sklearn.metrics import mutual_info_score

import json

import torch
from torch import nn
from torch.nn.functional import softmax
from scipy.stats import entropy

# from scipy.special import softmax

import uncertain
from uncertain import get_finetuned_checkpoint
from uncertain.constants import accuracy, f1_score
from uncertain.dataloader import load_data, load_dataloader, prepare_input
from .singletest import calibration_error, calculate_entropy, label_one_hot
from spacy import load
from tqdm import tqdm

en = load('en_core_web_sm')
sws = list(en.Defaults.stop_words)
sws.extend(['.',',','?','!','/','"','{','}','(',')','#','%',':','’','[',']'])

# def get_label(logits):
#     return np.argmax(softmax(logits, axis=0), axis=1)

def get_influential_token(tokens, coss):
    valid_tokens = [0 if each in sws else 1 for each in tokens]
    valid_coss = [cos if valid == 1 and cos > 0 else 2 for valid, cos in zip(valid_tokens, coss)]
    pos = np.argmin(valid_coss)
    if valid_coss[pos] != 2:
        return tokens[pos], valid_coss[pos]
    else:
        valid_coss = [cos if valid == 1 and cos < 0 else -2 for valid, cos in zip(valid_tokens, coss) ]
        pos = np.argmax(valid_coss)
        return tokens[pos], valid_coss[pos]

def process_inputcoss(input_ids, word_coss, tokenizer, dataset):
    allow_second_sep = False
    cls, sep = tokenizer.encode('')
        # end = tokenizer.encode(str(tokenizer.sep_token))[1]
    
    if dataset == "multi_nli":
        allow_second_sep = True
    
    sentences, words, coss = [], [], []
    print("***** Finding Influential Words *****")
    t = tqdm(total=len(input_ids))
    for input_id, word_coss in zip(input_ids, word_coss):
        input_id = input_id.tolist()
        if allow_second_sep:
            sep_pos = input_id.index(sep,input_id.index(sep))
            start = input_id.index(cls)+1
            end = sep_pos
            input_id = input_id[start:end]
            word_coss = word_coss[start:end]
        else:
            start = input_id.index(cls)+1
            end = input_id.index(sep)
            input_id = input_id[start:end]
            word_coss = word_coss[start:end]
        tokens = [tokenizer.decode(each) for each in input_id]
        word, cos = get_influential_token(tokens, word_coss)
        sentences += [" ".join(tokens)]
        words += [word]
        coss += [cos]
        t.update(1)
    t.close()
    return sentences, words, coss

def latent_importance_analysis(model_load:str="bert-base-uncased", model_config:str="bert-base-uncased", dataset:str="go_emotions"):
    # _,test_dataset, _, tokenizer, dataset_args = uncertain.load_data(dataset, model_config)
    _, _, test_dataset, tokenizer, dataset_args = uncertain.load_data(dataset, model_config)
    checkpoint_path = uncertain.get_finetuned_checkpoint(model_load)

    evaluation={}
    mask_strategy = [[0,10],[10,20],[20,30],[30,40],[40,50],[50,60],[60,70],[70,80], [80,90], [90,100]]
    for each_mask in mask_strategy:
        model = uncertain.models.analysis_automodels(checkpoint_path, model_load, dataset_args, mask_strategy=each_mask)
        batch_size = 1
        args = TrainingArguments(
            output_dir=model_load,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            report_to=None
        )
        trainer = Trainer(model, args, tokenizer=tokenizer, compute_metrics=uncertain.compute_metrics)
        trainer.remove_callback(PrinterCallback)
        result = trainer.predict(test_dataset=test_dataset)

        result_tensor = torch.from_numpy(np.asarray(result.predictions))
        result_prob = softmax(result_tensor, dim=1).numpy()
        entro = calculate_entropy(result_prob, dataset_args.num_labels)

        y_true = label_one_hot(result.label_ids, dataset_args.num_labels)
        expected_calibration_error = calibration_error(y_true, result_prob, n_bins=9)
        new_evaluation = {'acc':result.metrics["test_accuracy"], 'f1':result.metrics["test_f1"],\
            'mean_entro':entro['entro_mean'], 'ece': expected_calibration_error[2]}
        mask = "-".join([str(each_mask[0]),str(each_mask[1])])
        evaluation.update({mask:new_evaluation})
    all_results = pd.DataFrame.from_dict(evaluation).T

    return all_results

def analysis(model_load:str="bert-base-uncased", model_config:str="bert-base-uncased", dataset:str="go_emotions"):
    # _, _, test_dataset, tokenizer, dataset_args = uncertain.load_data(dataset, model_config)
    test_dataset, _, _, tokenizer, dataset_args = uncertain.load_data(dataset, model_config)
    checkpoint_path = uncertain.get_finetuned_checkpoint(model_load)

    model = uncertain.models.analysis_automodels(checkpoint_path, model_load, dataset_args)
    batch_size = 16
    args = TrainingArguments(
        output_dir=model_load,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        report_to=None
    )
    trainer = Trainer(model, args, tokenizer=tokenizer)
    result = trainer.predict(test_dataset=test_dataset)
    print('results',result.predictions)
    # result_tensor = torch.from_numpy(np.asarray(result.predictions))
    # result_prob = softmax(result_tensor, dim=1).numpy().to_list()
    logits, recon_logits, influntial_dims, influntial_dim_coss, input_ids, word_coss = result.predictions
    labels = result.label_ids
    # original_labels = get_label(logits)
    # recon_labels = get_label(recon_logits)
    texts, words, coss = process_inputcoss(input_ids, word_coss, tokenizer, dataset)

    df = pd.DataFrame()
    df['Texts'] = texts
    df['Pred'] = logits
    df['Recon'] = recon_logits
    df['Original'] = labels
    df['Dim'] = influntial_dims
    df['Dim_cos'] = influntial_dim_coss
    df['Word'] = words
    df['Word_cos'] = coss
    df.to_csv(f'{model_load}.csv', sep=',', index= None)

    print("***** Evaluation *****")
    ori_acc = accuracy.compute(predictions=logits, references=labels)
    ori_f1 = f1_score.compute(predictions=logits, references=labels, average='macro')
    print(f'Original acc: {ori_acc} f1: {ori_f1}')
    recon_acc = accuracy.compute(predictions=recon_logits, references=labels)
    recon_f1 = f1_score.compute(predictions=recon_logits, references=labels, average='macro')
    print(f'Recon acc: {recon_acc} f1: {recon_f1}')
    return df