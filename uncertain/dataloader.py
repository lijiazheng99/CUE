import torch
import pandas as pd
import numpy as np
from argparse import Namespace
from datasets import load_dataset, Dataset
from collections.abc import Mapping
from torch.utils.data import DataLoader
from typing import Any, Union
from transformers import AutoTokenizer, DataCollatorWithPadding, default_data_collator

import pickle
from uncertain.constants import DEVICE
mm = False

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def num_to_list(num, min_num, max_num):
    new_label = []
    for each in range(min_num, max_num):
        if each == num:
            new_label += [1]
        else:
            new_label += [0]
    return new_label

def __remove_dup_label(data_df:pd.DataFrame):
    to_be_drop = [idx for idx, each in enumerate(data_df['labels']) if len(each) > 1]
    # print(f'To be drop: {len(to_be_drop)} Total: {len(data_df)} Rate: {len(to_be_drop)/len(dataframe)*100:.2f}%')
    return data_df.drop(to_be_drop)

def keep_dup_label(data_df:pd.DataFrame):
    to_be_drop = [idx for idx, each in enumerate(data_df['labels']) if len(each) == 1]
    return data_df.drop(to_be_drop)


def load_data(dataset_name:str="go_emotions", model_config:str="bert-base-uncased"):

    if dataset_name == "go_emotions":
        go_emotions = load_dataset("go_emotions", "simplified") # "raw", "simplified"

        df_train = go_emotions['train'].to_pandas()
        df_dev = go_emotions['validation'].to_pandas()
        df_test = go_emotions['test'].to_pandas()

        df_train = __remove_dup_label(df_train)
        df_dev = __remove_dup_label(df_dev)
        df_test = __remove_dup_label(df_test)

        label_list=['admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral']
        num_labels = len(label_list)
        label_col = "labels"
    
    elif dataset_name == "emotion":
        emotion = load_dataset("emotion")

        df_train = emotion['train'].to_pandas()
        df_dev = emotion['validation'].to_pandas()
        df_test = emotion['test'].to_pandas()

        label_list = emotion["train"].unique("label")
        print(f'Label list:{label_list}')
        label_list=["sadness", "joy", "love", "anger", "fear", "surprise"]
        num_labels = len(label_list)
        label_col = "label"
    
    elif dataset_name == "cola":
        cola = load_dataset("glue",'cola')

        df_train = cola['train'].to_pandas()
        df_dev = cola['validation'].to_pandas()
        df_test = cola['test'].to_pandas()

        df_train['text'] = df_train['sentence']
        df_dev['text'] = df_dev['sentence']
        df_test = df_dev

        label_list = cola["train"].unique("label")
        print(f'Label list:{label_list}')
        num_labels = len(label_list)
        label_col = "label"

    elif dataset_name == "multi_nli":
        mnli = load_dataset("glue",'mnli')

        df_train = mnli['train'].to_pandas()
        if mm:
            df_dev = mnli['validation_mismatched'].to_pandas()
            df_test = mnli['test_mismatched'].to_pandas()
        else:
            df_dev = mnli['validation_matched'].to_pandas()
            df_test = mnli['test_matched'].to_pandas()
        df_test = df_dev
        df_test["text"] = [" ".join([prem, hypo]) for prem, hypo in zip(df_test["premise"].values.tolist(), df_test["hypothesis"].values.tolist())]
        label_list = mnli["train"].unique("label")
        print(f'Label list:{label_list}')
        label_list=['entailment','neutral','contradiction']
        num_labels = len(label_list)
        label_col = "label"

    else:
        raise ValueError(f'Speicified {dataset_name} not used in expriments.')
    
    id2label = {str(i):label for i, label in enumerate(label_list)}
    label2id = {label:str(i) for i, label in enumerate(label_list)}

    dataset_args = Namespace(num_labels=num_labels,labels=label_list, label_col=label_col, id2label=id2label,label2id=label2id, df_test=df_test)

    tokenizer = AutoTokenizer.from_pretrained(model_config)
    print((df_train.shape, df_dev.shape, df_test.shape))

    if "nli" in dataset_name:
        train_encodings = tokenizer(df_train["premise"].values.tolist(), df_train["hypothesis"].values.tolist(), truncation=True)
        dev_encodings = tokenizer(df_dev["premise"].values.tolist(), df_dev["hypothesis"].values.tolist(), truncation=True)
        test_encodings = tokenizer(df_test["premise"].values.tolist(), df_test["hypothesis"].values.tolist(), truncation=True)
    else:
        train_encodings = tokenizer(df_train["text"].values.tolist(), truncation=True)
        dev_encodings = tokenizer(df_dev["text"].values.tolist(), truncation=True)
        test_encodings = tokenizer(df_test["text"].values.tolist(), truncation=True)
                
    train_labels = df_train[label_col].values.tolist()
    dev_labels = df_dev[label_col].values.tolist()
    test_labels = df_test[label_col].values.tolist()   
    
    train_dataset = CustomDataset(train_encodings, train_labels)
    dev_dataset = CustomDataset(dev_encodings, dev_labels)
    test_dataset = CustomDataset(test_encodings, test_labels)
    return train_dataset, dev_dataset, test_dataset, tokenizer, dataset_args

def load_dataloader(dataset_name:str="go_emotions", model_config:str="bert-base-uncased", batch_size:int=1):
    train_dataset, dev_dataset, test_dataset, tokenizer, dataset_args = load_data(dataset_name, model_config)
    # data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=(8))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=default_data_collator)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, collate_fn=default_data_collator)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=default_data_collator)
    return train_loader, dev_loader, test_loader, tokenizer, dataset_args

device = DEVICE

def prepare_input(data: Union[torch.Tensor, Any]) -> Union[torch.Tensor, Any]:
    """
    Prepares one `data` before feeding it to the model, be it a tensor or a nested list/dictionary of tensors.
    """
    if isinstance(data, Mapping):
        return type(data)({k: prepare_input(v) for k, v in data.items()})
    elif isinstance(data, (tuple, list)):
        return type(data)(prepare_input(v) for v in data)
    elif isinstance(data, torch.Tensor):
        kwargs = dict(device=device)
        # if self.deepspeed and data.dtype != torch.int64:
        #     # NLP models inputs are int64 and those get adjusted to the right dtype of the
        #     # embedding. Other models such as wav2vec2's inputs are already float and thus
        #     # may need special handling to match the dtypes of the model
        #     kwargs.update(dict(dtype=self.args.hf_deepspeed_config.dtype()))
        return data.cuda()
    return data

def prepare_output(data: Union[torch.Tensor, Any]) -> Union[torch.Tensor, Any]:
    """
    Prepares one `data` before feeding it to the model, be it a tensor or a nested list/dictionary of tensors.
    """
    if isinstance(data, Mapping):
        return type(data)({k: prepare_output(v) for k, v in data.items()})
    elif isinstance(data, (tuple, list)):
        return type(data)(prepare_output(v) for v in data)
    elif isinstance(data, torch.Tensor):
        kwargs = dict(device=device)
        # if self.deepspeed and data.dtype != torch.int64:
        #     # NLP models inputs are int64 and those get adjusted to the right dtype of the
        #     # embedding. Other models such as wav2vec2's inputs are already float and thus
        #     # may need special handling to match the dtypes of the model
        #     kwargs.update(dict(dtype=self.args.hf_deepspeed_config.dtype()))
        return data.cpu()
    return data

def covert_results(result_prob, id2label):
    positive= set(["amusement", "excitement", "joy", "love", "desire", "optimism", "caring", "pride", "admiration", "gratitude", "relief", "approval"])
    negative= set(["fear", "nervousness", "remorse", "embarrassment", "disappointment", "sadness", "grief", "disgust", "anger", "annoyance", "disapproval"])
    ambiguous= set(["realization", "surprise", "curiosity", "confusion"])
    return np.array([sum([each for idx, each in enumerate(result_prob) if id2label[str(idx)] in positive]),
    sum([each for idx, each in enumerate(result_prob) if id2label[str(idx)] in negative]),
    sum([each for idx, each in enumerate(result_prob) if id2label[str(idx)] in ambiguous]),
    ])

def goemo_three_labels(result_prob, id2label):
    new_result_prob = []
    for each_result in result_prob:
        new_result_prob.append(covert_results(each_result, id2label))
    return np.array(new_result_prob)
