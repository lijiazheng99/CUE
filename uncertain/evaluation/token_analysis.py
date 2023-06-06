import regex as re
import os
import pandas as pd
import torch 
from torch.nn import CosineSimilarity, PairwiseDistance
from torch.nn.functional import softmax
from spacy import load
import numpy as np

import uncertain
from uncertain import get_finetuned_checkpoint
from uncertain.dataloader import load_dataloader, prepare_input
from transformers import AutoModelForSequenceClassification

from tqdm import tqdm

cos = CosineSimilarity(dim=1, eps=1e-6)

def remove_single_token(input_ids):
    removed_inputs = []
    tokens = []
    for idx, token in enumerate(input_ids):
        if idx == 0 or idx == (len(input_ids) - 1):
            pass
        else:
            # print(type(input_ids))
            removed_inputs += [torch.tensor([input_ids[:idx]+input_ids[idx+1:]])]
            tokens += [token]
    return removed_inputs, tokens

pdist = PairwiseDistance(p=2)

def token_analysis(model_load:str="bert-base-uncased", model_config:str="bert-base-uncased", dataset:str="go_emotions"):
    train_loader, dev_loader, test_loader, tokenizer, dataset_args = load_dataloader(dataset, model_config)

    checkpoint = get_finetuned_checkpoint(model_load)
    # model =  AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=dataset_args.num_labels)
    output_path = uncertain.sort_checkpoints(model_load)
    output_path = f'{model_load}/{output_path[-1]}'
    print(output_path)
    model = uncertain.models.automodels(output_path, model_load, dataset_args)
    model.eval()
    model.cuda()

    en = load('en_core_web_sm')
    sws = list(en.Defaults.stop_words)
    sws.extend(['[SEP]','[UNK]','[CLS]'])
    sws.extend(['.',',','?','!','/','"','{','}','(',')','#','%',':','’','[',']'])

    instances = [36, 297, 848, 1057, 2256]
    # instances = [74, 564,1616]
    instances = [72, 948,2764]
    # instances = [992, 993, 994, 995, 996]
    t = tqdm(total=len(test_loader))
    impact_tokens = []
    new_labels = []
    pred_labels = []
    for step, batch in enumerate(test_loader):
        # if step in instances:
        if 'itchy trigger fingers' in [tokenizer.decode(each) for each in batch['input_ids']][0]:

            t.update(1)
            sentence = '[CLS] somehow i got banned for replying to a troll. the mods over there have itchy trigger fingers. [SEP]'
            sentence = 'somehow, i got banned for replying to a troll. the mods over there have gone completely rogue. '
            sentence = 'Somehow I got banned for replying to a troll. The mods over there are excessively eager to take action.'
            sentence = tokenizer.encode(sentence)
            # print(sentence)
            # print(batch['input_ids'])
            batch['input_ids'] = torch.tensor([sentence])
            print([tokenizer.decode(each) for each in batch['input_ids']])
            gpu_batch = prepare_input(batch)
            with torch.no_grad():
                output = model(gpu_batch['input_ids'], output_hidden_states = True)
            logits1 = output.logits
            entropy = torch.distributions.Categorical(softmax(logits1, dim=1)).entropy()
            print(entropy)
            distri = softmax(logits1, dim=1)
            pred_label = torch.argmax(distri, dim=1).item()
            # print(distri)
            for idx, prob in enumerate(distri[0]):
                print(dataset_args.id2label[str(idx)], prob)
            # print(dataset_args.id2label[str(pred_label)])
            probability = softmax(logits1, dim=1)[0][pred_label]
            tokenizer.decode(batch['input_ids'][0].tolist())
            print(probability)
            removed_inputs, tokens = remove_single_token(batch['input_ids'][0].tolist())
            impact_token = []
            new_label_ =[]
            p_dist = []
            for each, token in zip(removed_inputs, tokens):
                gpu_batch = prepare_input(each)
                with torch.no_grad():
                    logits = model(gpu_batch).logits
                p_dist.append(pdist(logits1, logits).item())
                new_pred_label = torch.argmax(softmax(logits, dim=1), dim=1).item()
                new_probability = softmax(logits, dim=1)[0][new_pred_label]
                if new_probability < probability:
                    print(tokenizer.decode(token), new_probability)
            print(np.mean(p_dist))
            #         impact_token += [tokenizer.decode(token)]
            #         new_label = torch.argmax(softmax(logits, dim=1), dim=1).item()
            #         new_label_ += [str(new_label)]
            # impact_tokens += [impact_token]
            # new_labels += [new_label_]
    sb
    t.close()
    
    df = pd.DataFrame()
    df['Texts'] = dataset_args.df_test["text"]
    df['Pred'] = pred_labels
    df['Recon'] = new_labels
    df['Original'] = dataset_args.df_test[dataset_args.label_col]
    df['Word'] = impact_tokens
    df.to_csv(os.patsh.join(f'{model_load}-removal_approach.csv'), sep=',', index= None)