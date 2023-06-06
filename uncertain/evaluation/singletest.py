import os
import numpy as np
import pandas as pd

from transformers import TrainingArguments, Trainer
from sklearn.metrics import mutual_info_score

import json

import torch
from torch.nn.functional import softmax
from scipy.stats import entropy

import uncertain
from uncertain.constants import DATAFOLDER
from uncertain.dataloader import load_data

from transformers.trainer_callback import PrinterCallback

from torch.nn import CrossEntropyLoss
loss_fct = CrossEntropyLoss()

def calculate_entropy(emo, base):
    all_entro = []
    for each in emo:
        all_entro += [entropy(each,base=base)]
    return {'entro_mean': np.mean(all_entro), 'entro_std': np.std(all_entro)}

def cross_entro(pred, true):
    return loss_fct(pred, true)

def label_one_hot(labels, num_labels):
    new_labels = []
    for each in labels:
        new_labels.append(np.asarray([ 1 if i == each else 0 for i in range(0, num_labels)]))
    return np.asarray(new_labels)

def calibration_error(y_true, y_prob, n_bins=5, strategy='uniform', return_expected_caliberation_error=True):
    # https://medium.com/@wolframalphav1.0/evaluate-the-performance-of-a-model-in-high-risk-applications-using-expected-calibration-error-and-dbc392c68318
    if strategy == 'quantile':  # Determine bin edges by distribution of data
        quantiles = np.linspace(0, 1, n_bins + 1)
        bins = np.percentile(y_prob, quantiles * 100)
        bins[-1] = bins[-1] + 1e-8
    elif strategy == 'uniform':
        # bins = np.linspace(0.0, 1.0, n_bins + 1)
        bins = np.array([i*1.0/(n_bins+1) for i in range(0, n_bins+1)])
    else:
        raise ValueError("Invalid entry to 'strategy' input. Strategy "
                            "must be either 'quantile' or 'uniform'.")

    y_prob_max = np.max(y_prob, axis=-1)
    binids = np.digitize(y_prob_max, bins) - 1

    y_correct_classified = (np.argmax(y_true, axis=-1) == np.argmax(y_prob, axis=-1)).astype(int) 

    bin_sums = np.bincount(binids, weights=y_prob_max, minlength=len(bins))
    bin_true = np.bincount(binids, weights=y_correct_classified, minlength=len(bins))
    bin_total = np.bincount(binids, minlength=len(bins))

    nonzero = bin_total != 0

    # acc(Bm)
    prob_true = bin_true[nonzero] / bin_total[nonzero]

    #conf(Bm)
    prob_pred = bin_sums[nonzero] / bin_total[nonzero]

    expected_caliberation_error = np.sum(bin_total[nonzero] * np.abs(prob_true - prob_pred))/bin_total[nonzero].sum()
    overconfidence_error = np.sum(bin_total[nonzero] *  prob_pred * np.max(np.concatenate(((prob_pred - prob_true).reshape(-1, 1), np.zeros((1, len(prob_pred))).T), axis=1), axis=-1)/bin_total[nonzero].sum())
    return np.mean(prob_true), np.mean(prob_pred), expected_caliberation_error, overconfidence_error

def analyze_correct_label_distribution(pred, correct):
    correct = [each[0] for each in correct]
    labels = set(correct)
    labels = sorted(labels)
    
    stats = {}
    counts = {}
    for each in labels:
        stats[each] = 0
        counts[each] = 0

    for pre, cor in zip(pred, correct):
        counts[cor] += 1
        if pre == cor:
            stats[cor] += 1
    
    for each in labels:
        stats[each] = stats[each]/counts[each]
    
    print(stats)
    
def basic_eva(model_load:str="bert-base-uncased", model_config:str="bert-base-uncased", dataset:str="go_emotions", batchsize:int=8):
    _, _, test_dataset, tokenizer, dataset_args = uncertain.load_data(dataset, model_config)
    # _, test_dataset, _, tokenizer, dataset_args = uncertain.load_data(dataset, model_config)
    eva_json_path = os.path.join(model_load, 'ckp_eval.json')

    checkpoints = uncertain.sort_checkpoints(model_load)
    checkpoints = checkpoints[-1:]
    print(model_load)
    print(checkpoints)
    
    evaluation = {}
    calibration = {}
    test_loss = []
    for checkpoint in checkpoints:
        checkpoint_path = os.path.join(model_load,checkpoint)

        model = uncertain.models.automodels(checkpoint_path, model_load, dataset_args)

        batch_size = batchsize

        args = TrainingArguments(
            output_dir=model_load,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
        )

        trainer = Trainer(model, args, tokenizer=tokenizer, compute_metrics=uncertain.compute_metrics)
        trainer.remove_callback(PrinterCallback)
        result = trainer.predict(test_dataset=test_dataset)
        print(result.metrics['test_loss'])
        test_loss += [result.metrics['test_loss']]
        # print(evaluation)
        result_tensor = torch.from_numpy(result.predictions)
        result_prob = softmax(result_tensor, dim=1).numpy()
        result_ids = torch.argmax(softmax(result_tensor, dim=1), dim = 1).numpy()
        
        df = pd.DataFrame()
        print(result_prob[0])
        df['Probs'] = [f"{each[0]};{each[1]}" for each in result_prob]
        df['entro'] = list(torch.distributions.Categorical(softmax(result_tensor, dim=1)).entropy())
        df.to_csv(os.path.join(f'{model_load}.csv'), sep=',', index= None)

        # analyze_correct_label_distribution(result_ids, result.label_ids)
        entro = calculate_entropy(result_prob, dataset_args.num_labels)
        y_true = label_one_hot(result.label_ids, dataset_args.num_labels)
        expected_calibration_error = calibration_error(y_true, result_prob, n_bins=9)
        calibration_evaluation = {'prob_true':expected_calibration_error[0], 'prob_pred': expected_calibration_error[1],\
            'ece': expected_calibration_error[2], 'oce': expected_calibration_error[3]}
        checkpoint_evaluation = {'acc':result.metrics["test_accuracy"], 'f1':result.metrics["test_f1"],\
            'mean_entro':entro['entro_mean'], 'std_entro':entro['entro_std']}
        new_evaluation = {'acc':result.metrics["test_accuracy"], 'f1':result.metrics["test_f1"],\
            'mean_entro':entro['entro_mean'], 'ece': expected_calibration_error[2]}
        evaluation.update({checkpoint:new_evaluation})
        # calibration.update({checkpoint:calibration_evaluation})
    # min_loss = min(test_loss)
    # index = test_loss.index(min_loss)
    # print(checkpoints[index])
    # with open(eva_json_path, 'w') as f:
    #         f.write(json.dumps(evaluation))
    # test_df = pd.DataFrame.from_dict(evaluation, orient='index')
    # test_df.to_csv(eva_json_path.replace("json","csv"), sep=",")
    # with open(eva_json_path.replace("eval","calibaration"), 'w') as f:
    #         f.write(json.dumps(calibration))
    # test_df = pd.DataFrame.from_dict(calibration, orient='index')
    # test_df.to_csv(eva_json_path.replace("eval","calibaration").replace("json","csv"), sep=",")
    return evaluation
