from operator import truediv
import uncertain

import os
import typing
import json
from datasets import load_metric

from transformers import set_seed
from transformers import EarlyStoppingCallback, TrainingArguments, Trainer

import torch
from torch.nn.functional import softmax

from datetime import datetime
import argparse


def train(model_load:str="bert-base-uncased", model_config:str="bert-base-uncased", dataset:str="go_emotions", output_dir:str="",\
    batch_size:typing.Optional[int]=16, num_train_epochs:typing.Optional[int]=20, report: typing.Optional[typing.List[str]] = None):

    train_dataset, dev_dataset, test_dataset, tokenizer, dataset_args = uncertain.load_data(dataset, model_config)

    model = uncertain.models.automodels(model_load, output_dir, dataset_args)
    logging_steps = len(train_dataset) // batch_size
    args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy = "epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        metric_for_best_model = "eval_accuracy",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        save_total_limit = 2,
        load_best_model_at_end = True,
        weight_decay=0.01,
        logging_steps=logging_steps,
        report_to=report
    )

    trainer = Trainer(
        model,
        args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=uncertain.compute_metrics,
        callbacks = [EarlyStoppingCallback(early_stopping_patience=2)],
        tokenizer=tokenizer)

    trainer.train()
    trainer.model.save_pretrained(os.path.join(output_dir, 'checkpoint-best'))
    result = trainer.predict(test_dataset=test_dataset)
    result_tensor = torch.from_numpy(result.predictions)
    result_prob = softmax(result_tensor, dim=1).numpy()
    print(result.metrics)
    result_json_path = os.path.join(output_dir, 'result.json')
    with open(result_json_path, 'w') as f:
        f.write(json.dumps(result.metrics))

def main():
    parser = argparse.ArgumentParser(description="CUE train")
    parser.add_argument('-d', '--dataset', required=True, type=str, help="Dataset name. e.g., cola, emotion, go_emotions and multi_nli.")
    parser.add_argument('-b', '--batch_size', default=8, type=int, help="Batch size, default 8." )
    parser.add_argument('-e', '--epoch', default=30, type=int, help="Epoch, default 30." )
    parser.add_argument('-m', '--model', type=str, help="Model name" )

    parser.add_argument('-o', '--original', default=False, type=lambda x: (str(x).lower() == 'true'), help="Original finetune." )
    parser.add_argument('-p', '--path', required=True, type=str, help="Model path." )
    parser.add_argument('-r', '--round', required=True, default=5, type=int, help="Round to train" )
    parser.add_argument('-t', '--tokenizer', required=True, type=str, help="Tokenizer. e.g., bert-base-uncased." )

    args = parser.parse_args()
    start = 0
    times_to_train = args.round
    for idx in range(start, times_to_train):
        if args.original:
            model_load = args.model
        else:
            model_load = uncertain.get_finetuned_checkpoint(args.model)

        now = datetime.now()
        time = now.strftime("%m%d-%H%M")
        set_seed(idx)
        output_name = args.path + f'-{time}-{idx}'

        output_dir = uncertain.get_path(output_name)
        
        train(model_load=model_load, model_config=args.tokenizer, dataset=args.dataset, output_dir=output_dir, batch_size=args.batch_size, num_train_epochs=args.epoch, report="none")

if __name__ == "__main__":
    main()