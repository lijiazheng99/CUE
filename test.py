import uncertain
import os
import json
import pandas as pd
import argparse

def main():
    parser = argparse.ArgumentParser(description="CUE evaluation")
    parser.add_argument('-d', '--dataset', required=True, type=str, help="Dataset name. e.g., cola, emotion, go_emotions and multi_nli.")
    parser.add_argument('-b', '--batch_size', default=8, type=int, help="Batch size, default 8." )
    parser.add_argument('-p', '--path', required=True, type=str, help="Model path." )
    parser.add_argument('-t', '--tokenizer', required=True, type=str, help="Tokenizer. e.g., bert-base-uncased." )

    args = parser.parse_args()
    multiple_results = {}
    model_paths = uncertain.find_in_path(args.path)
    all_results = {}
    for model_path in model_paths:
        num_run = model_path.split('-')[-1]
        results = uncertain.evaluation.basic_eva(model_path, args.tokenizer, args.dataset, args.batch_size)
        all_results.update({num_run:results['checkpoint-best']})
    all_results = dict(sorted(all_results.items()))
    all_results = pd.DataFrame.from_dict(all_results).T
    multiple_results.update({args.path:{'results':all_results, 'average':all_results.mean(axis=0)}})

    for key in multiple_results.keys():
        print(key)
        print('results')
        print(multiple_results[key]['results'])
        print('average')
        print(multiple_results[key]['average'].T)

if __name__ == "__main__":
    main()