#%%
import uncertain
import os
import json
import argparse
#%%

def main():
    parser = argparse.ArgumentParser(description="CUE analysis")
    parser.add_argument('-d', '--dataset', required=True, type=str, help="Dataset name. e.g., cola, emotion, go_emotions and multi_nli.")
    parser.add_argument('-m', '--mode', required=True, type=str, help="Analysis mode, dimension or token." )
    parser.add_argument('-p', '--path', required=True, type=str, help="Model path." )
    parser.add_argument('-t', '--tokenizer', required=True, type=str, help="Tokenizer. e.g., bert-base-uncased." )

    args = parser.parse_args()

    # for each in test_models:
    if args.mode == "dimension":
        model_paths = uncertain.find_in_path(args.path)
        for model_path in model_paths:
            results = uncertain.evaluation.latent_importance_analysis(model_path, args.tokenizer, args.dataset)
            uncertain.evaluation.visual_deltae(results, args.path)
    elif args.mode == "token":
        model_paths = uncertain.find_in_path(args.path)
        for model_path in model_paths:
            df = uncertain.evaluation.analysis(model_path, args.tokenizer, args.dataset)
            df.to_csv(f'{args.path}.csv', sep=',', index= None)
            
if __name__ == "__main__":
    main()

