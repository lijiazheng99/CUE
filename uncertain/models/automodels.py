from transformers import AutoModelForSequenceClassification
from .distilbert import DistilBertBNNForSequenceAnalysis, DistilBertMCDForSequenceClassification, DistilBertVAEForSequenceAnalysis, DistilBertVAEForSequenceClassification, DistilBertBNNForSequenceClassification, DistilBertLabelSmoothingSequenceClassification, DistilBertVAEForSequenceClassificationMaskAnalysis
from .albert import AlbertVAEForSequenceClassification, AlbertVAEForSequenceAnalysis, AlbertBNNForSequenceClassification, AlbertBNNForSequenceAnalysis, AlbertMCDForSequenceClassification, AlbertLabelSmoothingSequenceClassification, AlbertVAEForSequenceClassificationMaskAnalysis
from .roberta import RobertaMCDForSequenceClassification, RobertaVAEForSequenceClassification, RobertaVAEForSequenceAnalysis, RobertaBNNForSequenceClassification, RobertaBNNForSequenceAnalysis, RobertaLabelSmoothingForSequenceClassification, RobertaVAEForSequenceClassificationMaskAnalysis
from .bert import BertVAEForSequenceClassification, BertVAEForSequenceClassificationAnalysis, BertBNNForSequenceClassification, BertBNNForSequenceClassificationAnalysis, BertMCDForSequenceClassification, BertLebelSmoothingSequenceClassification, BertVAEForSequenceClassificationMaskAnalysis, BertBNNForSequenceClassificationMaskAnalysis
from .utils.monte_carlo import activate_mc_dropout, convert_to_mc_dropout

def automodels(model_load:str="bert-base-uncased", output_path:str="", dataset_args:dict={}, loss_function:str="regression"):
    model = None
    if "distilbert" in output_path:
        if "labelsmoothing" in output_path:
            model = DistilBertLabelSmoothingSequenceClassification.from_pretrained(model_load, num_labels=dataset_args.num_labels)
        elif "bnn" in output_path:
            model = DistilBertBNNForSequenceClassification.from_pretrained(model_load, num_labels=dataset_args.num_labels)
        elif "cue" in output_path:
            model = DistilBertVAEForSequenceClassification.from_pretrained(model_load, num_labels=dataset_args.num_labels)
        elif "mc" in output_path:
            model = DistilBertMCDForSequenceClassification.from_pretrained(model_load, num_labels=dataset_args.num_labels)
        else:
            model = AutoModelForSequenceClassification.from_pretrained(model_load, num_labels=dataset_args.num_labels)
    elif "roberta" in output_path:
        if "labelsmoothing" in output_path:
            model = RobertaLabelSmoothingForSequenceClassification.from_pretrained(model_load, num_labels=dataset_args.num_labels)
        elif "bnn" in output_path:
            model = RobertaBNNForSequenceClassification.from_pretrained(model_load, num_labels=dataset_args.num_labels)
        elif "cue" in output_path:
            model = RobertaVAEForSequenceClassification.from_pretrained(model_load, num_labels=dataset_args.num_labels)
        elif "mc" in output_path:
            model = RobertaMCDForSequenceClassification.from_pretrained(model_load, num_labels=dataset_args.num_labels)
        else:
            model = AutoModelForSequenceClassification.from_pretrained(model_load, num_labels=dataset_args.num_labels)
    elif "albert" in output_path:
        if "labelsmoothing" in output_path:
            model = AlbertLabelSmoothingSequenceClassification.from_pretrained(model_load, num_labels=dataset_args.num_labels)
        elif "bnn" in output_path:
            model = AlbertBNNForSequenceClassification.from_pretrained(model_load, num_labels=dataset_args.num_labels)
        elif "cue" in output_path:
            model = AlbertVAEForSequenceClassification.from_pretrained(model_load, num_labels=dataset_args.num_labels)
        elif "mc" in output_path:
            model = AlbertMCDForSequenceClassification.from_pretrained(model_load, num_labels=dataset_args.num_labels)
        else:
            model = AutoModelForSequenceClassification.from_pretrained(model_load, num_labels=dataset_args.num_labels)
    elif "bert" in output_path:
        if "labelsmoothing" in output_path:
            model = BertLebelSmoothingSequenceClassification.from_pretrained(model_load, num_labels=dataset_args.num_labels)
        elif "bnn" in output_path:
            model = BertBNNForSequenceClassification.from_pretrained(model_load, num_labels=dataset_args.num_labels)
        elif "cue" in output_path:
            model = BertVAEForSequenceClassification.from_pretrained(model_load, num_labels=dataset_args.num_labels)
        elif "mc" in output_path:
            model = BertMCDForSequenceClassification.from_pretrained(model_load, num_labels=dataset_args.num_labels)
        else:
            model = AutoModelForSequenceClassification.from_pretrained(model_load, num_labels=dataset_args.num_labels)
    else:
        print("Undefined model type")

    if "bnn" in output_path or "cue" in output_path:
        print("***** Trainable *****\n")

        if "distilbert"in output_path:
            model.distilbert.trainable = False
            model.pre_classifier.trainable = False
            model.classifier.trainable = False
            print(f"DistilBert model {model.distilbert.trainable} Pre Classifier {model.pre_classifier.trainable} Classifier {model.classifier.trainable}")
            for param in model.distilbert.parameters():
                param.requires_grad = False
            for param in model.pre_classifier.parameters():
                param.requires_grad = False
            for param in model.classifier.parameters():
                param.requires_grad = False
        
        elif "roberta" in output_path:
            model.roberta.trainable = False
            model.classifier.trainable = False
            print(f"Roberta model {model.roberta.trainable} Classifier {model.classifier.trainable}")
            for param in model.roberta.parameters():
                param.requires_grad = False
            for param in model.classifier.parameters():
                param.requires_grad = False  

        elif "albert" in output_path:
            model.albert.trainable = False
            model.classifier.trainable = False
            print(f"Alberta model {model.albert.trainable} Classifier {model.classifier.trainable}")
            for param in model.albert.parameters():
                param.requires_grad = False
            for param in model.classifier.parameters():
                param.requires_grad = False  

        elif "bert" in output_path:
            model.bert.trainable = False
            model.classifier.trainable = False
            print(f"Bert model {model.bert.trainable} Classifier {model.classifier.trainable}")
            for param in model.bert.parameters():
                param.requires_grad = False
            for param in model.classifier.parameters():
                param.requires_grad = False
    elif "mc" in output_path:
        activate_mc_dropout(model, activate=True, random=0.1, verbose=True)

    model.config.id2label = dataset_args.id2label
    model.config.label2id = dataset_args.label2id

    return model


def analysis_automodels(model_load:str="bert-base-uncased", output_path:str="", dataset_args:dict={}, loss_function:str="regression", mask_strategy:list=[]):
    model = None
    if "distilbert" in output_path:
        if "bnn" in output_path:
            model = DistilBertBNNForSequenceAnalysis.from_pretrained(model_load, num_labels=dataset_args.num_labels)
        elif "cue" in output_path:
            if len(mask_strategy) > 0:
                model = DistilBertVAEForSequenceClassificationMaskAnalysis.from_pretrained(model_load, num_labels=dataset_args.num_labels, mask_strategy = mask_strategy)
            else:    
                model = DistilBertVAEForSequenceAnalysis.from_pretrained(model_load, num_labels=dataset_args.num_labels)
        else:
            print("Undefined model type")
    elif "roberta" in output_path:
        if "bnn" in output_path:
            model = RobertaBNNForSequenceAnalysis.from_pretrained(model_load, num_labels=dataset_args.num_labels)
        elif "cue" in output_path:
            if len(mask_strategy) > 0:
                model = RobertaVAEForSequenceClassificationMaskAnalysis.from_pretrained(model_load, num_labels=dataset_args.num_labels, mask_strategy = mask_strategy)
            else:
                model = RobertaVAEForSequenceAnalysis.from_pretrained(model_load, num_labels=dataset_args.num_labels)
        else:
            print("Undefined model type")
    elif "albert" in output_path:
        if "bnn" in output_path:
            model = AlbertBNNForSequenceAnalysis.from_pretrained(model_load, num_labels=dataset_args.num_labels)
        elif "cue" in output_path:
            if len(mask_strategy) > 0:
                model = AlbertVAEForSequenceClassificationMaskAnalysis.from_pretrained(model_load, num_labels=dataset_args.num_labels, mask_strategy = mask_strategy)
            else:
                model = AlbertVAEForSequenceAnalysis.from_pretrained(model_load, num_labels=dataset_args.num_labels)
        else:
            print("Undefined model type")
    elif "bert" in output_path:
        if "bnn" in output_path:
            if len(mask_strategy) > 0:
                model = BertBNNForSequenceClassificationMaskAnalysis.from_pretrained(model_load, num_labels=dataset_args.num_labels, mask_strategy = mask_strategy)
            else:
                model = BertBNNForSequenceClassificationAnalysis.from_pretrained(model_load, problem_type=loss_function, num_labels=dataset_args.num_labels)
        elif "cue" in output_path:
            if len(mask_strategy) > 0:
                model = BertVAEForSequenceClassificationMaskAnalysis.from_pretrained(model_load, num_labels=dataset_args.num_labels, mask_strategy = mask_strategy)
            else:
                model = BertVAEForSequenceClassificationAnalysis.from_pretrained(model_load, problem_type=loss_function, num_labels=dataset_args.num_labels)
        else:
            print("Undefined model type")
    else:
        print("Undefined model type")

    model.config.id2label = dataset_args.id2label
    model.config.label2id = dataset_args.label2id

    return model
