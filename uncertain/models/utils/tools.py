import pandas as pd
from csv import writer, reader
import os

def write_representations(ori_rep, recon_rep, labels, labels_recon):
    ori_rep = ori_rep.tolist()
    recon_rep = recon_rep.tolist()
    labels = labels.tolist()

    identifier = "bert_cola_"

    # with open(identifier+'rep.csv', 'a', newline='\n') as f_object:
    #     writer_object = writer(f_object, delimiter='\t')
    #     writer_object.writerows(ori_rep)  
    #     f_object.close()
    
    # with open(identifier+'recon_rep.csv', 'a', newline='\n') as f_object:
    #     writer_object = writer(f_object, delimiter='\t')
    #     writer_object.writerows(recon_rep)  
    #     f_object.close()
    
    # if type(labels[0])==type([]):
    #     labels = [each[0] for each in labels]
    
    # with open(identifier+'labels.csv', 'a', newline='\n') as f_object:
    #     writer_object = writer(f_object)
    #     writer_object.writerows([[each] for each in labels])  
    #     f_object.close()
    
    if type(labels_recon[0])==type([]):
        labels_recon = [each[0] for each in labels_recon]
    
    with open(identifier+'recon_labels.csv', 'a', newline='\n') as f_object:
        writer_object = writer(f_object)
        writer_object.writerows([[each.to('cpu')] for each in labels_recon])  
        f_object.close()

def custom_labels(labels, to_add):
    return [str(each)+"_"+to_add for each in labels]

def write_latent(latent, labels):
    latent = latent.tolist()
    labels = labels.tolist()

    identifier = "bert_cola_"

    with open(identifier+'latent.csv', 'a', newline='\n') as f_object:
        writer_object = writer(f_object, delimiter='\t')
        writer_object.writerows(latent)
        f_object.close()
    
    if type(labels[0])==type([]):
        labels = [each[0] for each in labels]
    
    labels = custom_labels(labels,"vae")
    
    with open(identifier+'labels.csv', 'a', newline='\n') as f_object:
        writer_object = writer(f_object)
        writer_object.writerows([[each] for each in labels])  
        f_object.close()