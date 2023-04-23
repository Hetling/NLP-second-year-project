#test the bert ner model from huggingface
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
# from helpers import load_conll
import pandas as pd
from datasets import load_dataset

#preprocess dataz

# wnut = load_dataset("wnut_17")


#preprocess data
def generate_masked_sentences(data):
    #creates new dataset where each token in a sentence is masked and used as a new sentence
    #the ner tag of the masked token is used as the label

    new_data = []
    id_ = 0
    for sentence in data:
        for i in range(len(sentence['tokens'])):
            new_sentence = sentence['tokens'].copy()
            new_sentence[i] = '[MASK]'
            new_data.append({'id': id_,
                             'tokens': new_sentence,
                             'is_ner': bool(sentence['ner_tags'][i]),
                             'ner_tag': sentence['ner_tags'][i],})
            id_ += 1

    return new_data
