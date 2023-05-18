#test the bert ner model from huggingface
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import torch
# from helpers import load_conll
import pandas as pd
from datasets import load_dataset

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



def generate_word2idx(data, max_len, PAD='<PAD>'):
    # Create word to index mappings
    word2idx = {PAD:0} # Padding is index 0
    idx2word = [PAD]
    for sentPos, sent in enumerate(data):
        for wordPos, word in enumerate(sent['tokens'][:max_len]):
            if word not in word2idx:
                word2idx[word] = len(idx2word)
                idx2word.append(word)

    return word2idx, idx2word


def convert_to_word_indices(data, word2idx, max_len, PAD='<PAD>'):
    # Convert dataset to word indices
    feats = torch.zeros((len(data), max_len), dtype=torch.long)
    for sentPos, sent in enumerate(data):
        for wordPos, word in enumerate(sent['tokens'][:max_len]):
            wordIdx = word2idx[PAD] if word not in word2idx else word2idx[word]
            feats[sentPos][wordPos] = wordIdx

    return feats


def preprocess_data(dataset, word2idx, max_len, num_entities, PAD='<PAD>'):
    sentence_feats = convert_to_word_indices(dataset, word2idx, max_len, PAD)

    # Generate labels as a tensor of booleans indicating if the masked token is a named entity
    mask_labels = torch.tensor([sent['is_ner'] for sent in dataset], dtype=torch.float)
    # Add a dimension to labels
    mask_labels = mask_labels.unsqueeze(1)

    # Filter out all the sentences in dataset where the masked token is not a named entity, i.e. the is_ner field is False
    named_entity_data = [sent for sent in dataset if sent['is_ner']]
    

    named_entity_data_labels = torch.eye(num_entities-1)[[sent['ner_tag']-1 for sent in named_entity_data]]
    approach1_model_2_test_data = torch.eye(num_entities-1)[[sent['ner_tag']-1 for sent in dataset]] # named_entity_data_labels_full_length


    named_entity_sentence_feats = convert_to_word_indices(named_entity_data, word2idx, max_len)

    # Generate an array of size (num_feats, num_entities + 1) where each row is a one hot encoded vector of the named entity class or non named entity class
    approach2_labels = torch.zeros((len(dataset), num_entities), dtype=torch.float)
    # For approach 3 we need to pad true_ner_tags to the same length as labels. We just add an array of 0's when the label is a non named entity
    approach_3_task_2_labels = torch.zeros((len(dataset), num_entities-1), dtype=torch.float)
    for sentPos, sent in enumerate(dataset):
        label = sent['ner_tag']
        # If the masked token is not a named entity, then the label is the non named entity class
        if not label:
            approach2_labels[sentPos][0] = 1
        else:
            approach2_labels[sentPos][label] = 1
            approach_3_task_2_labels[sentPos][label-1] = 1


    return sentence_feats, mask_labels, named_entity_sentence_feats, named_entity_data_labels, approach1_model_2_test_data, approach2_labels, approach_3_task_2_labels