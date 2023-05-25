import torch
from transformers import BertTokenizer, BertModel, DistilBertTokenizer, DistilBertModel
from datasets import load_dataset
from tqdm import tqdm
# from helpers import load_conll
import pandas as pd


BERT_MODEL = 'distilbert-base-uncased'
# Load pre-trained model tokenizer (vocabulary)
# tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
tokenizer = DistilBertTokenizer.from_pretrained(BERT_MODEL)

# Load pre-trained model (weights)
model = DistilBertModel.from_pretrained(BERT_MODEL,
                                  output_hidden_states = True, # Whether the model returns all hidden-states.
                                  )

# Put the model in "evaluation" mode, meaning feed-forward operation.
model.eval()

#preprocess data
def generate_masked_sentences(data):
    #creates new dataset where each token in a sentence is masked and used as a new sentence
    #the ner tag of the masked token is used as the label

    pbar = tqdm(total=len(data))
    new_data = []
    id_ = 0
    for sentence in data:
        for i in range(len(sentence['tokens'])):
            new_sentence = sentence['tokens'].copy()
            new_sentence[i] = '[MASK]'
            # Embed the input sentence
            word_embedding = get_word_embedding(new_sentence)
            new_data.append({'id': id_,
                             'tokens': new_sentence,
                             'is_ner': bool(sentence['ner_tags'][i]),
                             'ner_tag': sentence['ner_tags'][i],
                             'word_embedding': word_embedding})
            id_ += 1
        pbar.update(1)
    pbar.close()

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
    sentence_feats = [embedding['word_embedding'] for embedding in dataset]
    # Convert list sentence_feats to pytorch tensor
    sentence_feats = torch.stack(sentence_feats)
    # sentence_feats = convert_to_word_indices(dataset, word2idx, max_len, PAD)

    # Generate labels as a tensor of booleans indicating if the masked token is a named entity
    mask_labels = torch.tensor([sent['is_ner'] for sent in dataset], dtype=torch.float)
    # Add a dimension to labels
    mask_labels = mask_labels.unsqueeze(1)

    # Filter out all the sentences in dataset where the masked token is not a named entity, i.e. the is_ner field is False
    named_entity_data = [sent for sent in dataset if sent['is_ner']]
    

    named_entity_data_labels = torch.eye(num_entities-1)[[sent['ner_tag']-1 for sent in named_entity_data]]
    approach1_model_2_test_data = torch.eye(num_entities-1)[[sent['ner_tag']-1 for sent in dataset]] # named_entity_data_labels_full_length

    named_entity_sentence_feats = [embedding['word_embedding'] for embedding in named_entity_data]
    # Convert list named_entity_sentence_feats to pytorch tensor
    named_entity_sentence_feats = torch.stack(named_entity_sentence_feats)
    # named_entity_sentence_feats = convert_to_word_indices(named_entity_data, word2idx, max_len)

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



def get_word_embedding(text, sentence_length=32):
    # Pad or truncate the text to the specified sentence length
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=sentence_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    # Extract the input tensors
    input_ids = encoding["input_ids"]
    # segments_tensors = encoding["token_type_ids"]
    attention_mask = encoding["attention_mask"]

    # Run the text through BERT and collect all of the hidden states produced
    # from all 12 layers
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        # hidden_states = outputs[2]
        hidden_states = outputs[1]

    # Concatenate the tensors for all layers. We use `stack` here to
    # create a new dimension in the tensor.
    token_embeddings = torch.stack(hidden_states, dim=0)

    # Remove dimension 1, the "batches".
    token_embeddings = torch.squeeze(token_embeddings, dim=1)

    # Swap dimensions 0 and 1.
    token_embeddings = token_embeddings.permute(1, 0, 2)

    # Stores the token vectors, with shape [sentence_length x 768]
    token_vecs_sum = torch.zeros([token_embeddings.size()[0], 768])

    # `token_embeddings` is a [sentence_length x 12 x 768] tensor.

    # For each token in the sentence...
    for index, token in enumerate(token_embeddings):

        # `token` is a [12 x 768] tensor

        # Sum the vectors from the last four layers.
        sum_vec = torch.sum(token[-4:], dim=0)

        # Use `sum_vec` to represent `token`.
        token_vecs_sum[index] = sum_vec

    # Remove the first and the last token
    # token_vecs_sum = token_vecs_sum[1:-1]

    return token_vecs_sum
