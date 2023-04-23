from typing import List
import codecs
import torch
from transformers import AutoTokenizer

def read_data(file_path: str):
    """
    Read data and labels from a file
    Reads text data and labels from a file, assuming that the file
    is tab-separated, has the labels in the first column, and the 
    text in the second.
    Parameters
    ----------
    file_path : str
        The path to the file.
    Returns
    -------
    sents : list
        list of strings (sentences).
    labels : list
        list of labels.
    """
    sents = []
    labels = []

    for line in codecs.open(file_path, encoding='utf-8'):
        tok = line.strip().split('\t')
        sents.append(tok[1])
        labels.append(tok[0])

    return sents, labels


def tok(data: List[str], tokzr: AutoTokenizer):
    """
    Read data and labels from a file
    Reads text data and labels from a file, assuming that the file
    is tab-separated, has the labels in the first column, and the 
    text in the second.
    Parameters
    ----------
    data : List[str]
        List of sentences
    tokzr : AutoTokenizer
        Transformers AutoTokenizer to use
    Returns
    -------
    tok_data : List[List[int]]
        list of lists of subword indices, includeing special start and
        end tokens.
    """
    tok_data = []

    for sent in data:
        tok_data.append(tokzr.encode(sent))

    return tok_data

def to_batch(text: List[List[int]], labels: List[int], batch_size: int, padding_id: int, DEVICE: str):
    """
    Convert a list of inputs and labels to batches of size batch_size.
    
    We do not sort by size as is quite standard, because having varied 
    batches can be beneficial for robustness. Altough it might be less
    efficient.
    Note that some sentences might be not used if len(data)%size != 0
    If you want to include all, for example for dev/test data, just use
    batch_size = 1
    Parameters
    ----------
    text : List[List[int]]
        List of lists of wordpiece indices.
    labels : List[int]
        List of gold labels converted to indices.
    batch_size : int
        The number of instances to put in a batch.
    padding_id : int
        The id for the special padding token.
    device : str
        Description of CUDA device (gpu).
    Returns
    -------
    data_batches : List[torch.tensor]
        A list of tensors of size batch_size*max_len_of_batch
    label_batches : List[torch.tensor]
        A list of tensors of size batch_size
    """
    text_batches = []
    label_batches = []
    num_batches = int(len(text)/batch_size)

    for batch_idx in range(num_batches):
        beg_idx = batch_idx * batch_size
        end_idx = (batch_idx+1) * batch_size
        max_len = max([len(sent) for sent in text[beg_idx:end_idx]])

        new_batch_text = torch.full((batch_size, max_len), padding_id, dtype = torch.long, device=DEVICE)
        new_batch_labels = torch.zeros(batch_size, dtype = torch.long, device=DEVICE)

        for sent_idx in range(batch_size):
            new_batch_labels[sent_idx] = labels[beg_idx + sent_idx]
            for word_idx, word_id in enumerate(text[beg_idx + sent_idx]):
                new_batch_text[sent_idx][word_idx] = word_id
        text_batches.append(new_batch_text)
        label_batches.append(new_batch_labels)

    return text_batches, label_batches

def labels2lookup(labels: List[str], PAD):
    """
    Convert a list of strings to a lookup dictionary of id's
    
    Parameters
    ----------
    labels : List[str]
        List of strings to index.
    Returns
    -------
    id2label : 
        List with all types of the input.
    label2id : 
        Lookup dictionary, converting every type of the input to an id.
    """
    id2label = [PAD]
    label2id = {PAD: 0}
    for label in labels:
        if label not in label2id:
            label2id[label] = len(label2id)
            id2label.append(label)
    return id2label, label2id