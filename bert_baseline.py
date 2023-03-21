"""
A basic classifier based on the transformers (https://github.com/huggingface/transformers) 
library. It loads a masked language model (by default mBERT), and adds a linear layer for
prediction. Example usage:

python3 bert-topic.py topic-data/train.txt topic-data/dev.txt
"""
from typing import List, Dict
import codecs
import torch
import sys
import myutils
from transformers import AutoModel, AutoTokenizer

# set seed for consistency
torch.manual_seed(8446)
# Set some constants
MLM = 'distilbert-base-cased'
BATCH_SIZE = 8
LEARNING_RATE = 0.00001
EPOCHS = 3
# We have an UNK label for robustness purposes, it makes it easier to run on
# data with other labels, or without labels.
UNK = "[UNK]"
MAX_TRAIN_SENTS=64
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


class ClassModel(torch.nn.Module):
    def __init__(self, nlabels: int, mlm: str):
        """
        Model for classification with transformers.

        The architecture of this model is simple, we just have a transformer
        based language model, and add one linear layer to converts it output
        to our prediction.
    
        Parameters
        ----------
        nlabels : int
            Vocabulary size of output space (i.e. number of labels)
        mlm : str
            Name of the transformers language model to use, can be found on:
            https://huggingface.co/models
        """
        super().__init__()

        # The transformer model to use
        self.mlm = AutoModel.from_pretrained(mlm)

        # Find the size of the output of the masked language model
        if hasattr(self.mlm.config, 'hidden_size'):
            self.mlm_out_size = self.mlm.config.hidden_size
        elif hasattr(self.mlm.config, 'dim'):
            self.mlm_out_size = self.mlm.config.dim
        else: # if not found, guess
            self.mlm_out_size = 768

        # Create prediction layer
        self.hidden_to_label = torch.nn.Linear(self.mlm_out_size, nlabels)

    def forward(self, input: torch.tensor):
        """
        Forward pass
    
        Parameters
        ----------
        input : torch.tensor
            Tensor with wordpiece indices. shape=(batch_size, max_sent_len).

        Returns
        -------
        output_scores : torch.tensor
            ?. shape=(?,?)
        """
        # Run transformer model on input
        mlm_out = self.mlm(input)

        # Keep only the last layer: shape=(batch_size, max_len, DIM_EMBEDDING)
        mlm_out = mlm_out.last_hidden_state
        # Keep only the output for the first ([CLS]) token: shape=(batch_size, DIM_EMBEDDING)
        mlm_out = mlm_out[:,:1,:].squeeze()

        # Matrix multiply to get scores for each label: shape=(?,?)
        output_scores = self.hidden_to_label(mlm_out)

        return output_scores

    def run_eval(self, text_batched: List[torch.tensor], labels_batched: List[torch.tensor]):
        """
        Run evaluation: predict and score
    
        Parameters
        ----------
        text_batched : List[torch.tensor]
            list with batches of text, containing wordpiece indices.
        labels_batched : List[torch.tensor]
            list with batches of labels (converted to ints).
        model : torch.nn.module
            The model to use for prediction.
    
        Returns
        -------
        score : float
            accuracy of model on labels_batches given feats_batches
        """
        self.eval()
        match = 0
        total = 0
        for sents, labels in zip(text_batched, labels_batched):
            output_scores = self.forward(sents)
            pred_labels = torch.argmax(output_scores, 1)
            for gold_label, pred_label in zip(labels, pred_labels):
                total += 1
                if gold_label.item() == pred_label.item():
                    match+= 1
        return(match/total)        


if len(sys.argv) < 2:
    print('Please provide path to training and development data')

if __name__ == '__main__':
    print('reading data...')
    train_text, train_labels = myutils.read_data(sys.argv[1])
    train_text = train_text[:MAX_TRAIN_SENTS]
    train_labels = train_labels[:MAX_TRAIN_SENTS]
    
    id2label, label2id = myutils.labels2lookup(train_labels, UNK)
    NLABELS = len(id2label)
    print(train_labels)
    print(label2id)
    train_labels = [label2id[label] for label in train_labels]
    
    dev_text, dev_labels = myutils.read_data(sys.argv[2])
    dev_labels = [label2id[label] for label in dev_labels]
    
    print('tokenizing...')
    tokzr = AutoTokenizer.from_pretrained(MLM)
    train_tokked = myutils.tok(train_text, tokzr)
    dev_tokked = myutils.tok(dev_text, tokzr)
    PAD = tokzr.pad_token_id
    
    print('converting to batches...')
    train_text_batched, train_labels_batched = myutils.to_batch(train_tokked, train_labels, BATCH_SIZE, PAD, DEVICE)
    # Note, some data is trown away if len(text_tokked)%BATCH_SIZE!= 0
    dev_text_batched, dev_labels_batched = myutils.to_batch(dev_tokked, dev_labels, BATCH_SIZE, PAD, DEVICE)
    
    print('initializing model...')
    model = ClassModel(NLABELS, MLM)
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_function = torch.nn.CrossEntropyLoss(ignore_index=0, reduction='sum')
    
    print('training...')
    for epoch in range(EPOCHS):
        print('=====================')
        print('starting epoch ' + str(epoch))
        model.train() 
    
        # Loop over batches
        loss = 0
        for batch_idx in range(0, len(train_text_batched)):
            optimizer.zero_grad()

            output_scores = model.forward(train_text_batched[batch_idx])
            batch_loss = loss_function(output_scores, train_labels_batched[batch_idx])
            loss += batch_loss.item()
    
            batch_loss.backward()

            optimizer.step()
    
        dev_score = model.run_eval(dev_text_batched, dev_labels_batched)
        print('Loss: {:.2f}'.format(loss))
        print('Acc(dev): {:.2f}'.format(100*dev_score))
        print()
    
