import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset

from scripts.maskPrecessTest import generate_masked_sentences


wnut = load_dataset("wnut_17")

#preprocess data
train_data = generate_masked_sentences(wnut['train'])


# Define hyperparameters to share between all three approaches
# Ideally as much as possible should be shared so we can compare the approaches with more or less the same architecture
batch_size = 32
num_epochs = 10
lr = 0.001
max_len=32 # Length of sentence
emb_dim = 128 # The embedding dimension of each token 


# Create word to index mappings
PAD = '<PAD>'
word2idx = {PAD:0}
idx2word = [PAD]

# Generate word2idxs
for sentPos, sent in enumerate(train_data):
    for wordPos, word in enumerate(sent['tokens'][:max_len]):
        if word not in word2idx:
            word2idx[word] = len(idx2word)
            idx2word.append(word)

# Vocab length
vocab_dim = len(idx2word)

# Convert training dataset to word indices
feats = torch.zeros((len(train_data), max_len), dtype=torch.long)
for sentPos, sent in enumerate(train_data):
    for wordPos, word in enumerate(sent['tokens'][:max_len]):
        wordIdx = word2idx[PAD] if word not in word2idx else word2idx[word]
        feats[sentPos][wordPos] = wordIdx

# Generate labels as a tensor of booleans indicating if the masked token is a named entity
labels = torch.tensor([sent['is_ner'] for sent in train_data], dtype=torch.float)


# Generate named entity class labels:
# Get all ner_tags from wnut_17 dataset
ner_tags = wnut['train'].features['ner_tags'].feature.names
# Create a dictionary mapping ner_tags to indices
idx2ner = {i:ner for i, ner in enumerate(ner_tags)}
num_entities = len(idx2ner)
print("Number of entities", num_entities)

# Filter out all the sentences in train_data where the masked token is not a named entity, i.e. the is_ner field is False
filtered_train_data = [sent for sent in train_data if sent['is_ner']]
# Create a list of one hot encoded vectors for each sentence in filtered_train_data
true_ner_tags = torch.zeros((len(filtered_train_data), num_entities-1), dtype=torch.float)
for sentPos, sent in enumerate(filtered_train_data):
    # We minus 1 to the ner_tag since the ner_tag is 1 indexed (as the first index is reserved for the class of non named entities)
    true_ner_tags[sentPos][sent['ner_tag']-1] = 1
print("Shape of true_ner_tags", true_ner_tags.shape)

named_entity_sentence_feats = torch.zeros((len(filtered_train_data), max_len), dtype=torch.long)
for sentPos, sent in enumerate(filtered_train_data):
    for wordPos, word in enumerate(sent['tokens'][:max_len]):
        wordIdx = word2idx[PAD] if word not in word2idx else word2idx[word]
        named_entity_sentence_feats[sentPos][wordPos] = wordIdx


# Generate an array of size (num_feats, num_entities + 1) where each row is a one hot encoded vector of the named entity class or non named entity class
approach2_labels = torch.zeros((len(train_data), num_entities+1), dtype=torch.float)
for sentPos, sent in enumerate(train_data):
    # If the masked token is not a named entity, then the label is the non named entity class
    if not sent['is_ner']:
        approach2_labels[sentPos][0] = 1
    else:
        # NOTE: Maybe add 1 here? (probably not though)        
        approach2_labels[sentPos][sent['ner_tag']] = 1

############### Begin approach 1 model ###############
class Approach1MaskPrediction(nn.module);
    def __init__(self, vocab_dim, emb_dim):
        '''
        First model in approach 1
        This model serves the purpose of predicting whether a masked token is a named entity or not
        Then we train another model that predicts the actual named entity class
        '''
        super(Approach1MaskPrediction, self).__init__()        
        self.word_embeddings = nn.Embedding(vocab_dim, emb_dim)
        self.linear = nn.Linear(emb_dim, 128)                
        # Pool together all word embeddings after linear layer
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.output = nn.Linear(128, 1)

    
    def forward(self, x):
        x = self.word_embeddings(x)
        x = self.linear(x)
        x = nn.ReLU(x)
        x = self.pool(x.transpose(1, 2)).squeeze(2)        
        output = self.output(x)
        output = torch.sigmoid(output)

        return output

class Approach1EntityClassification(nn.module);
    def __init__(self, vocab_dim, emb_dim):
        '''
        Second model in approach 1
        This model predicts the named entity class of a masked token        
        '''
        super(Approach1EntityClassification, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_dim, emb_dim)
        self.linear = nn.Linear(emb_dim, 128)
        # Pool together all word embeddings after linear layer
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.output = nn.Linear(128, num_entities)

    
    def forward(self, x):
        x = self.word_embeddings(x)
        x = self.linear(x)
        x = nn.ReLU(x)
        x = self.pool(x.transpose(1, 2)).squeeze(2)        
        output = self.output(x)
        output = torch.softmax(output, dim=1)
        
        return output
    

approach1_mask_prediction = Approach1MaskPrediction(vocab_dim, emb_dim)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(approach1_mask_prediction.parameters(), lr=lr)

for epoch in range(num_epochs):
    for i in range(0, len(feats), batch_size):        
        batch_feats = feats[i:i+batch_size]
        batch_labels = labels[i:i+batch_size]
        optimizer.zero_grad()
        y_pred = approach1_mask_prediction(batch_feats)
        loss = criterion(y_pred, batch_labels)
        loss.backward()
        optimizer.step()    
    print("Epoch: {}/{}...".format(epoch+1, num_epochs),
            "Loss: {:.6f}...".format(loss.item()))


# TODO: Evaluate performance


approach1_entity_classification = Approach1EntityClassification(vocab_dim, emb_dim)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(approach1_entity_classification.parameters(), lr=lr)

for epoch in range(num_epochs):
    for i in range(0, len(named_entity_sentence_feats), batch_size):        
        batch_feats = named_entity_sentence_feats[i:i+batch_size]
        batch_labels = true_ner_tags[i:i+batch_size]
        optimizer.zero_grad()
        y_pred = approach1_entity_classification(batch_feats)
        loss = criterion(y_pred, batch_labels)
        loss.backward()
        optimizer.step()
    print("Epoch: {}/{}...".format(epoch+1, num_epochs),
        "Loss: {:.6f}...".format(loss.item()))


############### End approach 1 model ###############

############### Begin approach 2 model ###############
class Approach2(nn.module);
    def __init__(self, vocab_dim, emb_dim):
        '''
        The only model in approach 2
        This model takes as input a sentence with one masked word and predicts it to be one of 1 + num_entities classes
        That is it predicts the masked token to be either a non named entity or one of the num_entities entity classes.            
        '''
        super(Approach2, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_dim, emb_dim)
        self.linear = nn.Linear(emb_dim, 128)
        # Pool together all word embeddings after linear layer
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.output = nn.Linear(128, num_entities + 1)

    
    def forward(self, x):
        x = self.word_embeddings(x)
        x = self.linear(x)
        x = nn.ReLU(x)
        x = self.pool(x.transpose(1, 2)).squeeze(2)
        output = self.output(x)
        output = torch.softmax(output, dim=1)
        
        return output
    
approach2model = Approach2(vocab_dim, emb_dim)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(approach2model.parameters(), lr=lr)

for epoch in range(num_epochs):
    for i in range(0, len(feats), batch_size):        
        batch_feats = feats[i:i+batch_size]
        batch_labels = approach2_labels[i:i+batch_size]    
        optimizer.zero_grad()
        y_pred = approach2model(batch_feats)
        loss = criterion(y_pred, batch_labels)
        loss.backward()
        optimizer.step()
    print("Epoch: {}/{}...".format(epoch+1, num_epochs),
        "Loss: {:.6f}...".format(loss.item()))


# TODO: Prepare data for this approach and code training loop    
############### End approach 2 model ###############

############### Begin approach 3 model ###############
class Approach3CombinedModel(nn.Module):
    def __init__(self, vocab_dim, emb_dim):
        '''
        This is the combined model that shares weights between both the masked token prediction and entity classification
        Model 1 predicts if a masked token is a named entity
        Model 2 predicts what type of named entity a masked token is
        The two models output two individually predictions, and thus the loss functions for the two objectives are added together
        '''
        super(Approach3CombinedModel, self).__init__()
        # First part of the model is same between the two models
        self.word_embeddings = nn.Embedding(vocab_dim, emb_dim)
        self.linear = nn.Linear(emb_dim, 128)
        # Pool together all word embeddings after linear layer
        self.pool = nn.AdaptiveMaxPool1d(1)

        # Model 1 specific layers
        self.model1_output = nn.Linear(128, 1)

        # Add a gating mechanism to decide whether to run model 2
        self.gate = nn.Linear(1, 1)

        # Model 2 specific layers        
        # Output a single value for whether the masked token is a named entity
        self.model2_output = nn.Linear(128, num_entities)

        
    def forward(self, x):
        x = self.word_embeddings(x)
        x = self.linear(x)
        x = nn.ReLU(x)
        x = self.pool(x.transpose(1, 2)).squeeze(2)
        # Run through model 1
        model1_output = self.model1_output(x)
        model1_output = torch.sigmoid(model1_output)

        # Run through model 2 if model 1 predicts that the masked token is a named entity
        model2_output = self.model2_output(x)
        model2_output = torch.softmax(model2_output, dim=1)
        # Run through gate
        gate_output = self.gate(model1_output)        
        gate_output = nn.ReLU(gate_output)
        model2_output = model2_output * gate_output
        # The gating mechanism should be able to learn to not run model 2 if model 1 predicts that the masked token is not a named entity 

        return model1_output, model2_output
    

combined_model = Approach3CombinedModel(vocab_dim, 128)

criterion1 = nn.BCELoss()
criterion2 = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(combined_model.parameters(), lr=lr)

for epoch in range(num_epochs):
    for i in range(0, len(feats), batch_size):
        batch_feats = feats[i:i+batch_size]
        batch_labels_1 = labels[i:i+batch_size] # Labels if mask is a named entity or not
        batch_labels_2 = true_ner_tags[i:i+batch_size] # Labels for what type of named entity the mask is
        optimizer.zero_grad()
        y_pred1, y_pred2 = combined_model(batch_feats)
        loss1 = criterion1(y_pred1, batch_labels_1)
        loss2 = criterion2(y_pred2, batch_labels_2)           
        loss = loss1 + loss2
        loss.backward()
        optimizer.step()

    print("Epoch: {}/{}...".format(epoch+1, num_epochs),
            "Loss: {:.6f}...".format(loss.item()))
    

############### Begin approach 3 model ###############