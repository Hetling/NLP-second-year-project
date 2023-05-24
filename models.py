import torch
import torch.nn as nn
from datasets import load_dataset
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

from scripts.preprocess import generate_masked_sentences, generate_word2idx, preprocess_data

############### Begin approach 1 model ###############
class Approach1MaskPrediction(nn.Module):    
    def __init__(self, vocab_dim, emb_dim):
        '''
        First model in approach 1
        This model serves the purpose of predicting whether a masked token is a named entity or not
        Then we train another model that predicts the actual named entity class
        '''
        super(Approach1MaskPrediction, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_dim, emb_dim)
        # LSTM layer
        self.lstm = nn.LSTM(emb_dim, 128, batch_first=True)        
        # Pool together all LSTM hidden states
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.linear = nn.Linear(128, 128)
        self.output = nn.Linear(128, 1)

    
    def forward(self, x):
        x = self.word_embeddings(x)
        x, _ = self.lstm(x)
        x = self.pool(x.transpose(1, 2)).squeeze(2)
        output = self.output(x)
        x = self.linear(x)
        x = nn.functional.relu(x)

        output = torch.sigmoid(output)
        return output

class Approach1EntityClassification(nn.Module):
    def __init__(self, vocab_dim, emb_dim, num_entities):
        '''
        Second model in approach 1
        This model predicts the named entity class of a masked token
        '''
        super(Approach1EntityClassification, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_dim, emb_dim)
        # LSTM
        self.lstm = nn.LSTM(emb_dim, 128, batch_first=True)
        # Pool together all word embeddings after linear layer
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.linear = nn.Linear(128, 128)
        self.output = nn.Linear(128, num_entities - 1) # Subtract 1 since we don't include the non named entity class

    
    def forward(self, x):
        x = self.word_embeddings(x)
        x, _ = self.lstm(x)
        x = self.pool(x.transpose(1, 2)).squeeze(2)        
        x = self.linear(x)
        x = nn.functional.relu(x)
        output = self.output(x)
        output = torch.softmax(output, dim=1)
        
        return output

############### End approach 1 model ###############

############### Begin approach 2 model ###############
class Approach2(nn.Module):
    def __init__(self, vocab_dim, emb_dim, num_entities):
        '''
        The only model in approach 2
        This model takes as input a sentence with one masked word and predicts it to be one of 1 + num_entities classes
        That is it predicts the masked token to be either a non named entity or one of the num_entities entity classes.            
        '''
        super(Approach2, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_dim, emb_dim)
        # LSTM
        self.lstm = nn.LSTM(emb_dim, 128, batch_first=True)
        # Pool together all word embeddings after linear layer
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.linear = nn.Linear(128, 128)
        self.output = nn.Linear(128, num_entities)

    
    def forward(self, x):
        x = self.word_embeddings(x)
        x, _ = self.lstm(x)
        x = self.pool(x.transpose(1, 2)).squeeze(2)
        x = self.linear(x)
        x = nn.functional.relu(x)
        output = self.output(x)
        output = torch.softmax(output, dim=1)
        
        return output
    
############### End approach 2 model ###############

############### Begin approach 3 model ###############
class Approach3CombinedModel(nn.Module):
    def __init__(self, vocab_dim, emb_dim, num_entities):
        '''
        This is the combined model that shares weights between both the masked token prediction and entity classification
        Model 1 predicts if a masked token is a named entity
        Model 2 predicts what type of named entity a masked token is
        The two models output two individually predictions, and thus the loss functions for the two objectives are added together
        '''
        super(Approach3CombinedModel, self).__init__()
        # First part of the model is same between the two models
        self.word_embeddings = nn.Embedding(vocab_dim, emb_dim)
        # LSTM
        self.lstm = nn.LSTM(emb_dim, 128, batch_first=True)
        # Pool together all word embeddings after linear layer
        self.pool = nn.AdaptiveMaxPool1d(1)

        self.linear = nn.Linear(128, 128)

        # Model 1 specific layers
        self.model1_output = nn.Linear(128, 1)

        # Add a gating mechanism to decide whether to run model 2
        self.gate = nn.Linear(1, 1)

        # Model 2 specific layers        
        # Output a single value for whether the masked token is a named entity
        self.model2_output = nn.Linear(128, num_entities - 1)

        
    def forward(self, x):
        x = self.word_embeddings(x)
        # Pool over the output of relu        
        x = self.pool(x.transpose(1, 2)).squeeze(2)  
        x = self.linear(x)
        x = nn.functional.relu(x)

        # Run through model 1
        model1_output = self.model1_output(x)
        model1_output = torch.sigmoid(model1_output)

        # Run through model 2 if model 1 predicts that the masked token is a named entity
        model2_output = self.model2_output(x)
        model2_output = torch.softmax(model2_output, dim=1)
        # Run through gate
        gate_output = self.gate(model1_output)        
        gate_output = nn.functional.relu(gate_output)
        model2_output = model2_output * gate_output
        # The gating mechanism should be able to learn to not run model 2 if model 1 predicts that the masked token is not a named entity 

        return model1_output, model2_output
    
############### End approach 3 model ###############