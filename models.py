import torch
import torch.nn as nn
from datasets import load_dataset
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

from scripts.preprocess import generate_masked_sentences, generate_word2idx, preprocess_data


# wnut = load_dataset("wnut_17")

# # Preprocess data
# train_data = generate_masked_sentences(wnut['train'])
# test_data = generate_masked_sentences(wnut['test'])

# # Define hyperparameters to share between all three approaches
# # Ideally as much as possible should be shared so we can compare the approaches with more or less the same architecture
# PAD = '<PAD>'
# batch_size = 32
# num_epochs = 3
# lr = 0.001
# max_len=32 # Length of sentence
# emb_dim = 128 # The embedding dimension of each token 

# word2idx, idx2word = generate_word2idx(train_data, max_len, PAD)
# # Vocab length
# vocab_dim = len(idx2word)

# # Get all ner_tags from wnut_17 dataset
# ner_tags = wnut['train'].features['ner_tags'].feature.names
# num_entities = len(ner_tags) # Number of entities including non entity
# print("Number of entities", num_entities)

# train_sentence_feats, train_mask_labels, train_named_entity_sentence_feats, train_named_entity_data_labels, _, train_approach2_labels, train_approach_3_task_2_labels = preprocess_data(train_data, word2idx, max_len, num_entities, PAD)
# test_sentence_feats, test_mask_labels, test_named_entity_sentence_feats, test_named_entity_data_labels, approach1_model_2_test_data, test_approach2_labels, test_approach_3_task_2_labels = preprocess_data(test_data, word2idx, max_len, num_entities, PAD)


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
        self.linear = nn.Linear(emb_dim, 128)
        # Pool together all word embeddings after linear layer
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.output = nn.Linear(128, 1)

    
    def forward(self, x):
        x = self.word_embeddings(x)
        x = self.linear(x)
        x = nn.functional.relu(x)
        x = self.pool(x.transpose(1, 2)).squeeze(2)
        output = self.output(x)
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
        self.linear = nn.Linear(emb_dim, 128)
        # Pool together all word embeddings after linear layer
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.output = nn.Linear(128, num_entities - 1) # Subtract 1 since we don't include the non named entity class

    
    def forward(self, x):
        x = self.word_embeddings(x)
        x = self.linear(x)
        x = nn.functional.relu(x)
        x = self.pool(x.transpose(1, 2)).squeeze(2)        
        output = self.output(x)
        output = torch.softmax(output, dim=1)
        
        return output
    

# approach1_mask_prediction = Approach1MaskPrediction(vocab_dim, emb_dim)
# criterion = nn.BCELoss()
# optimizer = torch.optim.Adam(approach1_mask_prediction.parameters(), lr=lr)

# print("Approach 1: Training mask prediction model")
# for epoch in range(num_epochs):
#     for i in range(0, len(train_sentence_feats), batch_size):        
#         batch_feats = train_sentence_feats[i:i+batch_size]
#         batch_labels = train_mask_labels[i:i+batch_size]
#         optimizer.zero_grad()
#         y_pred = approach1_mask_prediction(batch_feats)
#         loss = criterion(y_pred, batch_labels)
#         loss.backward()
#         optimizer.step()    
#     print("Epoch: {}/{}...".format(epoch+1, num_epochs),
#             "Loss: {:.6f}...".format(loss.item()))
    
# approach1_entity_classification = Approach1EntityClassification(vocab_dim, emb_dim)
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(approach1_entity_classification.parameters(), lr=lr)

# print("Approach 1: Training entity classification model")
# for epoch in range(num_epochs):
#     for i in range(0, len(train_named_entity_sentence_feats), batch_size):
#         batch_feats = train_named_entity_sentence_feats[i:i+batch_size]
#         batch_labels = train_named_entity_data_labels[i:i+batch_size]
#         optimizer.zero_grad()
#         y_pred = approach1_entity_classification(batch_feats)
#         loss = criterion(y_pred, batch_labels)
#         loss.backward()
#         optimizer.step()

#     print("Epoch: {}/{}...".format(epoch+1, num_epochs),
#         "Loss: {:.6f}...".format(loss.item()))
    

# # Evaluate performance
# approach1_mask_prediction.eval()

# with torch.no_grad():
#     y_pred = approach1_mask_prediction(test_sentence_feats)
#     threshold = 0.5
#     y_pred = (y_pred > threshold).float() # Convert probabilities to binary predictions
#     y_test = test_mask_labels
#     print("Accuracy of mask prediction model", accuracy_score(y_test, y_pred))

# # Only run Approach1EntityClassification if y_pred is 1
# approach1_entity_classification.eval()

# with torch.no_grad():
#     # Get indices where y_pred is 1
#     indices = torch.nonzero(y_pred).squeeze(1)
#     indices = indices[:, 0]
#     y_pred = approach1_entity_classification(test_sentence_feats[indices])
#     y_pred = torch.argmax(y_pred, axis=1)
#     y_test = torch.argmax(approach1_model_2_test_data[indices], axis=1)
#     # print("Accuracy of entity classification model", accuracy_score(y_test, y_pred))
#     print("F1 score of entity classification model", f1_score(y_test, y_pred, average='macro'))

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
        self.linear = nn.Linear(emb_dim, 128)
        # Pool together all word embeddings after linear layer
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.output = nn.Linear(128, num_entities)

    
    def forward(self, x):
        x = self.word_embeddings(x)
        x = self.linear(x)
        x = nn.functional.relu(x)
        x = self.pool(x.transpose(1, 2)).squeeze(2)
        output = self.output(x)
        output = torch.softmax(output, dim=1)
        
        return output
    
# approach2model = Approach2(vocab_dim, emb_dim)
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(approach2model.parameters(), lr=lr)

# for epoch in range(num_epochs):
#     for i in range(0, len(train_sentence_feats), batch_size):        
#         batch_feats = train_sentence_feats[i:i+batch_size]
#         batch_labels = train_approach2_labels[i:i+batch_size]    
#         optimizer.zero_grad()
#         y_pred = approach2model(batch_feats)
#         loss = criterion(y_pred, batch_labels)
#         loss.backward()
#         optimizer.step()
#     print("Epoch: {}/{}...".format(epoch+1, num_epochs),
#         "Loss: {:.6f}...".format(loss.item()))

# # Evaluate
# approach2model.eval()

# with torch.no_grad():
#     y_pred = approach2model(test_sentence_feats)
#     y_pred = torch.argmax(y_pred, axis=1)
#     y_test = torch.argmax(test_approach2_labels, axis=1)
#     print("Accuracy of approach 2 model", accuracy_score(y_test, y_pred))
#     print("F1 score of approach 2 model", f1_score(y_test, y_pred, average='macro'))

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
        self.linear = nn.Linear(emb_dim, 128)
        # Pool together all word embeddings after linear layer
        self.pool = nn.AdaptiveMaxPool1d(1)

        # Model 1 specific layers
        self.model1_output = nn.Linear(128, 1)

        # Add a gating mechanism to decide whether to run model 2
        self.gate = nn.Linear(1, 1)

        # Model 2 specific layers        
        # Output a single value for whether the masked token is a named entity
        self.model2_output = nn.Linear(128, num_entities - 1)

        
    def forward(self, x):
        x = self.word_embeddings(x)
        x = self.linear(x)        
        x = nn.functional.relu(x)
        # Pool over the output of relu        
        x = self.pool(x.transpose(1, 2)).squeeze(2)  

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
    

# combined_model = Approach3CombinedModel(vocab_dim, 128)

# criterion1 = nn.BCELoss()
# criterion2 = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(combined_model.parameters(), lr=lr)

# for epoch in range(num_epochs):
#     for i in range(0, len(train_sentence_feats), batch_size):
#         batch_feats = train_sentence_feats[i:i+batch_size]
#         batch_labels_1 = train_mask_labels[i:i+batch_size] # Labels if mask is a named entity or not
#         batch_labels_2 = train_approach_3_task_2_labels[i:i+batch_size] # Labels for what type of named entity the mask is
#         optimizer.zero_grad()
#         y_pred1, y_pred2 = combined_model(batch_feats)        
#         loss1 = criterion1(y_pred1, batch_labels_1)        
#         loss2 = criterion2(y_pred2, batch_labels_2)
#         loss = loss1 + loss2
#         loss.backward()
#         optimizer.step()

#     print("Epoch: {}/{}...".format(epoch+1, num_epochs),
#             "Loss: {:.6f}...".format(loss.item()))
    
# # Evaluate
# combined_model.eval()


# with torch.no_grad():
#     y_pred1, y_pred2 = combined_model(test_sentence_feats)
#     y_pred1 = torch.round(y_pred1)
#     y_pred2 = torch.argmax(y_pred2, axis=1)
#     y_test1 = test_mask_labels
#     y_test2 = torch.argmax(test_approach_3_task_2_labels, axis=1)
#     print("Accuracy of approach 3 model 1", accuracy_score(y_test1, y_pred1))
#     print("F1 score of approach 3 model 1", f1_score(y_test1, y_pred1, average='macro'))
#     # print("Accuracy of approach 3 model 2", accuracy_score(y_test2, y_pred2))
#     print("F1 score of approach 3 model 2", f1_score(y_test2, y_pred2, average='macro'))

############### End approach 3 model ###############