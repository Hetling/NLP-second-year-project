import os
import argparse
import torch
import torch.nn as nn
from datasets import load_dataset
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import pickle
import numpy as np
from scripts.preprocess import generate_masked_sentences, generate_word2idx, preprocess_data

from models import Approach1MaskPrediction, Approach1EntityClassification, Approach2, Approach3CombinedModel

def main():
    torch.manual_seed(17)
    # Create the argument parser
    parser = argparse.ArgumentParser(description='Script for loading data and training neural networks')

    # Add the command line arguments
    parser.add_argument('--approach-1', action='store_true', help='Perform action on approach 1')
    parser.add_argument('--approach-2', action='store_true', help='Perform action on approach 2')
    parser.add_argument('--approach-3', action='store_true', help='Perform action on approach 3')
    parser.add_argument('--save', action='store_true', default=True, help='Save the trained models')
    parser.add_argument('--train', action='store_true', help='Train the models')
    parser.add_argument('--validate', action='store_true', default=False, help='Evaluate the trained models')
    parser.add_argument('--test', action='store_true', default=False, help='Test the trained models on the test set')

    # Parse the arguments
    args = parser.parse_args()

    # Check if any of the approaches were selected
    models_selected = []
    if not args.approach_1 and not args.approach_2 and not args.approach_3:
        # Select all approaches
        models_selected = [1, 2, 3]
    if args.approach_1:
        # Select approach 1
        models_selected.append(1)
    if args.approach_2:
        # Select approach 2
        models_selected.append(2)
    if args.approach_3:
        # Select approach 3
        models_selected.append(3)

    # Define hyperparameters to share between all three approaches
    # All these parameters are shared between models so we can compare the approaches as fairly as possible
    PAD = '<PAD>'
    batch_size = 32
    num_epochs = 10
    lr = 0.001
    max_len = 32 # Length of sentence
    emb_dim = 128 # The embedding dimension of each token 


    # Check if the models directory exists
    if not os.path.exists('models'):
        # Create the models directory
        os.makedirs('models')
    
    wnut = load_dataset("wnut_17")

    global label_list
    label_list=wnut["train"].features[f"ner_tags"].feature.names

    # Preprocess data
    # Check if models/data/train_data.pkl and models/data/test_data.pkl exists

    if not os.path.exists('models/data'):    
        os.makedirs('models/data')

    train_data_exists = os.path.exists('models/data/train_data.pkl')
    test_data_exists = os.path.exists('models/data/test_data.pkl')
    val_data_exists = os.path.exists('models/data/val_data.pkl')

    # If the train data does not exist, generate it
    if not train_data_exists:
        train_data = generate_masked_sentences(wnut['train'])
        # Save the processed data
        with open('models/data/train_data.pkl', 'wb') as f:
            pickle.dump(train_data, f)
    else:
        # Load data
        with open('models/data/train_data.pkl', 'rb') as f:
            train_data = pickle.load(f)

    # If the test data does not exist, generate it
    if not test_data_exists:
        test_data = generate_masked_sentences(wnut['test'])
        # Save the processed data
        with open('models/data/test_data.pkl', 'wb') as f:
            pickle.dump(test_data, f)
    else:
        # Load data
        with open('models/data/test_data.pkl', 'rb') as f:
            test_data = pickle.load(f)

    if not val_data_exists:
        val_data = generate_masked_sentences(wnut['validation'])
        # Save the processed data
        with open('models/data/val_data.pkl', 'wb') as f:
            pickle.dump(val_data, f)
    else:
        # Load data
        with open('models/data/val_data.pkl', 'rb') as f:
            val_data = pickle.load(f)

    # Load datasets
    word2idx, idx2word = generate_word2idx(train_data, max_len, PAD)
    # Vocab length
    vocab_dim = len(idx2word)
    print("Vocab length", vocab_dim)

    # Get all ner_tags from wnut_17 dataset
    ner_tags = wnut['train'].features['ner_tags'].feature.names
    num_entities = len(ner_tags) # Number of entities including non entity
    print("Number of entities", num_entities)

    state = {
        'PAD': PAD,
        'batch_size': batch_size,
        'num_epochs': num_epochs,
        'lr': lr,
        'max_len': max_len,
        'emb_dim': emb_dim,
        'num_entities': num_entities,
        'vocab_dim': vocab_dim,
        'word2idx': word2idx,
        'save': args.save
    }
        

    if args.train:
        train(models_selected, train_data, state)

    if args.test:
        evaluate(models_selected, test_data, state)

    if args.validate:
        evaluate(models_selected, val_data, state)

def train(models_to_train: list, train_data: list, state: dict):


    train_sentence_feats, train_mask_labels, train_named_entity_sentence_feats, train_named_entity_data_labels, _, train_approach2_labels, train_approach_3_task_2_labels = preprocess_data(train_data, state['word2idx'], state['max_len'], state['num_entities'], state['PAD'])

    # Train the models
    if 1 in models_to_train:
        print("Approach 1: Training mask prediction model")
        approach1_mask_prediction = Approach1MaskPrediction(state['vocab_dim'], state['emb_dim'])
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(approach1_mask_prediction.parameters(), lr=state['lr'])

        for epoch in range(state['num_epochs']):
            for i in range(0, len(train_sentence_feats), state['batch_size']):
                batch_feats = train_sentence_feats[i:i+state['batch_size']]
                batch_labels = train_mask_labels[i:i+state['batch_size']]
                optimizer.zero_grad()
                y_pred = approach1_mask_prediction(batch_feats)
                loss = criterion(y_pred, batch_labels)
                loss.backward()
                optimizer.step()
            print("Epoch: {}/{}...".format(epoch+1, state['num_epochs']),
                    "Loss: {:.6f}...".format(loss.item()))
            
        # Save the model
        if state['save']:
            torch.save(approach1_mask_prediction.state_dict(), 'models/approach1_mask_prediction.pt')
            
        print("Approach 1: Training entity classification model")
        approach1_entity_classification = Approach1EntityClassification(state['vocab_dim'], state['emb_dim'], state['num_entities'])
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(approach1_entity_classification.parameters(), lr=state['lr'])

        for epoch in range(state['num_epochs']+50):
            for i in range(0, len(train_named_entity_sentence_feats), state['batch_size']):
                batch_feats = train_named_entity_sentence_feats[i:i+state['batch_size']]
                batch_labels = train_named_entity_data_labels[i:i+state['batch_size']]
                optimizer.zero_grad()
                y_pred = approach1_entity_classification(batch_feats)
                loss = criterion(y_pred, batch_labels)
                loss.backward()
                optimizer.step()

            print("Epoch: {}/{}...".format(epoch+1, state['num_epochs']),
                "Loss: {:.6f}...".format(loss.item()))
            
        # Save the model
        if state['save']:
            torch.save(approach1_entity_classification.state_dict(), 'models/approach1_entity_classification.pt')

    if 2 in models_to_train:
        print("Approach 2: Training model")
        approach2model = Approach2(state['vocab_dim'], state['emb_dim'], state['num_entities'])
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(approach2model.parameters(), lr=state['lr'])

        for epoch in range(state['num_epochs']):
            for i in range(0, len(train_sentence_feats), state['batch_size']):        
                batch_feats = train_sentence_feats[i:i+state['batch_size']]
                batch_labels = train_approach2_labels[i:i+state['batch_size']]    
                optimizer.zero_grad()
                y_pred = approach2model(batch_feats)
                loss = criterion(y_pred, batch_labels)
                loss.backward()
                optimizer.step()
            print("Epoch: {}/{}...".format(epoch+1, state['num_epochs']),
                "Loss: {:.6f}...".format(loss.item()))
            
        # Save the model
        if state['save']:
            torch.save(approach2model.state_dict(), 'models/approach2.pt')
            
    if 3 in models_to_train:
        print("Approach 3: Training model")
        combined_model = Approach3CombinedModel(state['vocab_dim'], state['emb_dim'], state['num_entities'])

        criterion1 = nn.BCELoss()
        criterion2 = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(combined_model.parameters(), lr=state['lr'])

        for epoch in range(state['num_epochs']):
            for i in range(0, len(train_sentence_feats), state['batch_size']):
                batch_feats = train_sentence_feats[i:i+state['batch_size']]
                batch_labels_1 = train_mask_labels[i:i+state['batch_size']] # Labels if mask is a named entity or not
                batch_labels_2 = train_approach_3_task_2_labels[i:i+state['batch_size']] # Labels for what type of named entity the mask is
                optimizer.zero_grad()
                y_pred1, y_pred2 = combined_model(batch_feats)        
                loss1 = criterion1(y_pred1, batch_labels_1)        
                loss2 = criterion2(y_pred2, batch_labels_2)
                loss = loss1 + loss2
                loss.backward()
                optimizer.step()

            print("Epoch: {}/{}...".format(epoch+1, state['num_epochs']),
                    "Loss: {:.6f}...".format(loss.item()))
            
        # Save the model
        if state['save']:
            torch.save(combined_model.state_dict(), 'models/approach3.pt')

def evaluate(models_to_evaluate, test_data: list, state: dict):
    # Check if anything is stored in the models directory
    if not os.listdir('models'):
        print('No models found in models directory.')
        return
    
    test_sentence_feats, test_mask_labels, _, _, approach1_model_2_test_data, test_approach2_labels, test_approach_3_task_2_labels = preprocess_data(test_data, state['word2idx'], state['max_len'], state['num_entities'], state['PAD'])


    if 1 in models_to_evaluate:
        print("Approach 1: Evaluating binary entity model")
        # Load models for approach 1
        approach1_mask_prediction = Approach1MaskPrediction(state['vocab_dim'], state['emb_dim'])
        approach1_mask_prediction.load_state_dict(torch.load('models/approach1_mask_prediction.pt'))

        approach1_entity_classification = Approach1EntityClassification(state['vocab_dim'], state['emb_dim'], state['num_entities'])
        approach1_entity_classification.load_state_dict(torch.load('models/approach1_entity_classification.pt'))

        # Evaluate performance
        approach1_mask_prediction.eval()
        with torch.no_grad():
            y_pred = approach1_mask_prediction(test_sentence_feats)
            threshold = 0.5
            y_pred = (y_pred > threshold).float() # Convert probabilities to binary predictions
            y_test = test_mask_labels
            print("Accuracy of mask prediction model", accuracy_score(y_test, y_pred))
            print("F1 score of mask prediction model", f1_score(y_test, y_pred, average='macro'))

            # Convert predictions and gold labels tensors to lists of integers
            preds = y_pred.tolist()
            golds= y_test.tolist()
        
        pred_list = [label_list[int(pred[0])] for pred in preds]
        gold_list = [label_list[int(gold[0])] for gold in golds]
    
        #save predictions and gold labels to csv using pandas
        np.savetxt('predictions/approach1_pred_binary.csv', pred_list, delimiter=',', fmt='%s')
        np.savetxt('predictions/approach1_gold_binary.csv', gold_list, delimiter=',', fmt='%s')


        print("Approach 1: Evaluating entity classification model")
        # Only run Approach1EntityClassification if y_pred is 1
        approach1_entity_classification.eval()
        with torch.no_grad():
            # Get indices where y_pred is 1
            indices = torch.nonzero(y_pred).squeeze(1)
            indices = indices[:, 0]
            if indices.shape[0] == 0:
                print("No entities found in test data")
                return
            y_pred = approach1_entity_classification(test_sentence_feats[indices])
            y_pred = torch.argmax(y_pred, axis=1)
            y_test = torch.argmax(approach1_model_2_test_data[indices], axis=1)
            print("Accuracy of entity classification model", accuracy_score(y_test, y_pred))
            print("F1 score of entity classification model", f1_score(y_test, y_pred, average='macro'))

            # Convert predictions and gold labels tensors to lists of integers
            preds = y_pred.tolist()
            golds = y_test.tolist()
        
        pred_list = [label_list[int(pred)] for pred in preds]
        gold_list = [label_list[int(gold)] for gold in golds]
    
        #save predictions and gold labels to csv using pandas
        np.savetxt('predictions/approach1_pred_multiclass.csv', pred_list, delimiter=',', fmt='%s')
        np.savetxt('predictions/approach1_gold_multiclass.csv', gold_list, delimiter=',', fmt='%s')


    if 2 in models_to_evaluate:
        print("Approach 2: Evaluating model")
        approach2model = Approach2(state['vocab_dim'], state['emb_dim'], state['num_entities'])
        approach2model.load_state_dict(torch.load('models/approach2.pt'))
        
        # Evaluate
        approach2model.eval()

        with torch.no_grad():
            y_pred = approach2model(test_sentence_feats)
            y_pred = torch.argmax(y_pred, axis=1)
            y_test = torch.argmax(test_approach2_labels, axis=1)
            print("Accuracy of approach 2 model", accuracy_score(y_test, y_pred))
            print("F1 score of approach 2 model", f1_score(y_test, y_pred, average='macro'))

    
    if 3 in models_to_evaluate:
        print("Approach 3: Evaluating model")
        combined_model = Approach3CombinedModel(state['vocab_dim'], state['emb_dim'], state['num_entities'])
        combined_model.load_state_dict(torch.load('models/approach3.pt'))

        # Evaluate
        combined_model.eval()

        with torch.no_grad():
            y_pred1, y_pred2 = combined_model(test_sentence_feats)
            y_pred1 = torch.round(y_pred1)
            y_pred2 = torch.argmax(y_pred2, axis=1)
            y_test1 = test_mask_labels
            y_test2 = torch.argmax(test_approach_3_task_2_labels, axis=1)
            print("Accuracy of approach 3 model 1", accuracy_score(y_test1, y_pred1))
            print("F1 score of approach 3 model 1", f1_score(y_test1, y_pred1, average='macro'))
            print("Accuracy of approach 3 model 2", accuracy_score(y_test2, y_pred2))
            print("F1 score of approach 3 model 2", f1_score(y_test2, y_pred2, average='macro'))

if __name__ == '__main__':
    '''
    Usage:
    To train all models and save them to disk
    `python main.py --train`
    To train only approach 1 and 2 without saving them
    `python main.py --train --approach-1 --approach-2 --save False`

    To test all models from disk. Remember to train them first
    `python main.py --test`
    To evaluate only approach 1 and 2
    `python main.py --test --approach-1 --approach-2`
    '''
    main()