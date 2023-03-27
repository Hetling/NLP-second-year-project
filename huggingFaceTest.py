#test the bert ner model from huggingface
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
from helpers import load_conll
import pandas as pd
tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
#model_large = AutoModelForTokenClassification.from_pretrained("dslim/bert-large-NER")
#model_lowercase = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER-uncased")

lowercased = False

# make pipeline return all scores
nlp = pipeline("ner", model=model, tokenizer=tokenizer, ignore_labels=[])


example = ['My', 'name', 'is', 'Wolfgang', 'and', 'I', 'live', 'in', 'Iguazu', '.', 'My', 'name', 'is', 'Obama']

#make example into Union data type
example = ' '.join(example)


def predict(example, lowercased):
    string = ' '.join(example) #needs to be joined to a string

    if lowercased:
        string = string.lower()

    ner_results = nlp(string)
    #combine tokens back to original sentence
    sent = ([],[])
    word = ''
    for result in ner_results:
        #if token is part of a word add it to earlier token
        if result['word'].startswith('##'): 
           word = word+result['word'][2:]
        #if token is a new word add previous word to sentence and start new word
        else: 
            if word != '':
                sent[0].append(word)
                
            word = result['word'] #this gets saved with a delay to account for adding ## words
            sent[1].append(result['entity'])
    sent[0].append(word) #that's why this has to be added as the very last thing
        
    return sent

# #load test data
test = load_conll('data/en_ewt_nn_answers_test.conll')

results = [predict(sent[0], lowercased) for sent in test]

if lowercased:
    with open('base_preds/bert-lower-test.conll', 'w') as f:
        for sent in results:
            for word, tag in zip(sent[0], sent[1]):
                f.write(word + '\t' + tag + '\n')
            f.write('\n')

else: 
    #write to conll file
    with open('base_preds/bert-test.conll', 'w') as f:
        for sent in results:
            for word, tag in zip(sent[0], sent[1]):
                f.write(word + '\t' + tag + '\n')
            f.write('\n')

    #classify evertyhing as being a person
    with open('base_preds/null-test.conll', 'w') as f:
        for sent in test:
            for word in sent[0]:
                f.write(word + '\t' + 'B-PER' + '\n')
            f.write('\n')

