from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorForTokenClassification
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
import numpy as np
from datasets import load_metric

wnut = load_dataset('wnut_17')
label_list=wnut["train"].features[f"ner_tags"].feature.names
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples[f"ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenized_wnut = wnut.map(tokenize_and_align_labels, batched=True)
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer) #adds padding after tokenizing
model = AutoModelForTokenClassification.from_pretrained("distilbert-base-uncased", num_labels=14) #-100 is an extra label

# accuracy = load_metric("accuracy")
# metric = load_metric("f1")
metric = load_metric("seqeval")

def compute_metrics(df):
    logits, labels = df
    predictions = np.argmax(logits, axis=-1)

    # #flatten predictions and labels
    # flat_predictions = np.ravel(predictions)
    # flat_labels = np.ravel(labels)
    # #save flat predictions and flat labels to csv
    # np.savetxt("flat_predictions.csv", flat_predictions, delimiter=",")
    # np.savetxt("flat_labels.csv", flat_labels, delimiter=",")

    # #only choose indexes that are not -100
    # filtered_predictions = flat_predictions[flat_labels != -100]
    # filtered_labels = flat_labels[flat_labels != -100]
    # #save filtered predictions and filtered labels to csv
    # np.savetxt("filtered_predictions.csv", filtered_predictions, delimiter=",")
    # np.savetxt("filtered_labels.csv", filtered_labels, delimiter=",")

    # f1 = metric.compute(predictions=filtered_predictions, references=filtered_labels, average='macro')
    # accuracy_score = accuracy.compute(predictions=filtered_predictions, references=filtered_labels)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels, zero_division=1) #default classes predicted 0 times to 1

    return {"precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],}


training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_wnut["train"],
    eval_dataset=tokenized_wnut["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.evaluate()

# # SPAN F1 EVALUATION

# filtered_labels = np.loadtxt("filtered_labels.csv", delimiter=",")
# filtered_predictions = np.loadtxt("filtered_predictions.csv", delimiter=",")

# #map using label_list
# filtered_labels = np.vectorize(label_list.__getitem__)(filtered_labels.astype(int))
# filtered_predictions = np.vectorize(label_list.__getitem__)(filtered_predictions.astype(int))

# #save filtered predictions and filtered labels to csv
# np.savetxt("filtered_predictions_mapped.csv", filtered_predictions, delimiter=",", fmt="%s")
# np.savetxt("filtered_labels_mapped.csv", filtered_labels, delimiter=",", fmt="%s")