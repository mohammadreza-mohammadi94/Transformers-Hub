import os
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, AutoConfig, Trainer, TrainingArguments
)
from datasets import Dataset, DatasetDict
import evaluate

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Constants
MODEL_CKPT = "bert-base-uncased"
BATCH_SIZE = 64
TRAINING_DIR = "bert_base_training_dir"

def load_data():
    """Load the dataset from a remote CSV file."""
    df = pd.read_csv("https://raw.githubusercontent.com/laxmimerit/All-CSV-ML-Data-Files-Download/master/twitter_multi_class_sentiment.csv")
    return df

def preprocess_data(df):
    """Perform basic text preprocessing and feature engineering."""
    df['words_per_tweet'] = df["text"].str.split().apply(len)
    return df

def plot_label_distribution(df):
    """Plot the distribution of sentiment labels."""
    df['label_name'].value_counts(ascending=True).plot.barh()
    plt.title("Label Frequencies", fontsize=12, fontweight="bold")
    plt.show()

def plot_words_per_tweet(df):
    """Plot a boxplot showing the number of words per tweet for each sentiment label."""
    df.boxplot("words_per_tweet", by="label_name")
    plt.title("Words per Tweet Distribution", fontsize=12, fontweight="bold")
    plt.show()

def split_dataset(df):
    """Split the dataset into train, test, and validation sets while maintaining label balance."""
    train, test = train_test_split(df, test_size=0.3, stratify=df['label_name'])
    test, validation = train_test_split(test, test_size=0.3, stratify=test['label_name'])
    return train, test, validation

def create_dataset_dict(train, test, validation):
    """Convert Pandas DataFrames into Hugging Face DatasetDict format."""
    dataset = DatasetDict({
        "train": Dataset.from_pandas(train, preserve_index=False),
        "test": Dataset.from_pandas(test, preserve_index=False),
        "validation": Dataset.from_pandas(validation, preserve_index=False)
    })
    return dataset

def tokenize_dataset(dataset, tokenizer):
    """Apply tokenization to the dataset using the specified tokenizer."""
    def tokenize_fn(batch):
        return tokenizer(batch['text'], padding=True, truncation=True)
    return dataset.map(tokenize_fn, batched=True, batch_size=None)

def build_label_mappings(dataset):
    """Create mappings from label names to IDs and vice versa."""
    label2id = {x['label_name']: x['label'] for x in dataset['train']}
    id2label = {v: k for k, v in label2id.items()}
    return label2id, id2label

def load_model(label2id, id2label):
    """Load a pre-trained BERT model for sequence classification."""
    config = AutoConfig.from_pretrained(MODEL_CKPT, label2id=label2id, id2label=id2label)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_CKPT, config=config).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return model

def compute_metrics(pred):
    """Compute evaluation metrics (accuracy and weighted F1-score)."""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted")
    }

def train_model(model, training_args, tokenized_dataset, tokenizer):
    """Train the model using the Hugging Face Trainer API."""
    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['validation'],
        tokenizer=tokenizer
    )
    trainer.train()
    return trainer

def evaluate_model(trainer, tokenized_dataset):
    """Evaluate the model's performance on the test dataset."""
    preds_output = trainer.predict(tokenized_dataset['test'])
    y_pred = np.argmax(preds_output.predictions, axis=1)
    y_true = tokenized_dataset['test'][:]['label']
    print(classification_report(y_true, y_pred))
    return y_true, y_pred

def plot_confusion_matrix(y_true, y_pred, label2id):
    """Plot the confusion matrix to visualize model performance."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, xticklabels=label2id.keys(), yticklabels=label2id.keys(), fmt='d', cbar=False, cmap='Reds')
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.show()

def get_prediction(text, model, tokenizer, id2label):
    """Generate sentiment predictions for a given input text."""
    input_encoded = tokenizer(text, return_tensors='pt').to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    with torch.no_grad():
        outputs = model(**input_encoded)
    pred = torch.argmax(outputs.logits, dim=1).item()
    return id2label[pred]

def main():
    """Execute all steps in sequence."""
    df = load_data()
    df = preprocess_data(df)
    plot_label_distribution(df)
    plot_words_per_tweet(df)
    
    train, test, validation = split_dataset(df)
    dataset = create_dataset_dict(train, test, validation)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CKPT)
    tokenized_dataset = tokenize_dataset(dataset, tokenizer)
    label2id, id2label = build_label_mappings(dataset)
    
    model = load_model(label2id, id2label)
    training_args = TrainingArguments(
        output_dir=TRAINING_DIR,
        overwrite_output_dir=True,
        num_train_epochs=2,
        learning_rate=2e-5,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        weight_decay=0.01,
        evaluation_strategy='epoch',
        disable_tqdm=False,
        report_to='none'
    )
    trainer = train_model(model, training_args, tokenized_dataset, tokenizer)
    
    y_true, y_pred = evaluate_model(trainer, tokenized_dataset)
    plot_confusion_matrix(y_true, y_pred, label2id)
    
    text_sample = "I am super happy today!"
    print(f"Prediction: {get_prediction(text_sample, model, tokenizer, id2label)}")

if __name__ == "__main__":
    main()