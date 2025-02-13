#  pip install transformers datasets torch scikit-learn 

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    DistilBertForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import torch
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import numpy as np

# Constants
MODEL_NAME = "distilbert-base-uncased"
OUTPUT_DIR = "./fine_tuned_model"
TRAIN_SAMPLES = 1000
EVAL_SAMPLES = 200
TEST_SAMPLES = 100
SEED = 42

# Load IMDb dataset
def load_imdb_dataset():
    dataset = load_dataset("imdb")
    train_dataset = dataset["train"].shuffle(seed=SEED).select(range(TRAIN_SAMPLES))
    eval_dataset = dataset["test"].shuffle(seed=SEED).select(range(EVAL_SAMPLES))
    test_dataset = dataset["test"].shuffle(seed=SEED).select(range(TEST_SAMPLES))
    return train_dataset, eval_dataset, test_dataset

# Tokenize dataset
def tokenize_dataset(dataset, tokenizer):
    return dataset.map(
        lambda examples: tokenizer(examples["text"], padding="max_length", truncation=True),
        batched=True,
        num_proc=4,
    ).with_format("torch")

# Fine-tune the model
def fine_tune_model(train_dataset, eval_dataset, output_dir):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenized_train = tokenize_dataset(train_dataset, tokenizer)
    tokenized_eval = tokenize_dataset(eval_dataset, tokenizer)

    model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    training_args = TrainingArguments(
        output_dir=output_dir,          # Directory to save model checkpoints
        evaluation_strategy="epoch",    # Evaluate after each epoch
        learning_rate=2e-5,             # Learning rate
        per_device_train_batch_size=32, # Batch size for training (larger for GPU)
        per_device_eval_batch_size=32,  # Batch size for evaluation
        num_train_epochs=10,            # Number of epochs
        fp16=True,                      # Enable mixed precision (FP16) for faster training
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
    )

    trainer.train()
    eval_results = trainer.evaluate()
    print(f"Evaluation results: {eval_results}")

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")

# Load the fine-tuned model
def load_model(model_dir):
    model = DistilBertForSequenceClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    print("Model and tokenizer loaded successfully!")
    return model, tokenizer

# Evaluate the model
def predict(model, tokenizer, test_dataset):
    tokenized_test = tokenize_dataset(test_dataset, tokenizer)
    trainer = Trainer(model=model)
    predictions = trainer.predict(tokenized_test)

    predicted_labels = torch.argmax(torch.tensor(predictions.predictions), dim=-1)
    true_labels = np.array([example["label"] for example in test_dataset])

    accuracy = accuracy_score(true_labels, predicted_labels)
    cr = classification_report(true_labels, predicted_labels, target_names=["Negative", "Positive"])

    print(f"Classification Accuracy: {accuracy:.4f}")
    print("Classification Report:\n", cr) 
    

train_dataset, eval_dataset, test_dataset = load_imdb_dataset()
fine_tune_model(train_dataset, eval_dataset, OUTPUT_DIR)  # commanet out after training model
model, tokenizer = load_model(OUTPUT_DIR)
evaluate_model(model, tokenizer, test_dataset)
    
    