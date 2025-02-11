#!pip install transformers datasets

import json
import random
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import Dataset

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def save_json(data, output_path):
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)

def split_json(file_path, train_file, test_file, train_ratio=0.9):
    data = load_json(file_path)

    random.shuffle(data)

    split_index = int(len(data) * train_ratio)

    # Split the data into training and testing sets
    train_data = data[:split_index]
    test_data = data[split_index:]

    save_json(train_data, train_file)
    save_json(test_data, test_file)

    print(f"Training data saved to {train_file}")
    print(f"Testing data saved to {test_file}")

input_file = '/content/training_qa_pairs.json'
train_file = 'Training_description.json'
test_file = 'Testing_description.json'

split_json(input_file, train_file, test_file)

#!unzip fine_tuned_model.zip -d fine_tuned_model/ #to re-finetune the model

def load_data(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

# Preprocess the data
def preprocess_data(data, tokenizer):
    inputs = []
    targets = []

    for item in data:
        inputs.append(item['question'])
        targets.append(item['answer'])

    encodings = tokenizer(inputs, padding=True, truncation=True, max_length=512)
    target_encodings = tokenizer(targets, padding=True, truncation=True, max_length=512)

    encodings['labels'] = target_encodings['input_ids']
    return Dataset.from_dict(encodings)

def main(train_file, test_file, output_dir='fine_tuned_model', model_dir='t5-small', batch_size=4, epochs=100):
    # Load tokenizer and model
    tokenizer = T5Tokenizer.from_pretrained(model_dir)
    model = T5ForConditionalGeneration.from_pretrained(model_dir)

    # Load and preprocess training and test data
    train_data = load_data(train_file)
    test_data = load_data(test_file)

    train_dataset = preprocess_data(train_data, tokenizer)
    test_dataset = preprocess_data(test_data, tokenizer)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="steps",  
        save_steps=100,               # Evaluate and save every 100 steps
        eval_steps=100,               
        learning_rate=5e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        save_total_limit=3,
        logging_dir=f'{output_dir}/logs',
        logging_steps=10,
        load_best_model_at_end=True,   # Will load the best model at the end of training
        fp16 = True
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
    )

    # Fine-tune
    trainer.train()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")

train_file = "/content/Training_description.json"  # Path to the training file
test_file = "/content/Testing_description.json"    # Path to the testing file
output_dir = "/content/new_fine_tuned_model"
model_dir = "t5-small"  # Replace this with the fine-tuned model if a fine-tuning is done again
batch_size = 4
epochs = 100

main(train_file, test_file, output_dir=output_dir, model_dir=model_dir, batch_size=batch_size, epochs=epochs)