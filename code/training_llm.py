import json
import pandas as pd
import os

def extract_filename_from_path(file_path):
    filename_with_extension = os.path.basename(file_path)
    filename = os.path.splitext(filename_with_extension)[0]
    df = pd.read_csv(file_path)
    return df, filename

def create_qa_pair_with_context(df, table_name):
    data_type = str(df[column].dtype),
    sample_data = df[column].dropna().head(5).tolist()
    context = f"You are working with the '{table_name}' table from the eICU database. Here are some sample rows for the column '{column}': {sample_data}. The type of the data is '{data_type}'"

    question = f"What does the '{column}' column represent?"
    
    print("\nContext:", context)
    print("Question:", question)
    answer = input("Please provide the answer for this question: ")
        
    
    return {"input_text": context + " " + question, "answer": answer}

def save_qa_pairs_to_json(qa_pairs, filename):
    if os.path.exists(filename):
        with open(filename, 'r') as json_file:
            existing_data = json.load(json_file)
        existing_data.extend(qa_pairs)
        
        with open(filename, 'w') as json_file:
            json.dump(existing_data, json_file, indent=4)
        print(f"\nNew QA pairs appended to {filename}")
    else:
        with open(filename, 'w') as json_file:
            json.dump(qa_pairs, json_file, indent=4)
        print(f"\nQA pairs saved to {filename}")