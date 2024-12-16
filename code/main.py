qa_pairs = []
while True:
    file_path = input("Enter the file path: ")
    table, name_table = extract_filename_from_path(file_path)
    for column in table.columns:
        qa_pair = create_qa_pair_with_context(table, name_table)
        qa_pairs.append(qa_pair)
    
    continue_choice = input("Do you want to add another QA pair? (yes/no): ").lower()
    if continue_choice != 'yes':
        break
print(qa_pairs)
save_qa_pairs_to_json(qa_pairs, "Training_eICU.json")