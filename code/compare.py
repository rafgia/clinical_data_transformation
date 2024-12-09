from fuzzywuzzy import fuzz
import pandas as pd

def compare_metadata(eICU_metadata, MIMIC_IV_metadata):
    mappings = []
    
    for idx_eICU, row_eICU in eICU_metadata.iterrows():
        best_match = None
        highest_score = 0
        for idx_MIMIC, row_MIMIC in MIMIC_IV_metadata.iterrows():
            score = 0
            # Fuzzy matching based on column names
            name_score = fuzz.token_sort_ratio(row_eICU['Column Name'], row_MIMIC['Column Name'])
            score += name_score
            # Compare data types
            if row_eICU['Data Type'] == row_MIMIC['Data Type']:
                score += 20  # Arbitrary score for matching data types
            # Compare missing values (columns with similar missing data rates)
            missing_diff = abs(row_eICU['Missing Values'] - row_MIMIC['Missing Values'])
            if missing_diff < 5:  # You can adjust this threshold
                score += 10
            # Compare categorical values (if they exist)
            if row_eICU['Categorical Values'] and row_MIMIC['Categorical Values']:
                common_categorical_values = set(row_eICU['Categorical Values']).intersection(set(row_MIMIC['Categorical Values']))
                if common_categorical_values:
                    score += 15  # Arbitrary score for common categorical values
            # If score is high enough, consider this a good match
            if score > highest_score:
                highest_score = score
                best_match = row_MIMIC['Column Name']
        mappings.append({
            "eICU Column": row_eICU['Column Name'],
            "MIMIC-IV Column": best_match,
            "Similarity Score": highest_score
        })
    mapping_df = pd.DataFrame(mappings)
    return mapping_df