def create_column_mapping_dict(column_mapping_df):
    column_mapping_dict = {}
    for _, row in column_mapping_df.iterrows():
        eICU_column = row['eICU Column']
        MIMIC_IV_column = row['MIMIC-IV Column']
        column_mapping_dict[eICU_column] = MIMIC_IV_column
    return column_mapping_dict

#usage
column_mapping_dict = create_column_mapping_dict(column_mapping)
for eICU_col, MIMIC_IV_col in column_mapping_dict.items():
    print(f"eICU Column: {eICU_col} -> MIMIC-IV Column: {MIMIC_IV_col}")