import pandas as pd
import numpy as np
import os
import re

def extract_filename_from_path(file_path):
    filename_with_extension = os.path.basename(file_path)
    filename = os.path.splitext(filename_with_extension)[0]
    df = pd.read_csv(file_path)
    return df, filename

def extract_meta(df, table_name):
    metadata = []
    
    for column in df.columns:
        column_data = {
            "Table Name": table_name,
            "Column Name": column,
            "Data Type": str(df[column].dtype),
            "Sample Data": df[column].dropna().head(5).tolist(),
            "Missing Values": df[column].isnull().sum(),
            "Unique Values": df[column].nunique(),
            "Value Range (Min/Max)": (df[column].min(), df[column].max()) if pd.api.types.is_numeric_dtype(df[column]) else None,
            "Unit of Measurement": None,
            "Categorical Values": None,
            "Textual Description": None,
            "Date Granularity": None,
            "Time Format": None,
            "Ontology Mapping": None
        }
        # Check if the column is categorical
        if pd.api.types.is_categorical_dtype(df[column]):
            column_data["Categorical Values"] = df[column].cat.categories.tolist()
        # Check if the column contains datetime data
        if pd.api.types.is_object_dtype(df[column]):
            if all(isinstance(val, str) and re.match(r'^\d{4}-\d{2}-\d{2}$', val) for val in df[column].dropna()):
                column_data["Date Granularity"] = "YYYY-MM-DD"
            elif all(isinstance(val, str) and re.match(r'^\d{2}/\d{2}/\d{4}$', val) for val in df[column].dropna()):
                column_data["Date Granularity"] = "MM/DD/YYYY"
            elif all(isinstance(val, str) and re.match(r'^\d{1,2}:\d{2}:\d{2}$', val) for val in df[column].dropna()):
                column_data["Time Format"] = "h:m:s"
        if pd.api.types.is_datetime64_any_dtype(df[column]):
            column_data["Date Granularity"] = "Date"
        
        metadata.append(column_data)
    
    metadata_df = pd.DataFrame(metadata)
    return metadata_df