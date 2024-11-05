import pandas as pd

application_path = 'data/application_record.csv'
credit_path = 'data/credit_record.csv'

# Load and Clean application_record.csv
application_df = pd.read_csv(application_path)

# Check for missing values and drop columns with high missingness
missing_data = application_df.isnull().sum()
high_missing_cols = missing_data[missing_data > len(application_df) * 0.4].index.tolist()
application_df.drop(columns=high_missing_cols, inplace=True)

# Drop rows with any remaining missing values
application_df.dropna(inplace=True)

# Handle high cardinality columns (more than 100 unique values)
high_cardinality_cols = [col for col in application_df.select_dtypes(include=['object', 'category']).columns 
                         if application_df[col].nunique() > 100]
application_df.drop(columns=high_cardinality_cols, inplace=True)

# Convert days columns to positive
application_df['DAYS_BIRTH'] = application_df['DAYS_BIRTH'].abs()
application_df['DAYS_EMPLOYED'] = application_df['DAYS_EMPLOYED'].abs()

# Save cleaned application data to CSV
application_df.to_csv('data/cleaned_application_record.csv', index=False)

# Load and Clean credit_record.csv in chunks
credit_cleaned = []
chunk_size = 100000

for chunk in pd.read_csv(credit_path, chunksize=chunk_size):
    chunk.dropna(inplace=True)
    credit_cleaned.append(chunk)

# Concatenate and save cleaned credit data to CSV
credit_df = pd.concat(credit_cleaned, ignore_index=True)
credit_df.to_csv('data/cleaned_credit_record.csv', index=False)
