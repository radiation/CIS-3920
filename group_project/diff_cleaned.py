import pandas as pd

# Define paths to your files
application_path = 'data/application_record.csv'
credit_path = 'data/credit_record.csv'

# Step 1: Load and Clean application_record.csv
application_df = pd.read_csv(application_path)

# Identify rows with missing data in application_record.csv
removed_application_rows = application_df[application_df.isnull().any(axis=1)]
# Save these rows to a new CSV file for reference
removed_application_rows.to_csv('data/removed_application_record.csv', index=False)

# Continue with cleaning by dropping rows with missing data
application_df.dropna(inplace=True)

# Additional cleaning steps...
# (e.g., high cardinality columns or days columns conversion as before)
# ...

# Save cleaned data to a CSV file
application_df.to_csv('data/cleaned_application_record.csv', index=False)


# Step 2: Load and Clean credit_record.csv in Chunks
credit_cleaned = []
removed_credit_rows = []  # List to store chunks of removed rows
chunk_size = 100000

for chunk in pd.read_csv(credit_path, chunksize=chunk_size):
    # Identify rows with missing data in this chunk
    removed_rows_chunk = chunk[chunk.isnull().any(axis=1)]
    removed_credit_rows.append(removed_rows_chunk)  # Collect these rows
    
    # Drop rows with missing data from the current chunk and add cleaned data
    chunk.dropna(inplace=True)
    credit_cleaned.append(chunk)

# Concatenate and save removed rows for credit data to a separate CSV
removed_credit_df = pd.concat(removed_credit_rows, ignore_index=True)
removed_credit_df.to_csv('data/removed_credit_record.csv', index=False)

# Concatenate and save cleaned credit data to a CSV file
credit_df = pd.concat(credit_cleaned, ignore_index=True)
credit_df.to_csv('data/cleaned_credit_record.csv', index=False)
