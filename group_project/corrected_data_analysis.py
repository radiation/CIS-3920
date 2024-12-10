
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm

# Load the datasets
application_df = pd.read_csv('data/cleaned_application_record.csv')
credit_df = pd.read_csv('data/cleaned_credit_record.csv')

# Clean application_df (drop duplicates and handle missing values)
application_df = application_df.drop_duplicates(subset='ID')
application_df['CODE_GENDER'] = application_df['CODE_GENDER'].replace('XNA', None)
application_df = application_df.dropna(subset=['CODE_GENDER'])

# Clean credit_df (remove irrelevant columns)
credit_df = credit_df.dropna(subset=['STATUS'])

# Aggregate credit status to determine if an individual is 'Approved' or 'Rejected'
def determine_credit_status(status_series):
    if any(s in ['2', '3', '4', '5'] for s in status_series):
        return 'Rejected'
    else:
        return 'Approved'

credit_summary = credit_df.groupby('ID')['STATUS'].apply(determine_credit_status).reset_index()
credit_summary.rename(columns={'STATUS': 'Credit_Status'}, inplace=True)

# Merge the datasets on 'ID'
merged_df = pd.merge(application_df, credit_summary, on='ID', how='inner')

# Add Age column (convert DAYS_BIRTH to years)
merged_df['AGE'] = (-merged_df['DAYS_BIRTH']) // 365

# Hypothesis Testing: Chi-squared test for Credit Status vs Gender
contingency_table = pd.crosstab(merged_df['CODE_GENDER'], merged_df['Credit_Status'])
chi2, p, dof, expected = stats.chi2_contingency(contingency_table)

print("Chi-squared Test Results:")
print(f"Chi-squared value: {chi2}")
print(f"p-value: {p}")
print(f"Degrees of Freedom: {dof}")
print("Contingency Table:")
print(contingency_table)

# Logistic Regression: Predicting Credit Status
# Convert Credit_Status to binary (1 for Rejected, 0 for Approved)
merged_df['Credit_Status_Binary'] = (merged_df['Credit_Status'] == 'Rejected').astype(int)

# Select independent variables
X = merged_df[['CODE_GENDER', 'AMT_INCOME_TOTAL', 'AGE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS']]
X = pd.get_dummies(X, drop_first=True)  # Encode categorical variables
y = merged_df['Credit_Status_Binary']

# Debugging: Ensure all columns in X and y are numeric
print("Data types before conversion:")
print(X.dtypes)
print(y.dtypes)

print("First rows of X:")
print(X.head())
print("First rows of y:")
print(y.head())

# Convert to numeric and drop rows with NaNs
print("Converting to numeric and dropping NaNs...")
X = X.apply(pd.to_numeric, errors='coerce')
y = pd.to_numeric(y, errors='coerce')
X = X.dropna()
y = y.loc[X.index]  # Align y with the cleaned X

# Confirm matching indices
print("Matching indices after cleaning:")
print(f"X index: {X.index}")
print(f"y index: {y.index}")

# Check for residual object types
if any(X.dtypes == 'object'):
    raise ValueError("Residual object types found in X")
if y.dtype == 'object':
    raise ValueError("Residual object type found in y")

# Add a constant for the intercept
X = sm.add_constant(X)

# Recheck dtypes and unique values
print("Data types and unique values in X:")
for column in X.columns:
    print(f"{column}: dtype={X[column].dtype}, unique={X[column].unique()[:5]}")  # Show up to 5 unique values

print("\nDependent variable y:")
print(f"dtype={y.dtype}, unique={y.unique()}")

# Force boolean columns to integers
X = X.astype({col: 'int' for col in X.select_dtypes('bool').columns})

# Logistic regression model
logit_model = sm.Logit(y, X)
logit_results = logit_model.fit()

print("Logistic Regression Results:")
print(logit_results.summary())
