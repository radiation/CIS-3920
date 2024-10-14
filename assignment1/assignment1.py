import pandas as pd
import statsmodels.api as sm
import statsmodels.stats.api as sms
from statsmodels.compat import lzip
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro
import sys

# Display settings for pandas: expand the column display limit for better readability
pd.set_option('display.max_columns', 500)

# Load the insurance dataset from a CSV file
house_data: pd.DataFrame = pd.read_csv("kc_house_data.csv")

# Redirect standard output to a file for logging results
output_file = open('output.txt', 'wt')
sys.stdout = output_file

# Display basic statistics for the dataset (useful for initial exploration)
print(house_data.describe())

def run_regression(x: pd.Series, y: pd.Series) -> None:
    # Add a constant (intercept) to the independent variable
    x_with_constant: pd.DataFrame = sm.add_constant(x)

    # Fit the OLS regression model
    model: sm.regression.linear_model.RegressionResultsWrapper = sm.OLS(y, x_with_constant).fit()

    # Print the summary of the regression model
    print(model.summary())

    # Store the predicted charges in a new column based on the first model
    df["predicted1"] = model1.predict(x_with_constant)

    # Plot actual charges vs. predicted charges for the first model
    sns.scatterplot(data=df, x="charges", y="predicted1")
    plt.savefig("charges_vs_predicted1.png")  # Save the plot to the filesystem
    plt.clf()  # Clear the current figure for subsequent plots

    # Define predictor variables for the second model (multiple regression)
    x_multivariate: pd.DataFrame = df[['age', 'male', 'bmi', 'children', 'smoker']]

    # Add a constant to the independent variables for the second model
    x_multivariate_with_constant: pd.DataFrame = sm.add_constant(x_multivariate)

    # Fit the second OLS regression model using multiple predictors
    model2 = sm.OLS(y, x_multivariate_with_constant).fit()

    # Print the summary of the second regression model
    results_model2 = model2.summary()
    print(results_model2)

    # Store the predicted charges in a new column based on the second model
    df["predicted2"] = model2.predict(x_multivariate_with_constant)

    # Plot actual charges vs. predicted charges for the second model
    sns.scatterplot(data=df, x="charges", y="predicted2")
    plt.savefig("charges_vs_predicted2.png")  # Save the plot to the filesystem
    plt.clf()  # Clear the current figure for subsequent plots

# Create a DataFrame and define predictor (x) and response (y) variables for the first model
df: pd.DataFrame = pd.DataFrame(house_data)
df['sqft'] = df['sqft_above'] * df['sqft_basement']
df['age_sqft'] = df['AgeOfHouse'] * df['sqft']
print(df.describe())
x: pd.Series = df['AgeOfHouse']
y: pd.Series = df['price']

# Add a constant (intercept) to the independent variable (BMI)
x_with_constant: pd.DataFrame = sm.add_constant(x)

# Fit the first OLS regression model using BMI as the predictor
model1: sm.regression.linear_model.RegressionResultsWrapper = sm.OLS(y, x_with_constant).fit()

# Print the summary of the first regression model
results_model1: str = model1.summary()
print(results_model1)

# Store the predicted charges in a new column based on the first model
df["predicted1"] = model1.predict(x_with_constant)

# Plot actual x vs. x charges for the first model
sns.scatterplot(data=df, x="AgeOfHouse", y="predicted1")
plt.savefig("AgeOfHouse_vs_predicted1.png")  # Save the plot to the filesystem
plt.clf()  # Clear the current figure for subsequent plots

# Define predictor variables for the second model (multiple regression)
x_multivariate: pd.DataFrame = df[['AgeOfHouse','sqft','age_sqft']]

# Add a constant to the independent variables for the second model
x_multivariate_with_constant: pd.DataFrame = sm.add_constant(x_multivariate)

# Fit the second OLS regression model using multiple predictors
model2 = sm.OLS(y, x_multivariate_with_constant).fit()

# Print the summary of the second regression model
results_model2 = model2.summary()
print(results_model2)

# Store the predicted x in a new column based on the second model
df["predicted2"] = model2.predict(x_multivariate_with_constant)

# Plot actual x vs. predicted x for the second model
sns.scatterplot(data=df, x="AgeOfHouse", y="predicted2")
plt.savefig("AgeOfHouse_vs_predicted2.png")  # Save the plot to the filesystem
plt.clf()  # Clear the current figure for subsequent plots

# Diagnostic 1: Multicollinearity check (correlation matrix)
corr_matrix: pd.DataFrame = df.corr()
print(corr_matrix)

# Diagnostic 2: Test for heteroscedasticity using the Breusch-Pagan test
names: list[str] = ['Lagrange multiplier statistic', 'p-value', 'f-value', 'f p-value']
test = sms.het_breuschpagan(model2.resid, model2.model.exog)
print(lzip(names, test))

# Display basic statistics for the dataset again (optional)
print(house_data.describe())

# Perform a normality test (Shapiro-Wilk test) on the residuals of the second model
residuals = model2.resid
print(shapiro(residuals))

# Close the output file (end of logging)
output_file.close()
