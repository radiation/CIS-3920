import pandas as pd
import statsmodels.api as sm 
import sys

pd.set_option('display.max_columns', 500)

# I/O files
mydata = pd.read_csv("advertising.csv")
file = open('output.txt','wt')
sys.stdout = file

# What we're working with
print(mydata.describe())

df = pd.DataFrame(mydata)

# Convert 'Timestamp' to datetime
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# Convert 'Timestamp' to seconds since Unix epoch
df['Timestamp'] = (df['Timestamp'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')

# Simple Regression
x = df['Male']
y = df['ClickedonAd']
x = sm.add_constant(x)
model1 = sm.Logit(y, x).fit()
results_model1 = model1.summary()
print(results_model1)

# Multiple Regression with Timestamp converted
x = df[['DailyTimeSpentonSite','Age','AreaIncome','DailyInternetUsage','Male','Timestamp']]
y = df['ClickedonAd']
x = sm.add_constant(x)
model2 = sm.Logit(y, x).fit()
results_model2 = model2.summary()
print(results_model2)

# Close the output file
file.close()
