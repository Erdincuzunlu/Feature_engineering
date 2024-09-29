# Import necessary libraries
from operator import index
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

# Display settings for pandas
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.float_format", lambda x: "%.3f" % x)
pd.set_option("display.width", 500)

# Function to load the application_train.csv dataset
def load_application_train():
    data = pd.read_csv("Feature_Engineering/application_train.csv")
    return data

dff = load_application_train()
df.head()
df.info()

# Function to load the Titanic dataset
def load():
    data = pd.read_csv("Feature_Engineering/titanic.csv")
    return data

df = load()
df.head()

### 1. Catching outliers

###########################################
# Outliers with graphical techniques
############################################

# Visualizing the outliers in the 'Age' column using a boxplot
sns.boxplot(x=df["Age"])
plt.show()

### Identifying outliers with IQR

q1 = df["Age"].quantile(0.25)
q3 = df["Age"].quantile(0.75)
iqr = q3 - q1

up = q3 + 1.5 * iqr
low = q1 - 1.5 * iqr

# Displaying the outliers in 'Age'
df[(df["Age"] < low) | (df["Age"] > up)]

# Index of the outliers
df[(df["Age"] < low) | (df["Age"] > up)].index

# Checking if outliers exist
df[(df["Age"] < low) | (df["Age"] > up)].any(axis=None)

# Another method to check for any outliers
df[~(df["Age"] < low) | (df["Age"] > up)].any(axis=None)

### Creating a function to calculate outlier thresholds
def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile
    low_limit = quartile1 - 1.5 * interquantile
    return low_limit, up_limit

# Example usage of outlier_thresholds
outlier_thresholds(df, "Age")
outlier_thresholds(df, "Fare")

low, up = outlier_thresholds(df, "Age")
df[(df["Age"] < low) | (df["Age"] > up)].index

# Function to check for outliers in a column
def check_outlier(dataframe, col_name):
    low, up = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] < low) | (dataframe[col_name] > up)].any(axis=None):
        return True
    else:
        return False

# Check for outliers in 'Age' and 'Fare'
check_outlier(df, "Age")
check_outlier(df, "Fare")

### Handling multiple variables with outliers

####################################
#### grab_col_names ####
####################################

# Function to grab categorical and numerical columns
def grab_col_names(dataframe, cat_th=10, car_th=20):
    # Categorical columns
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtype == "object"]

    # Numerical but categorical columns (columns with unique values less than a threshold)
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtype != "object"]

    # Categorical but cardinal columns (to be checked further)
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() < car_th and dataframe[col].dtype == "object"]

    # Updating categorical columns by removing cat_but_car
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # Numerical columns
    num_cols = [col for col in dataframe.columns if dataframe[col].dtype in ["int64", "float64"]]
    num_cols = [col for col in num_cols if col not in num_but_cat]  # Remove numerical but categorical columns

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_cat: {len(num_but_cat)}")

    return cat_cols, num_cols, cat_but_car

# Example usage:
cat_cols, num_cols, cat_but_car = grab_col_names(df)

num_cols = [col for col in num_cols if col not in "PassengerId"]

# Checking for outliers in numerical columns
for col in num_cols:
    print(col, check_outlier(df, col))

######## Accessing outliers ######

# Function to grab outliers and optionally return their indexes
def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_thresholds(dataframe, col_name)
    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))]
        return outlier_index

grab_outliers(df, "Age")
grab_outliers(df, "Age", True)

### Solving outlier problems

####################################

# Removing outliers

####################################

low, up = outlier_thresholds(df, "Fare")
df.shape

# Removing the outliers in 'Fare'
df[~(df["Fare"] < low) | (df["Fare"] > up)].shape

# Function to remove outliers from a specific column
def remove_outlier(dataframe, col_name):
    low, up = outlier_thresholds(dataframe, col_name)
    df_without_outliers = dataframe[~((dataframe[col_name] < low) | (dataframe[col_name] > up))]
    return df_without_outliers

# Reload Titanic dataset and redefine columns
cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols = [col for col in num_cols if col not in ["PassengerId"]]

df.shape

# Remove outliers from numerical columns
for col in num_cols:
    df = remove_outlier(df, col)

print(f"Number of rows after removing outliers: {df.shape[0]}")

# Difference between original and new dataframe sizes
new_df.shape
new_df.shape[0] - df.shape[0]

### Capping Method (Re-assignment with thresholds)

low, up = outlier_thresholds(df, "Fare")

# Accessing outliers in 'Fare'
df[(df["Fare"] < low) | (df["Fare"] > up)]

# Capping the outliers to the threshold limits
df.loc[((df["Fare"] < low) | (df["Fare"] > up)), "Fare"]

df.loc[(df["Fare"] > up), "Fare"] = up

# Function to replace outliers with thresholds
def replace_with_thresholds(dataframe, variable):
    low, up = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low), variable] = low
    dataframe.loc[(dataframe[variable] > up), variable] = up

df = load()
cat_cols, num_cols, cat_but_car = grab_col_names(df)

num_cols = [col for col in num_cols if col not in ["PassengerId"]]

df.shape

# Check for outliers after replacement
for col in num_cols:
    print(col, check_outlier(df, col))

# Replace outliers in numerical columns
for col in num_cols:
    replace_with_thresholds(df, col)

# Check if there are still outliers after replacement
for col in num_cols:
    print(col, check_outlier(df, col))  ### Should return False for all

###################################
# Recap of the process
###################################

df = load()
outlier_thresholds(df, "Age")
check_outlier(df, "Age")
grab_outliers(df, "Age", index=True)
remove_outlier(df, "Age").shape
replace_with_thresholds(df, "Age")
check_outlier(df, "Age")