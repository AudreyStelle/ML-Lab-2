#%%
import sys
print(sys.executable)
# %%
import pandas as pd  # For data manipulation and analysis
# %%
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For data visualization
# %%
# Make sure to install sklearn in your terminal first!
# Use: pip install scikit-learn
from sklearn.model_selection import train_test_split  # For splitting data
# %%
from sklearn.preprocessing import MinMaxScaler, StandardScaler  # For scaling
# %%from io import StringIO  # For reading string data as file
import requests  # For HTTP requests to download data
#%%
pd.read("college_completion.csv")
# %%
college_completion = pd.read_csv('college_completion.csv')
print(f"Data loaded: {college_completion.shape}")
college_completion.head()

# %%
# %%
def get_college_completion_train_test(filepath_or_url, train_size=0.55, random_state=42):
    """
    Generate train and test datasets for College Completion problem.
    """
    df = prep_college_completion_data(filepath_or_url)
    
    test_size = 1 - train_size
    X_train, X_test, y_train, y_test = create_train_test_split(
        df,
        target_column='high_completion_q',
        test_size=test_size,
        random_state=random_state,
        stratify=True
    )
    
    return X_train, X_test, y_train, y_test
#%%
#All needed data
# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# %%
def load_data(filepath_or_url):
    return pd.read_csv(filepath_or_url)

def convert_to_categorical(df, column_list):
    df_copy = df.copy()
    df_copy[column_list] = df_copy[column_list].astype("category")
    return df_copy

def drop_columns(df, columns_to_drop):
    return df.drop(columns=columns_to_drop, errors='ignore')

def one_hot_encode(df, category_columns, drop_first=True):
    return pd.get_dummies(df, columns=category_columns, drop_first=drop_first)

def create_train_test_split(df, target_column, test_size=0.3, random_state=42, stratify=True):
    y = df[target_column]
    X = df.drop(columns=[target_column])
    stratify_var = y if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify_var)
    return X_train, X_test, y_train, y_test

def create_binned_column(df, column_name, bins, labels, new_column_name=None):
    df_copy = df.copy()
    if new_column_name is None:
        new_column_name = f"{column_name}_group"
    df_copy[new_column_name] = pd.cut(df_copy[column_name], bins=bins, labels=labels)
    return df_copy

def create_binary_target_from_quantile(df, column_name, quantile=0.75, new_column_name=None):
    df_copy = df.copy()
    if new_column_name is None:
        new_column_name = f"{column_name}_high"
    threshold = df_copy[column_name].quantile(quantile)
    df_copy[new_column_name] = (df_copy[column_name] >= threshold).astype(int)
    return df_copy

def prep_college_completion_data(filepath_or_url):
    df = load_data(filepath_or_url)
    categorical_cols = ['level', 'control', 'hbcu', 'flagship']
    df = convert_to_categorical(df, categorical_cols)
    df = create_binned_column(df, column_name='pell_value', bins=[0, 33, 66, 100], labels=['Low', 'Medium', 'High'], new_column_name='pell_group')
    df = create_binary_target_from_quantile(df, column_name='grad_150_value', quantile=0.75, new_column_name='high_completion_q')
    vsa_cols = [col for col in df.columns if col.startswith('vsa_')]
    other_cols_to_drop = ['exp_award_value', 'exp_award_state_value', 'exp_award_natl_value', 'exp_award_percentile', 'fte_value', 'fte_percentile', 'med_sat_value', 'med_sat_percentile', 'endow_value', 'endow_percentile']
    all_cols_to_drop = vsa_cols + other_cols_to_drop
    df = drop_columns(df, all_cols_to_drop)
    category_list = ['level', 'control', 'hbcu', 'flagship', 'pell_group']
    df = one_hot_encode(df, category_list, drop_first=True)
    return df

def get_college_completion_train_test(filepath_or_url, train_size=0.55, random_state=42):
    df = prep_college_completion_data(filepath_or_url)
    test_size = 1 - train_size
    X_train, X_test, y_train, y_test = create_train_test_split(df, target_column='high_completion_q', test_size=test_size, random_state=random_state, stratify=True)
    return X_train, X_test, y_train, y_test

print("All functions loaded successfully!")
#%%
"""
Question 1: Use the question/target variable and build a model 
(ensure it's a classification problem)
"""
#%%
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# %%
# Load and prepare the data using the get_college_completion_train_test function
X_train, X_test, y_train, y_test = get_college_completion_train_test(
    'college_completion.csv',
    train_size=0.55,
    random_state=42
)

print("Data loaded successfully!")

# %%
# Verify this is a classification problem
print("=" * 60)
print("QUESTION 1: VERIFY AND BUILD MODEL")
print("=" * 60)
#%%
print("\nMy Question: How does student financial need impact college completion rates?")
print("\nTarget Variable: high_completion_q")
print("  - Type: Binary Classification")
print("  - 1 = High completion (top quartile of graduation rates)")
print("  - 0 = Low completion (below top quartile)")
#%%
print(f"\nData loaded successfully:")
print(f"  Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
print(f"  Test set: {X_test.shape[0]} samples, {X_test.shape[1]} features")
#%%
print(f"\nTarget Distribution:")
print(f"  Training: {y_train.value_counts().to_dict()}")
print(f"  Test: {y_test.value_counts().to_dict()}")
#%%
print(f"\n✓ Confirmed: This is a CLASSIFICATION problem (binary)")
print(f"✓ Target variable is already properly formatted for classification")
#%%
print("\n" + "=" * 60)
print("✓ QUESTION 1 COMPLETE")
print("=" * 60)


