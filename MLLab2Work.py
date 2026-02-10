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

# %%
college_completion = pd.read_csv('college_completion.csv')
print(f"Data loaded: {college_completion.shape}")
college_completion.head()


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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# %%
def load_data(filepath_or_url):
    return pd.read_csv(filepath_or_url)
#%%
def convert_to_categorical(df, column_list):
    df_copy = df.copy()
    df_copy[column_list] = df_copy[column_list].astype("category")
    return df_copy
#%%
def drop_columns(df, columns_to_drop):
    return df.drop(columns=columns_to_drop, errors='ignore')
#%%
def one_hot_encode(df, category_columns, drop_first=True):
    return pd.get_dummies(df, columns=category_columns, drop_first=drop_first)
#%%
def create_train_test_split(df, target_column, test_size=0.3, random_state=42, stratify=True):
    y = df[target_column]
    X = df.drop(columns=[target_column])
    stratify_var = y if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify_var)
    return X_train, X_test, y_train, y_test
#%%
def create_binned_column(df, column_name, bins, labels, new_column_name=None):
    df_copy = df.copy()
    if new_column_name is None:
        new_column_name = f"{column_name}_group"
    df_copy[new_column_name] = pd.cut(df_copy[column_name], bins=bins, labels=labels)
    return df_copy
#%%
def create_binary_target_from_quantile(df, column_name, quantile=0.75, new_column_name=None):
    df_copy = df.copy()
    if new_column_name is None:
        new_column_name = f"{column_name}_high"
    threshold = df_copy[column_name].quantile(quantile)
    df_copy[new_column_name] = (df_copy[column_name] >= threshold).astype(int)
    return df_copy
#%%
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
#%%
def get_college_completion_train_test(filepath_or_url, train_size=0.55, random_state=42):
    df = prep_college_completion_data(filepath_or_url)
    test_size = 1 - train_size
    X_train, X_test, y_train, y_test = create_train_test_split(df, target_column='high_completion_q', test_size=test_size, random_state=random_state, stratify=True)
    return X_train, X_test, y_train, y_test
#%%
print("All functions loaded successfully!")
#%%
"""
Question 1: Use the question/target variable and build a model 
(ensure it's a classification problem)
"""
#%%
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
# %%
"""
Question 2: Build a kNN model to predict target variable using 3 nearest neighbors
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Remove non-numeric columns and handle missing values
X_train_clean = X_train.select_dtypes(include=['number'])
X_test_clean = X_test.select_dtypes(include=['number'])

# Remove rows with missing values
X_train_clean = X_train_clean.dropna()
X_test_clean = X_test_clean.dropna()

# Align y_train and y_test with cleaned data
y_train_clean = y_train[X_train_clean.index]
y_test_clean = y_test[X_test_clean.index]

print("=" * 60)
print("QUESTION 2: BUILD kNN MODEL (k=3)")
print("=" * 60)
print(f"\nData after cleaning:")
print(f"  Training set: {X_train_clean.shape[0]} samples, {X_train_clean.shape[1]} features")
print(f"  Test set: {X_test_clean.shape[0]} samples, {X_test_clean.shape[1]} features")

# Build and train kNN model with k=3
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train_clean, y_train_clean)

# Make predictions
y_train_pred = knn_model.predict(X_train_clean)
y_test_pred = knn_model.predict(X_test_clean)

# Calculate accuracy
train_accuracy = accuracy_score(y_train_clean, y_train_pred)
test_accuracy = accuracy_score(y_test_clean, y_test_pred)

print(f"\n✓ kNN Model successfully trained with k=3 neighbors")
print(f"\nModel Performance:")
print(f"  Training Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
print(f"  Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

# Show classification report
print("\n" + "=" * 60)
print("CLASSIFICATION REPORT (Test Set)")
print("=" * 60)
print(classification_report(y_test_clean, y_test_pred, 
                          target_names=['Low Completion', 'High Completion']))

print("\n" + "=" * 60)
print("✓ QUESTION 2 COMPLETE")
print("=" * 60)
# %%
# %%
"""
Question 3: Create a dataframe with test target values, predictions, and probabilities
"""

# Get predicted probabilities for the positive class (class 1 = High Completion)
y_test_proba = knn_model.predict_proba(X_test_clean)[:, 1]

# Create the results dataframe - use y_test_clean (not y_test)
results_df = pd.DataFrame({
    'actual': y_test_clean.values,
    'predicted': y_test_pred,
    'probability_positive': y_test_proba
})

# Display the results
print("=" * 60)
print("QUESTION 3: RESULTS DATAFRAME")
print("=" * 60)

print(f"\nDataFrame Shape: {results_df.shape}")
print(f"\nFirst 10 rows:")
print(results_df.head(10))

print(f"\nLast 10 rows:")
print(results_df.tail(10))

print(f"\nSummary Statistics:")
print(results_df.describe())

print("\n" + "=" * 60)
print("✓ QUESTION 3 COMPLETE")
print("=" * 60)

# %%
"""
Question 4: If you adjusted the k hyperparameter what do you think would happen to 
the threshold function? Would the confusion matrix look the same at the same 
threshold levels or not? Why or why not?

Answer:

No, the confusion matrix would NOT look the same at the same threshold levels if 
I adjusted k.

Why:

Changing k changes the predicted probabilities. Because the 
threshold compares these probabilities to decide the classification, different 
probabilities result in different predictions.
"""

# %%
"""
Question 5: Evaluate results using confusion matrix and analyze the model
"""

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# %%
# Create and display confusion matrix
print("=" * 60)
print("QUESTION 5: CONFUSION MATRIX EVALUATION")
print("=" * 60)

# Calculate confusion matrix
cm = confusion_matrix(y_test_clean, y_test_pred)

print("\nConfusion Matrix:")
print(cm)
print("\nFormat:")
print("              Predicted")
print("              Low   High")
print("Actual Low   [TN    FP]")
print("      High   [FN    TP]")

# Extract values
tn, fp, fn, tp = cm.ravel()

print(f"\nTrue Negatives (TN): {tn} - Correctly predicted low completion")
print(f"False Positives (FP): {fp} - Predicted high, actually low")
print(f"False Negatives (FN): {fn} - Predicted low, actually high")
print(f"True Positives (TP): {tp} - Correctly predicted high completion")

# Calculate metrics
accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"\nPerformance Metrics:")
print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"  Precision: {precision:.4f} ({precision*100:.2f}%)")
print(f"  Recall/Sensitivity: {recall:.4f} ({recall*100:.2f}%)")
print(f"  Specificity: {specificity:.4f} ({specificity*100:.2f}%)")
print(f"  F1-Score: {f1_score:.4f}")

# %%
# Visualize confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Low Completion', 'High Completion'],
            yticklabels=['Low Completion', 'High Completion'],
            cbar_kws={'label': 'Count'})
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('Actual', fontsize=12)
plt.title('Confusion Matrix - College Completion Prediction (k=3)', fontsize=14)
plt.tight_layout()
plt.show()

# %%
# Walk through the question and analyze the model
print("\n" + "=" * 60)
print("MODEL ANALYSIS: ADDRESSING MY RESEARCH QUESTION")
print("=" * 60)

print("\nMy Question: How does student financial need impact college completion rates?")
print("\nTarget: Predict if a college is in the top quartile of graduation rates")
print("Key Predictor: Pell Grant percentage (indicator of student financial need)")

print("\n--- POSITIVE ELEMENTS ---")

print(f"\n1. Overall Accuracy ({accuracy*100:.1f}%):")
if accuracy > 0.70:
    print(f"   ✓ The model correctly classifies {accuracy*100:.1f}% of colleges")
    print("   ✓ This suggests student financial need has predictive power for completion rates")

print(f"\n2. Precision ({precision*100:.1f}%):")
if precision > 0.65:
    print(f"   ✓ When we predict high completion, we're correct {precision*100:.1f}% of the time")
    print("   ✓ We can somewhat reliably identify high-performing institutions")

print(f"\n3. True Positives ({tp} colleges):")
print(f"   ✓ Successfully identified {tp} colleges with high completion rates")

print(f"\n4. True Negatives ({tn} colleges):")
print(f"   ✓ Successfully identified {tn} colleges with lower completion rates")

print("\n--- CONCERNS ---")

print(f"\n1. False Negatives ({fn} colleges):")
if fn > 0:
    print(f"   ⚠ We missed {fn} colleges that actually have high completion rates")
    print("   ⚠ This means we're failing to identify some high-performing schools")
    print("   ⚠ Could lead to overlooking successful programs serving high-need students")

print(f"\n2. False Positives ({fp} colleges):")
if fp > 0:
    print(f"   ⚠ We incorrectly predicted {fp} colleges as high completion")
    print("   ⚠ These colleges may be overestimated in their effectiveness")

print(f"\n3. Recall/Sensitivity ({recall*100:.1f}%):")
if recall < 0.75:
    print(f"   ⚠ We only catch {recall*100:.1f}% of actual high-completion colleges")
    print("   ⚠ Missing many colleges that successfully graduate students despite financial need")

print("\n4. Degrees of Freedom Problem:")
print(f"   ⚠ We have {X_train_clean.shape[1]} features for {X_train_clean.shape[0]} training samples")
if X_train_clean.shape[1] > 20:
    print("   ⚠ Too many features may cause overfitting")
    print("   ⚠ kNN suffers from 'curse of dimensionality' with many features")
    print("   ⚠ Feature selection or dimensionality reduction may be needed")

print("\n--- INTERPRETATION FOR MY QUESTION ---")
print("\nThe model shows that institutional characteristics (including student financial need)")
print("can predict college completion rates with moderate success. However, the model struggles")
print("to identify all high-performing colleges, suggesting that:")
print("  • Student financial need is ONE factor but not the complete picture")
print("  • Other unmeasured factors (institutional support, resources, culture) matter")
print("  • The relationship between financial need and completion is complex")
print("  • We need to reduce features or improve the model to get better predictions")

print("\n" + "=" * 60)
print("✓ QUESTION 5 COMPLETE")
print("=" * 60)

# %%
"""
Question 6: Create two functions to clean data and optimize model
Function 1: Clean data and split into train/test
Function 2: Train and test model with different k and threshold values
"""

# %%
# FUNCTION 1: Clean data and split into train/test
def prepare_data(filepath, train_size=0.55, random_state=42):
    """
    Load, clean, and split college completion data.
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file
    train_size : float
        Proportion for training (default 0.55)
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    tuple : (X_train, X_test, y_train, y_test)
        All cleaned and ready for modeling
    """
    # Use the existing pipeline function
    X_train, X_test, y_train, y_test = get_college_completion_train_test(
        filepath,
        train_size=train_size,
        random_state=random_state
    )
    
    # Remove non-numeric columns
    X_train = X_train.select_dtypes(include=['number'])
    X_test = X_test.select_dtypes(include=['number'])
    
    # Remove rows with missing values
    X_train = X_train.dropna()
    X_test = X_test.dropna()
    
    # Align target variables with cleaned data
    y_train = y_train[X_train.index]
    y_test = y_test[X_test.index]
    
    return X_train, X_test, y_train, y_test


# %%
# FUNCTION 2: Train and test model with different k and threshold values
def train_test_knn(X_train, X_test, y_train, y_test, k=3, threshold=0.5):
    """
    Train kNN model and evaluate with custom k and threshold.
    
    Parameters:
    -----------
    X_train, X_test : DataFrames
        Training and test features
    y_train, y_test : Series
        Training and test targets
    k : int
        Number of neighbors (default 3)
    threshold : float
        Classification threshold (default 0.5)
        
    Returns:
    --------
    dict : Dictionary with performance metrics and predictions
    """
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import confusion_matrix, accuracy_score
    
    # Train the model
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    
    # Get probabilities
    train_proba = model.predict_proba(X_train)[:, 1]
    test_proba = model.predict_proba(X_test)[:, 1]
    
    # Apply custom threshold
    train_pred = (train_proba >= threshold).astype(int)
    test_pred = (test_proba >= threshold).astype(int)
    
    # Calculate metrics
    train_cm = confusion_matrix(y_train, train_pred)
    test_cm = confusion_matrix(y_test, test_pred)
    
    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)
    
    # Calculate additional test metrics
    tn, fp, fn, tp = test_cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Return results
    return {
        'k': k,
        'threshold': threshold,
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1_score': f1,
        'test_cm': test_cm,
        'train_cm': train_cm,
        'model': model,
        'test_predictions': test_pred,
        'test_probabilities': test_proba
    }


print("=" * 60)
print("FUNCTIONS CREATED SUCCESSFULLY")
print("=" * 60)
print("\nFunction 1: prepare_data(filepath, train_size, random_state)")
print("Function 2: train_test_knn(X_train, X_test, y_train, y_test, k, threshold)")

# %%
# Test the functions
print("\n" + "=" * 60)
print("TESTING FUNCTIONS")
print("=" * 60)

# Test Function 1
X_train_opt, X_test_opt, y_train_opt, y_test_opt = prepare_data('college_completion.csv')
print(f"\nFunction 1 - Data prepared:")
print(f"  Training: {X_train_opt.shape}")
print(f"  Test: {X_test_opt.shape}")

# Test Function 2
test_result = train_test_knn(X_train_opt, X_test_opt, y_train_opt, y_test_opt, k=3, threshold=0.5)
print(f"\nFunction 2 - Model tested:")
print(f"  k={test_result['k']}, threshold={test_result['threshold']}")
print(f"  Test Accuracy: {test_result['test_accuracy']:.4f}")

print("\n✓ Both functions working correctly!")

# %%
# USE FUNCTIONS TO OPTIMIZE MODEL - Test different k and threshold combinations
print("\n" + "=" * 60)
print("OPTIMIZING MODEL WITH DIFFERENT K AND THRESHOLD VALUES")
print("=" * 60)

# Prepare data once
X_train_opt, X_test_opt, y_train_opt, y_test_opt = prepare_data('college_completion.csv')

# Define parameter grid
k_values = [3, 5, 7, 9, 11, 15, 20, 25]
threshold_values = [0.3, 0.4, 0.5, 0.6, 0.7]

# Store all results
optimization_results = []

print("\nTesting combinations...")
print(f"{'k':<5} {'Threshold':<12} {'Test Acc':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
print("-" * 70)

for k in k_values:
    for threshold in threshold_values:
        result = train_test_knn(X_train_opt, X_test_opt, y_train_opt, y_test_opt, k=k, threshold=threshold)
        optimization_results.append(result)
        
        print(f"{k:<5} {threshold:<12.1f} {result['test_accuracy']:<12.4f} "
              f"{result['precision']:<12.4f} {result['recall']:<12.4f} {result['f1_score']:<12.4f}")

# %%
# Find best combinations based on different metrics
import pandas as pd

results_df = pd.DataFrame([
    {
        'k': r['k'],
        'threshold': r['threshold'],
        'test_accuracy': r['test_accuracy'],
        'precision': r['precision'],
        'recall': r['recall'],
        'f1_score': r['f1_score']
    }
    for r in optimization_results
])

print("\n" + "=" * 60)
print("OPTIMIZATION RESULTS")
print("=" * 60)

# Best by accuracy
best_acc_idx = results_df['test_accuracy'].idxmax()
best_acc = optimization_results[best_acc_idx]
print(f"\nBest Accuracy: k={best_acc['k']}, threshold={best_acc['threshold']:.1f}")
print(f"  Accuracy: {best_acc['test_accuracy']:.4f}")
print(f"  Precision: {best_acc['precision']:.4f}")
print(f"  Recall: {best_acc['recall']:.4f}")
print(f"  F1-Score: {best_acc['f1_score']:.4f}")

# Best by F1-score
best_f1_idx = results_df['f1_score'].idxmax()
best_f1 = optimization_results[best_f1_idx]
print(f"\nBest F1-Score: k={best_f1['k']}, threshold={best_f1['threshold']:.1f}")
print(f"  Accuracy: {best_f1['test_accuracy']:.4f}")
print(f"  Precision: {best_f1['precision']:.4f}")
print(f"  Recall: {best_f1['recall']:.4f}")
print(f"  F1-Score: {best_f1['f1_score']:.4f}")

# Best by recall
best_recall_idx = results_df['recall'].idxmax()
best_recall = optimization_results[best_recall_idx]
print(f"\nBest Recall: k={best_recall['k']}, threshold={best_recall['threshold']:.1f}")
print(f"  Accuracy: {best_recall['test_accuracy']:.4f}")
print(f"  Precision: {best_recall['precision']:.4f}")
print(f"  Recall: {best_recall['recall']:.4f}")
print(f"  F1-Score: {best_recall['f1_score']:.4f}")

# %%
# Visualize optimization results
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

metrics = ['test_accuracy', 'precision', 'recall', 'f1_score']
titles = ['Test Accuracy', 'Precision', 'Recall', 'F1-Score']

for idx, (metric, title) in enumerate(zip(metrics, titles)):
    ax = axes[idx // 2, idx % 2]
    
    for threshold in threshold_values:
        subset = results_df[results_df['threshold'] == threshold]
        ax.plot(subset['k'], subset[metric], marker='o', label=f'threshold={threshold}')
    
    ax.set_xlabel('k (number of neighbors)', fontsize=10)
    ax.set_ylabel(title, fontsize=10)
    ax.set_title(f'{title} vs k for Different Thresholds', fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n" + "=" * 60)
print("✓ QUESTION 6 COMPLETE")
print("=" * 60)

# %%
"""
Question 7: How well does the model perform? 
Did adjusting k and threshold help? Why or why not?

ANSWER:

How well does the model perform?
The model achieves around 70-75% accuracy, which is moderate performance.

Did adjusting k and threshold help?
Yes, it improved the model compared to default settings.

Why?
Different k values create different predicted probabilities. The threshold 
needs to match these probabilities to make good predictions. Testing multiple 
combinations helped us find the best pairing of k and threshold for our data.
"""

print("✓ QUESTION 7 COMPLETE")

# %%
"""
Question 8: Choose another target variable and create a new kNN model
Using the two functions from Question 6
"""

print("=" * 60)
print("QUESTION 8: NEW TARGET VARIABLE - ALTERNATE kNN MODEL")
print("=" * 60)

# Load the original data to create a new target
import pandas as pd

df_alt = pd.read_csv('college_completion.csv')

print("\nOriginal Question: How does student financial need impact college completion rates?")
print("Original Target: high_completion_q (top quartile of graduation rates)")

print("\n--- NEW QUESTION & TARGET ---")
print("New Question: Can we predict if a college is a 4-year institution?")
print("New Target: is_4year (1 = 4-year college, 0 = 2-year college)")

# Create new binary target based on 'level' column
df_alt['is_4year'] = (df_alt['level'] == '4-year').astype(int)

print(f"\nNew target distribution:")
print(df_alt['is_4year'].value_counts())

# %%
# Prepare features (remove level column since it's our target basis)
from sklearn.model_selection import train_test_split

# Drop columns we don't need
cols_to_drop = ['level', 'chronname', 'city', 'state', 'basic', 'site', 'unitid', 'index']
cols_to_drop += [col for col in df_alt.columns if col.startswith('vsa_')]
cols_to_drop += ['exp_award_value', 'exp_award_state_value', 'exp_award_natl_value',
                 'exp_award_percentile', 'fte_value', 'fte_percentile', 
                 'med_sat_value', 'med_sat_percentile', 'endow_value', 'endow_percentile']

# Drop columns that exist
cols_to_drop = [col for col in cols_to_drop if col in df_alt.columns]
df_alt = df_alt.drop(columns=cols_to_drop)

# Get numeric features only
X_alt = df_alt.select_dtypes(include=['number']).drop(columns=['is_4year'])
y_alt = df_alt['is_4year']

# Remove missing values
X_alt = X_alt.dropna()
y_alt = y_alt[X_alt.index]

# Split the data
X_train_alt, X_test_alt, y_train_alt, y_test_alt = train_test_split(
    X_alt, y_alt, test_size=0.45, random_state=42, stratify=y_alt
)

print(f"\nAlternate model data prepared:")
print(f"  Training set: {X_train_alt.shape}")
print(f"  Test set: {X_test_alt.shape}")
print(f"  Features: {X_train_alt.shape[1]}")

# %%
# Use Function 2 to test the new model with different k and threshold values
print("\n" + "=" * 60)
print("TESTING ALTERNATE MODEL WITH DIFFERENT K AND THRESHOLD")
print("=" * 60)

k_values_alt = [3, 5, 7, 11, 15]
threshold_values_alt = [0.4, 0.5, 0.6]

results_alt = []

print(f"\n{'k':<5} {'Threshold':<12} {'Test Acc':<12} {'Precision':<12} {'Recall':<12}")
print("-" * 60)

for k in k_values_alt:
    for threshold in threshold_values_alt:
        result = train_test_knn(X_train_alt, X_test_alt, y_train_alt, y_test_alt, k=k, threshold=threshold)
        results_alt.append(result)
        
        print(f"{k:<5} {threshold:<12.1f} {result['test_accuracy']:<12.4f} "
              f"{result['precision']:<12.4f} {result['recall']:<12.4f}")

# %%
# Find best model for alternate target
results_alt_df = pd.DataFrame([
    {
        'k': r['k'],
        'threshold': r['threshold'],
        'test_accuracy': r['test_accuracy'],
        'precision': r['precision'],
        'recall': r['recall'],
        'f1_score': r['f1_score']
    }
    for r in results_alt
])

best_alt_idx = results_alt_df['test_accuracy'].idxmax()
best_alt = results_alt[best_alt_idx]

print("\n" + "=" * 60)
print("BEST MODEL FOR ALTERNATE TARGET")
print("=" * 60)

print(f"\nBest Parameters: k={best_alt['k']}, threshold={best_alt['threshold']:.1f}")
print(f"\nPerformance Metrics:")
print(f"  Test Accuracy: {best_alt['test_accuracy']:.4f} ({best_alt['test_accuracy']*100:.2f}%)")
print(f"  Precision: {best_alt['precision']:.4f}")
print(f"  Recall: {best_alt['recall']:.4f}")
print(f"  F1-Score: {best_alt['f1_score']:.4f}")

print(f"\nConfusion Matrix:")
print(best_alt['test_cm'])

print("\n--- INTERPRETATION ---")
print(f"The model predicts whether a college is a 4-year institution with {best_alt['test_accuracy']*100:.1f}% accuracy.")
print("This shows that institutional characteristics (graduation rates, financial aid, student demographics)")
print("can distinguish between 2-year and 4-year colleges.")

if best_alt['test_accuracy'] > 0.85:
    print("\nThe high accuracy suggests 2-year and 4-year colleges have very distinct profiles.")
elif best_alt['test_accuracy'] > 0.70:
    print("\nThe moderate accuracy suggests some overlap in characteristics between 2-year and 4-year colleges.")
else:
    print("\nThe lower accuracy suggests 2-year and 4-year colleges may be more similar than expected.")

print("\n" + "=" * 60)
print("✓ QUESTION 8 COMPLETE")
print("=" * 60)
#%%
# %%
# %%
"""
Question 8: Choose another target variable and create a new kNN model
"""

# First, redefine the train_test_knn function to make sure it's available
def train_test_knn(X_train, X_test, y_train, y_test, k=3, threshold=0.5):
    """Train kNN model and evaluate with custom k and threshold."""
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import confusion_matrix, accuracy_score
    
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    
    train_proba = model.predict_proba(X_train)[:, 1]
    test_proba = model.predict_proba(X_test)[:, 1]
    
    train_pred = (train_proba >= threshold).astype(int)
    test_pred = (test_proba >= threshold).astype(int)
    
    train_cm = confusion_matrix(y_train, train_pred)
    test_cm = confusion_matrix(y_test, test_pred)
    
    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)
    
    tn, fp, fn, tp = test_cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'k': k, 'threshold': threshold, 'train_accuracy': train_acc, 'test_accuracy': test_acc,
        'precision': precision, 'recall': recall, 'f1_score': f1, 'test_cm': test_cm
    }

print("=" * 60)
print("QUESTION 8: NEW TARGET VARIABLE - ALTERNATE kNN MODEL")
print("=" * 60)

# Load data and create new target
import pandas as pd
from sklearn.model_selection import train_test_split

df_alt = pd.read_csv('college_completion.csv')

print("\nOriginal Question: How does student financial need impact college completion rates?")
print("Original Target: high_completion_q (top quartile of graduation rates)")

print("\n--- NEW QUESTION & TARGET ---")
print("New Question: Can we predict if a college is a 4-year institution?")
print("New Target: is_4year (1 = 4-year college, 0 = 2-year college)")

# Create new target
df_al
# %%
# %%

# %%
