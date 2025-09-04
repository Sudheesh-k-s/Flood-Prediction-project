# performs basic exploratory data analysis (EDA) on the flood dataset
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 

# Load the dataset
print("Loading the data and performing EDA.")
data = pd.read_csv('flood_risk_dataset_india.csv') 


print("1. Dataset Shape (Rows, Columns):", data.shape)
print("\n2. Column Data Types:")
print(data.dtypes)
print("\n3. First 5 rows:")
print(data.head())

#  Checking for missing values

print("\n4. Missing Values:")
print(data.isnull().sum())

# 3. Analyze the Target Variable
print("\n5. Distribution of Target Variable ('Flood'):")
target_counts = data['Flood Occurred'].value_counts() 
print(target_counts)

# Plot the distribution
plt.figure(figsize=(8, 5))
sns.barplot(x=target_counts.index, y=target_counts.values)
plt.title('Flood vs No-Flood Events in Dataset')
plt.xlabel('Flood Occurred (1=Yes, 0=No)')
plt.ylabel('Count')
plt.savefig('target_distribution.png') 
plt.show()

#  Check Data Distributions for Numerical Features
print("\n6. Generating distribution plots for numerical features..")


numerical_features = data.select_dtypes(include=['float64', 'int64']).columns
numerical_features = numerical_features.drop('Flood Occurred')

# Clean numerical features
for feature in numerical_features:
    data[feature] = pd.to_numeric(data[feature], errors='coerce')

data = data.dropna(subset=numerical_features)

import re

for feature in numerical_features:
    plt.figure(figsize=(8, 5))
    sns.histplot(data[feature], kde=True) 
    plt.title(f'Distribution of {feature}')
    safe_feature = re.sub(r'[^a-zA-Z0-9]', '_', feature)
    plt.savefig(f'dist_{safe_feature}.png') 
    plt.show()

# Correlation Heatmap 
print("\n7. Generating Correlation Heatmap.")
plt.figure(figsize=(12, 8))
numeric_data = data.select_dtypes(include=['float64', 'int64'])
correlation_matrix = numeric_data.corr()

# Creating a heatmap
sns.heatmap(correlation_matrix,
            annot=True, 
            cmap='coolwarm', 
            center=0) 

plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=100) 
plt.show()


print("\nEDA Completed.")