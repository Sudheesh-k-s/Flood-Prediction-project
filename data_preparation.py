import pandas as pd

print("loading the data")
data = pd.read_csv('flood_risk_dataset_india.csv')

print("Data Shape (Rows,Columns):", data.shape)
print("\nFirst 5 rows:")
print(data.head())
print("\nColumn names:")
print(data.columns)


print("\nChecking for missing values...")
print(data.isnull().sum())


X = data.drop('Flood Occurred', axis=1) 

y = data['Flood Occurred']

print("\nFeatures shape (X):", X.shape)
print("Target shape (y):", y.shape)


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nTraining Set: {X_train.shape}, {y_train.shape}")
print(f"Testing Set: {X_test.shape}, {y_test.shape}")


X_train.to_csv('X_train.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_test.to_csv('y_test.csv', index=False)

print("\nCleaned and split data has been saved to CSV files!")
print("\n completed!")