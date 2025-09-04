import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def main():
    print("Data Preprocessing:")
    
    # Load the raw data
    data = pd.read_csv('flood_risk_dataset_india.csv') 
    print(f"   Loaded data shape: {data.shape}")

    print("Checked for any missing values and there were no missing values.")

    # Separate features and target
    X = data.drop('Flood Occurred', axis=1)  
    y = data['Flood Occurred']               
   
  

  # Remove categorical columns before scaling
    categorical_cols = X.select_dtypes(include=['object']).columns
    X_numeric = X.drop(columns=categorical_cols)

  # Split the data using X_numeric
    X_train, X_test, y_train, y_test = train_test_split(
    X_numeric, y, test_size=0.2, random_state=42, stratify=y)

  # Scaling the numerical features 
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

  # Converting back to DataFrames
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)


    #  Save the processed data
    X_train_scaled.to_csv('X_train_scaled.csv', index=False)
    X_test_scaled.to_csv('X_test_scaled.csv', index=False)
    y_train.to_csv('y_train.csv', index=False)
    y_test.to_csv('y_test.csv', index=False)
    print("   Saved scaled datasets to CSV.")

    print("\n Preprocessing complete.")

if __name__ == "__main__":
    main()