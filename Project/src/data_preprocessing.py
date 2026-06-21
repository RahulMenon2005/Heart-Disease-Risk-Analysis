import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_prepare_data(filepath):

    data = pd.read_csv(filepath)

    if "Patient_ID" in data.columns:
        data.drop("Patient_ID", axis=1, inplace=True)
        
    # Split features and target
    X = data.drop("Heart_Disease_Risk", axis=1)
    y = data["Heart_Disease_Risk"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler