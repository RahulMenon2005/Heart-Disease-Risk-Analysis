import pandas as pd  

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input

def get_trained_classical_models(X_train, y_train):
   
    log_model = LogisticRegression()
    log_model.fit(X_train, y_train)

    
    svm_model = SVC(kernel='rbf')
    svm_model.fit(X_train, y_train)

    
    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=42
    )
    rf_model.fit(X_train, y_train)

  
    xgb_model = XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss'
    )
    xgb_model.fit(X_train, y_train)

    return log_model, svm_model, rf_model, xgb_model

def build_dl_model(input_dim):
    model = Sequential([
        Input(shape=(input_dim,)),

        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),

        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.25),

        Dense(32, activation='relu'),

        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model
def get_performance_dataframe(models, X_test, y_test):
   
    log_model, svm_model, rf_model, xgb_model, dl_model = models
    
    y_pred_lr = log_model.predict(X_test)
    y_pred_svm = svm_model.predict(X_test)
    y_pred_rf = rf_model.predict(X_test)
    y_pred_xgb = xgb_model.predict(X_test)
    y_pred_dl = (dl_model.predict(X_test) > 0.5).astype(int).flatten()
    
    model_names = ["Logistic Regression", "SVM", "Random Forest", "XGBoost", "Deep Learning"]
    predictions = [y_pred_lr, y_pred_svm, y_pred_rf, y_pred_xgb, y_pred_dl]
    
    results_df = pd.DataFrame({
        "Model": model_names,
        "Accuracy": [accuracy_score(y_test, p) for p in predictions],
        "Precision": [precision_score(y_test, p) for p in predictions],
        "Recall": [recall_score(y_test, p) for p in predictions],
        "F1 Score": [f1_score(y_test, p) for p in predictions]
    })
    
    return results_df