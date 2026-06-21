import pandas as pd
import numpy as np

def get_user_input(feature_columns):
    
    user_values = []
    print("\n--- Patient Medical Input Console ---")
    for col in feature_columns:
        val = float(input(f"Enter {col}: "))
        user_values.append(val)
        
    return pd.DataFrame([user_values], columns=feature_columns)

def generate_recommendations(user_df):
   
    recs = []
    
    if user_df['BMI'].values[0] > 25:
        recs.append("✔ Reduce weight")
    if user_df['Systolic_BP'].values[0] > 130:
        recs.append("✔ Control blood pressure")
    if user_df['Cholesterol_LDL'].values[0] > 130:
        recs.append("✔ Reduce cholesterol")
    if user_df['Smoking_Status'].values[0] == 1:
        recs.append("✔ Quit smoking")
    if user_df['Physical_Activity_Level'].values[0] < 2:
        recs.append("✔ Increase exercise")
    if user_df['Sleep_Hours'].values[0] < 7:
        recs.append("✔ Improve sleep")
    if user_df['Stress_Level'].values[0] > 6:
        recs.append("✔ Manage stress")
        
    if not recs:
        recs.append("✔ Maintain healthy lifestyle")
        
    return recs