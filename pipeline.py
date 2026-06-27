import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
def engineer_features(df):
    """
    Applies the specific feature engineering rules specified in image_400d05.png
    """
    df = df.copy()
    df['Balance_Salary_Ratio'] = df['Balance'] / (df['EstimatedSalary'] + 1e-5)
    
    # Product density indicator
    df['Product_Density'] = df['NumOfProducts'] / (df['Tenure'] + 1e-5)
    df['Engagement_Product_Interaction'] = df['NumOfProducts'] * (df['IsActiveMember'] * 0.5 + 0.5)

    df['Age_Tenure_Interaction'] = df['Age'] * df['Tenure']
    
    return df
def train_and_save_pipeline(data_path="churn_data.csv"):
    try:
        raw_df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"'{data_path}' not found. Generating sample data for training pipeline initialization...")
        np.random.seed(42)
        n = 1000
        raw_df = pd.DataFrame({
            'CustomerId': np.random.randint(15600000, 15800000, n),
            'Surname': ['Smith'] * n,
            'CreditScore': np.random.randint(400, 850, n),
            'Geography': np.random.choice(['France', 'Spain', 'Germany'], n),
            'Gender': np.random.choice(['Female', 'Male'], n),
            'Age': np.random.randint(18, 85, n),
            'Tenure': np.random.randint(0, 11, n),
            'Balance': np.random.uniform(0, 200000, n),
            'NumOfProducts': np.random.randint(1, 5, n),
            'HasCrCard': np.random.choice([0, 1], n),
            'IsActiveMember': np.random.choice([0, 1], n),
            'EstimatedSalary': np.random.uniform(30000, 180000, n),
            'Exited': np.random.choice([0, 1], n, p=[0.8, 0.2])
        })
    df_cleaned = raw_df.drop(columns=['CustomerId', 'Surname'], errors='ignore')
    df_engineered = engineer_features(df_cleaned)
    X = df_engineered.drop(columns=['Exited'])
    y = df_engineered['Exited']
    numeric_features = [
        'CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 
        'EstimatedSalary', 'Balance_Salary_Ratio', 'Product_Density', 
        'Engagement_Product_Interaction', 'Age_Tenure_Interaction'
    ]
    categorical_features = ['Geography', 'Gender']
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )
    full_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))
    ])
    print("Training model pipeline...")
    full_pipeline.fit(X_train, y_train)
    with open("churn_pipeline.pkl", "wb") as f:
        pickle.dump(full_pipeline, f)
    print("Pipeline successfully trained and exported as 'churn_pipeline.pkl'!")

if __name__ == "__main__":
    train_and_save_pipeline()