import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def train_general_rf(df: pd.DataFrame, target_col: str):
    df_encoded = df.copy()
    
    
    for col in df_encoded.select_dtypes(include=["object"]).columns:
        if col != target_col:
            df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col])

    X = df_encoded.drop(columns=[target_col])
    y = df_encoded[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test)

    return model, explainer, shap_values, X
