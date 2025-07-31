import pandas as pd
import numpy as np
import shap
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix, roc_auc_score
)

def train_lgbm_model(csv_path):
    df = pd.read_csv(csv_path)
    df_encoded = df.copy()
    for col in ["asi_durumu", "yem_turu", "kumes_id"]:
        df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col])

    X = df_encoded.drop(columns=["olur_mu", "gunluk_kumes_log_id", "tarih", "saat"])
    y = df_encoded["olur_mu"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LGBMClassifier()
    model.fit(X_train, y_train)

    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test)

    shap_df = pd.DataFrame(shap_values.values, columns=X.columns)
    shap_df["prediction"] = model.predict(X_test)
    shap_df["actual"] = y_test.reset_index(drop=True)
    shap_df["tarih"] = df_encoded.loc[y_test.index, "tarih"].values

    tahmin_df = X_test.copy()
    tahmin_df["prediction"] = model.predict(X_test)
    tahmin_df["actual"] = y_test.values
    tahmin_df["tarih"] = df_encoded.loc[y_test.index, "tarih"].values

    mean_shap = pd.DataFrame(shap_values.values, columns=X.columns).abs().mean().sort_values(ascending=False)
    mean_shap_df = mean_shap.reset_index()
    mean_shap_df.columns = ["feature", "mean_shap_value"]

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    roc_auc = roc_auc_score(y_test, y_proba)
    conf_matrix = confusion_matrix(y_test, y_pred)

    return model, explainer, shap_df, mean_shap_df, tahmin_df, X_test, y_test, accuracy, f1, report, roc_auc, conf_matrix
