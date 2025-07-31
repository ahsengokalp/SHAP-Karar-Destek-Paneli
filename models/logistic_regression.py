import pandas as pd
import numpy as np
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix, roc_auc_score
)

def train_lr_model(csv_path):
    df = pd.read_csv(csv_path)
    df_encoded = df.copy()
    for col in ["asi_durumu", "yem_turu", "kumes_id"]:
        df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col])

    X = df_encoded.drop(columns=["olur_mu", "gunluk_kumes_log_id", "tarih", "saat"])
    y = df_encoded["olur_mu"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    explainer = shap.Explainer(model.predict_proba, X_train, algorithm="permutation")
    shap_values = explainer(X_test)

    shap_df = pd.DataFrame(shap_values.values[:, :, 1], columns=X.columns)
    shap_df["prediction"] = model.predict(X_test)
    shap_df["actual"] = y_test.reset_index(drop=True)
    shap_df["tarih"] = df_encoded.loc[y_test.index, "tarih"].values

    tahmin_df = X_test.copy()
    tahmin_df["prediction"] = model.predict(X_test)
    tahmin_df["actual"] = y_test.values
    tahmin_df["tarih"] = df_encoded.loc[y_test.index, "tarih"].values

    mean_shap = np.abs(shap_values.values[:, :, 1]).mean(axis=0)
    mean_shap_df = pd.DataFrame({
        "feature": X.columns,
        "mean_shap_value": mean_shap
    }).sort_values("mean_shap_value", ascending=False)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    roc_auc = roc_auc_score(y_test, y_proba)
    conf_matrix = confusion_matrix(y_test, y_pred)

    return model, explainer, shap_df, mean_shap_df, tahmin_df, X_test, y_test, accuracy, f1, report, roc_auc, conf_matrix
