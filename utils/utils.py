import pandas as pd
from utils.timing_utils import hesapla_sure
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

def skor_tablosu_olustur(models_dict, csv_path):
    skorlar = []
    for name, fn in models_dict.items():
        sonuc, sure = hesapla_sure(fn, csv_path)
        model, expl, shap_df, mean_shap, tahmin_df, X_test, y_test, acc, f1, report, roc_auc, cm = sonuc

        skorlar.append({
            "Model": name,
            "Accuracy": round(acc, 3),
            "F1 Score": round(f1, 3),
            "Precision": round(report['1']['precision'], 3),
            "Recall": round(report['1']['recall'], 3),
            "ROC AUC": round(roc_auc, 3),
            "Süre (s)": sure
        })

    return pd.DataFrame(skorlar).set_index("Model")
