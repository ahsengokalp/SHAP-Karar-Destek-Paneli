from .db_config import supabase
import uuid
from datetime import datetime


def kaydet_tahmin(user_id, tahmin, olasilik, model_adi, input_dict, shap_df, giris_tipi="manual", veri_seti_adi=None):
    tahmin_id = str(uuid.uuid4())
    shap_json = [
        {
            "ozellik": row["Özellik"],
            "etki": float(row["SHAP Etkisi"]),
        } for _, row in shap_df.iterrows()
    ]
    response = supabase.table("tahmin_kayitlari").insert({
        "id": tahmin_id,
        "user_id": user_id,
        "tahmin": tahmin,
        "olasilik": olasilik,
        "tahmin_tarihi": datetime.now().isoformat(),
        "model_adi": model_adi,
        "input_json": input_dict,
        "shap_json": shap_json,
        "giris_tipi": giris_tipi,
        "veri_seti_adi": veri_seti_adi
    }).execute()
    return tahmin_id


def kaydet_shap_degerleri(tahmin_id, shap_df, model_adi):
    data = []
    for _, row in shap_df.iterrows():
        data.append({
            "id": str(uuid.uuid4()),
            "tahmin_id": tahmin_id,
            "ozellik": row["Özellik"],
            "deger": float(row["Değer"]),
            "shap_etikisi": float(row["SHAP Etkisi"]),
            "model_adi": model_adi
        })
    supabase.table("shap_kayitlari").insert(data).execute()


