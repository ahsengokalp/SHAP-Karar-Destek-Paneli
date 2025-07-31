import streamlit as st
import pandas as pd
from utils.shap_utils import yorumla_shap
from database.db_operations import kaydet_tahmin, kaydet_shap_degerleri
from database.db_config import get_supabase_client
import numpy as np

st.set_page_config(page_title="📁 Veri ile Tahmin", layout="wide")
st.title("📁 Kendi Verinizle Tahmin ve SHAP Açıklaması")

supabase = get_supabase_client()
user_id = "demo_user"

uploaded = st.file_uploader("📤 CSV dosyanızı yükleyin", type=["csv"])

if uploaded:
    user_df_raw = pd.read_csv(uploaded)
    st.subheader("📄 Veri Önizlemesi")
    st.dataframe(user_df_raw.head())

    model_secimi = st.selectbox(
        "🧠 Kullanılacak Modeli Seçin",
        ["Random Forest", "Logistic Regression", "LightGBM", "XGBoost"]
    )

    target_col = st.text_input("🎯 Hedef (tahmin edilecek) sütunun adı:")

    # Model seçimine göre fonksiyon ve model adı atanıyor
    if model_secimi == "Random Forest":
        from general_models.g_random_forest import train_general_rf as selected_train_fn
        secilen_model_adi = "General RF (CSV)"
    elif model_secimi == "Logistic Regression":
        from general_models.g_logistic_regression import train_general_lr as selected_train_fn
        secilen_model_adi = "General LR (CSV)"
    elif model_secimi == "LightGBM":
        from general_models.g_lightgbm import train_general_lgbm as selected_train_fn
        secilen_model_adi = "General LGBM (CSV)"
    elif model_secimi == "XGBoost":
        from general_models.g_xgboost import train_general_xgb as selected_train_fn
        secilen_model_adi = "General XGB (CSV)"

    if target_col and target_col in user_df_raw.columns:
        try:
            model, explainer, shap_vals, X = selected_train_fn(user_df_raw, target_col)
            st.success("✅ Model başarıyla eğitildi!")

            st.subheader("📥 Yeni Girdi ile Tahmin")
            user_inputs = []
            for col in X.columns:
                min_val = float(X[col].min())
                max_val = float(X[col].max())
                mean_val = float(X[col].mean())
                val = st.slider(f"{col}", min_val, max_val, mean_val)
                user_inputs.append(val)

            if st.button("▶️ Tahmin Et"):
                input_df = pd.DataFrame([user_inputs], columns=X.columns)
                prediction = model.predict(input_df)[0]
                proba = model.predict_proba(input_df)[0][1] if hasattr(model, "predict_proba") else None

                shap_values = explainer(input_df)
                shap_array = shap_values.values[0, :, 1] if len(shap_values.values.shape) == 3 else shap_values.values[0]

                features = input_df.iloc[0]
                shap_df = pd.DataFrame({
                    "Özellik": features.index,
                    "Değer": features.values,
                    "SHAP Etkisi": shap_array
                }).sort_values("SHAP Etkisi", key=abs, ascending=False)

                st.markdown(f"### 🎯 Tahmin Sonucu: `{prediction}`")
                if proba is not None:
                    st.markdown(f"**Olasılık:** %{proba*100:.1f}")

                st.subheader("📊 Özellik Bazlı SHAP Değerleri")
                st.dataframe(shap_df)

                st.subheader("🗣️ Model Kararı Açıklaması")
                st.markdown(yorumla_shap(shap_df))

                tahmin_id = kaydet_tahmin(
                    user_id=user_id,
                    tahmin=int(prediction),
                    olasilik=float(proba),
                    model_adi=secilen_model_adi,
                    input_dict=input_df.iloc[0].to_dict(),
                    shap_df=shap_df,
                    giris_tipi="csv",
                    veri_seti_adi=uploaded.name
                )
                kaydet_shap_degerleri(tahmin_id, shap_df, model_adi=secilen_model_adi)
                st.success("✅ Tahmin ve SHAP değerleri veritabanına kaydedildi!")

        except Exception as e:
            st.error("❌ Bir hata oluştu.")
            st.exception(e)

    elif target_col:
        st.warning("⚠️ Girilen hedef sütun veri setinde bulunamadı.")
