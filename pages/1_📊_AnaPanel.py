import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt

from models import (
    random_forest,
    gradient_boosting,
    logistic_regression as lr,
    xgboost_model as xgb,
    lightgbm_model as lgb
)
from general_models import g_random_forest
from utils.utils import skor_tablosu_olustur
from utils.shap_utils import yorumla_shap
from database.db_operations import kaydet_tahmin, kaydet_shap_degerleri
import uuid
from database.db_operations import kaydet_tahmin



st.set_page_config(page_title="SHAP Karar Destek Paneli", layout="wide")
st.markdown("<h1 style='color:#3A5FCD;'>SHAP Karar Destek Paneli</h1>", unsafe_allow_html=True)
st.markdown("Bu panelde önce modeli seçin, ardından tahmin ve SHAP açıklamalarını inceleyin.")


if "model" not in st.session_state:
    result = random_forest.train_rf_model("data/tavuk_olum_analiz.csv")
    model, explainer, shap_df, mean_shap_df, tahmin_df, X_test, y_test, acc, f1, report, roc_auc, cm = result
    st.session_state.update({
        "model": model,
        "explainer": explainer,
        "shap_df": shap_df,
        "mean_shap": mean_shap_df,
        "feature_cols": X_test.columns.tolist(),
        "model_adi": "Random Forest"
    })


tabs = st.tabs(["⚙️ Model Seçimi", "🔍 Tahmin", "🧠 SHAP Açıklama"])


with tabs[0]:
    st.header("⚙️ Model Seçimi ve Eğitim")

    models = {
        "Random Forest": random_forest.train_rf_model,
        "Gradient Boosting": gradient_boosting.train_gb_model,
        "Logistic Regression": lr.train_lr_model,
        "XGBoost": xgb.train_xgb_model,
        "LightGBM": lgb.train_lgbm_model
    }

    df_scores = skor_tablosu_olustur(models, "data/tavuk_olum_analiz.csv")
    st.dataframe(df_scores)

    choice = st.selectbox("Model Seçin:", list(models.keys()))
    if st.button("▶️ Eğit ve Uygula"):
        result = models[choice]("data/tavuk_olum_analiz.csv")
        model, explainer, shap_df, mean_shap_df, tahmin_df, X_test, y_test, acc, f1, report, roc_auc, cm = result

        st.session_state.update({
            "model": model,
            "explainer": explainer,
            "shap_df": shap_df,
            "mean_shap": mean_shap_df,
            "feature_cols": X_test.columns.tolist(),
            "model_adi": choice
        })
        st.success(f"{choice} modeli başarıyla eğitildi!")
        st.rerun()

with tabs[1]:
    st.header("🔍 Tavuk Ölüm Tahmini")
    col1, col2, col3 = st.columns(3)
    with col1:
        sicaklik = st.slider("Sıcaklık (°C)", 10, 45, 30)
        yas = st.slider("Yaş (gün)", 1, 100, 20)
        hava_kalite = st.slider("Hava Kalite İndeksi", 0, 100, 50)
        su_tuketimi = st.slider("Su Tüketimi (L)", 0.0, 50.0, 25.0)
    with col2:
        nem = st.slider("Nem (%)", 20, 100, 60)
        asi = st.selectbox("Aşı Durumu", ["Aşılı", "Aşısız"])
        tavuk_sayisi = st.slider("Tavuk Sayısı", 0, 200, 20)
        gunluk_olum = st.slider("Günlük Ölüm", 0, 50, 0)
    with col3:
        yem = st.selectbox("Yem Türü", ["Organik", "Standart", "Vitaminli"])
        yem_miktari = st.slider("Yem Miktarı (kg)", 0.0, 50.0, 10.0)
        kumes_id = st.selectbox("Kümes ID", [0, 1, 2, 3, 4, 5])

    if st.button("Tahmin Et"):
        asi_enc = 1 if asi == "Aşılı" else 0
        yem_enc = {"Organik": 0, "Standart": 1, "Vitaminli": 2}[yem]
        cols = st.session_state["feature_cols"]
        inp = pd.DataFrame([[kumes_id, tavuk_sayisi, yas, asi_enc, sicaklik, nem,
                             hava_kalite, yem_enc, yem_miktari, su_tuketimi, gunluk_olum]], columns=cols)

        pred = st.session_state["model"].predict(inp)[0]
        proba = st.session_state["model"].predict_proba(inp)[0][1]
        st.markdown(f"### Tahmin: {'❌ Ölecek' if pred==1 else '✅ Yaşayacak'} (%{proba*100:.1f})")
        explainer = st.session_state["explainer"]
        shap_val = explainer(inp)

        if len(shap_val.values.shape) == 3:
            shap_array = shap_val.values[0, :, 1]
        else:
            shap_array = shap_val.values[0]

        shap_df = pd.DataFrame({
            "Özellik": inp.columns,
            "Değer": inp.values[0],
            "SHAP Etkisi": shap_array
        })

        user_id = "demo_user"  

        
        tahmin_id = kaydet_tahmin(
            user_id=user_id,
            tahmin=int(pred),
            olasilik=float(proba),
            model_adi=st.session_state["model_adi"],
            input_dict=inp.to_dict(orient="records")[0],
            shap_df=shap_df
        )

        
        kaydet_shap_degerleri(tahmin_id, shap_df)

        st.success("✅ Tahmin ve SHAP değerleri başarıyla kaydedildi!")
        if pred == 1 and proba > 0.8:
            st.error("⚠️ Risk çok yüksek!")

with tabs[2]:
    st.header("🧠 SHAP Açıklama")
    st.subheader(f"Ortalama Özellik Etkileri - {st.session_state['model_adi']}")
    st.bar_chart(st.session_state["mean_shap"].set_index("feature"))


