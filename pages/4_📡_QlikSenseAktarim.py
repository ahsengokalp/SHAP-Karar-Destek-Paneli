import streamlit as st
import pandas as pd
import requests
import io
from database.db_config import get_supabase_client

st.set_page_config(page_title="📡 QlikSense Veri Aktarımı", layout="wide")
st.title("📡 QlikSense Entegrasyonu ve Veri Aktarımı")

supabase = get_supabase_client()
user_id = "demo_user"

col1, col2 = st.columns(2)
with col1:
    qlik_url_tahmin = st.text_input("🔗 Tahmin Webhook URL", placeholder="https://.../tahmin")
with col2:
    qlik_url_shap = st.text_input("🔗 SHAP Webhook URL", placeholder="https://.../shap")

with st.spinner("🔄 Veriler Supabase'ten yükleniyor..."):
    tahminler = supabase.table("tahmin_kayitlari").select("*").eq("user_id", user_id).execute().data
    shaplar = supabase.table("shap_kayitlari").select("*").execute().data

if tahminler:
    df_tahmin = pd.DataFrame(tahminler)
    df_shap = pd.DataFrame(shaplar)

    
    veri_seti_listesi = df_tahmin["veri_seti_adi"].dropna().unique().tolist()
    if not veri_seti_listesi:
        veri_seti_listesi = ["Bilinmeyen"]

    secilen_set = st.selectbox("📁 İncelenecek Veri Setini Seçin", options=veri_seti_listesi)

    tahmin_df = df_tahmin[df_tahmin["veri_seti_adi"] == secilen_set]
    shap_df = df_shap[df_shap["tahmin_id"].isin(tahmin_df["id"])]

    st.markdown(f"### 📄 {secilen_set} Veri Seti - Tahmin Verileri")
    st.dataframe(tahmin_df)

    tahmin_csv = io.StringIO()
    tahmin_df.to_csv(tahmin_csv, index=False)
    st.download_button(
        label="⬇️ Tahmin CSV indir",
        data=tahmin_csv.getvalue(),
        file_name=f"{secilen_set}_tahmin.csv",
        mime="text/csv"
    )

    st.markdown("### 🧮 SHAP Değerleri")
    st.dataframe(shap_df)

    shap_csv = io.StringIO()
    shap_df.to_csv(shap_csv, index=False)
    st.download_button(
        label="⬇️ SHAP CSV indir",
        data=shap_csv.getvalue(),
        file_name=f"{secilen_set}_shap.csv",
        mime="text/csv"
    )

    if st.button(f"📤 {secilen_set} verilerini QlikSense'e gönder"):
        try:
            if qlik_url_tahmin:
                res1 = requests.post(qlik_url_tahmin, json=tahmin_df.to_dict(orient="records"))
                st.success(f"✅ Tahmin gönderimi ({secilen_set}): {res1.status_code}")

            if qlik_url_shap:
                res2 = requests.post(qlik_url_shap, json=shap_df.to_dict(orient="records"))
                st.success(f"✅ SHAP gönderimi ({secilen_set}): {res2.status_code}")
        except Exception as e:
            st.error(f"❌ Gönderim hatası: {e}")
else:
    st.warning("Henüz gönderilecek veri yok.")
