import streamlit as st
import pandas as pd
from database.db_config import get_supabase_client
from utils.shap_utils import yorumla_shap


st.set_page_config(page_title="📂 Geçmiş Tahminler", layout="wide")
st.title("📂 Geçmiş Tahmin Kayıtları")

supabase = get_supabase_client()
user_id = "demo_user"  


with st.spinner("Tahminler yükleniyor..."):
    response = supabase.table("tahmin_kayitlari").select("*").eq("user_id", user_id).order("tahmin_tarihi", desc=True).execute()

if response.data:
    tahmin_df = pd.DataFrame(response.data)

    giris_tipi_gruplari = tahmin_df.groupby("giris_tipi")

    for giris_tipi, grup_df in giris_tipi_gruplari:
        st.subheader("📌 Tahmin Türü: " + ("Veri Seti ile" if giris_tipi == "csv" else "Manuel Giriş"))

        for _, row in grup_df.iterrows():
            veri_seti_bilgi = f" | Veri Seti: {row['veri_seti_adi']}" if row.get("veri_seti_adi") else ""
            with st.expander(f"🧠 {row['model_adi']} | {row['tahmin_tarihi'][:19]} | Tahmin: {row['tahmin']} (%{row['olasilik']*100:.1f}){veri_seti_bilgi}"):

                shap_resp = supabase.table("shap_kayitlari").select("*").eq("tahmin_id", row["id"]).execute()
                shap_data = shap_resp.data
                if shap_data:
                    shap_df = pd.DataFrame(shap_data)
                    shap_df = shap_df[["ozellik", "deger", "shap_etikisi"]].rename(columns={
                        "ozellik": "Özellik", "deger": "Değer", "shap_etikisi": "SHAP Etkisi"
                    }).sort_values("SHAP Etkisi", key=abs, ascending=False)
                    st.dataframe(shap_df)
                    st.markdown("#### 🗣️ Model Kararı Açıklaması")
                    st.markdown(yorumla_shap(shap_df))
                else:
                    st.warning("SHAP değerleri bulunamadı.")
else:
    st.info("Henüz tahmin yapılmamış.")
