

def yorumla_shap(shap_df):
    etkili = shap_df.sort_values("SHAP Etkisi", key=abs, ascending=False).head(3)
    yorumlar = []
    for _, row in etkili.iterrows():
        ozellik = row["Özellik"]
        deger = row["Değer"]
        etki = row["SHAP Etkisi"]
        yon = "artırdı" if etki > 0 else "azalttı"
        yorum = f"- **{ozellik}** özelliği ({deger:.2f}) tahmini {yon} (SHAP: {etki:.4f})"
        yorumlar.append(yorum)
    return (
        f"🔎 Modelin verdiği karara en çok etki eden 3 özellik:\n\n"
        + "\n".join(yorumlar)
    )
