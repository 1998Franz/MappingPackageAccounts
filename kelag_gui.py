import pandas as pd
from sentence_transformers import SentenceTransformer, util
import streamlit as st
from transformers import pipeline
import io

st.set_page_config(page_title="Konten-Matching international", layout="centered")
st.title("Automatisiertes Matching: Lokale Konten auf Konzernkonten")

# --- Sprache auswählen ---
sprachen_map = {
    "Kroatisch": "hr",
    "Italienisch": "it",
    "Slowenisch": "sl",
    "Bosnisch": "bs",
    "Rumänisch": "ro",
    "Bulgarisch": "bg",
    "Tschechisch": "cs",
    "Spanisch": "es",
    "Französisch": "fr",
    "Portugiesisch": "pt"
}
sprache = st.selectbox(
    "Welche Sprache verwendet dein Upload-Sheet?",
    list(sprachen_map.keys())
)
lang_code = sprachen_map[sprache]

st.info(
    "Bitte lade das Excel mit den lokalen Konten hoch. "
    "Erwartete Spalten: 'lokale Kontonummer', 'lokale Kontobezeichnung', 'alte Positionsnummer'."
)
uploaded_file = st.file_uploader("Lade dein Excel mit den lokalen Konten hoch", type=["xlsx"])

# --- Datenbasis laden (cached, weil meist gleich) ---
@st.cache_data
def lade_datenbasis(pfad):
    return pd.read_excel(pfad)

datenbasis_pfad = "Konzernkontenplan_template.xlsx"
datenbasis = lade_datenbasis(datenbasis_pfad)

# --- Modell laden (cached, schnell) ---
@st.cache_resource
def lade_modell(name):
    return SentenceTransformer(name)

modell = lade_modell("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")

# --- Übersetzer laden (cached, nur falls nicht deutsch) ---
@st.cache_resource
def get_translator(src_lang):
    if src_lang == "de":
        return None
    model_name = f"Helsinki-NLP/opus-mt-{src_lang}-de"
    return pipeline("translation", model=model_name)

# --- Matching-Funktion ---
def match_konten(lokalkonten, datenbasis, modell, translator, sprache):
    ergebnis_liste = []
    n = len(lokalkonten)
    for idx, row in lokalkonten.iterrows():
        posnummer_alt = row["alte Positionsnummer"]
        bezeichnung_lokal = str(row["lokale Kontobezeichnung"])
        kontonummer_lokal = row["lokale Kontonummer"]

        # Übersetzung ins Deutsche (für bessere Scores)
        if translator:
            try:
                bezeichnung_de = translator(bezeichnung_lokal, max_length=128)[0]['translation_text']
            except Exception:
                bezeichnung_de = "(Übersetzung nicht möglich)"
        else:
            bezeichnung_de = bezeichnung_lokal

        datenbasis_match = datenbasis[datenbasis.iloc[:, 0] == posnummer_alt]
        if datenbasis_match.empty:
            ergebnis_liste.append({
                "lokale Kontonummer": kontonummer_lokal,
                "alte Positionsnummer": posnummer_alt,
                f"lokale Kontobezeichnung ({sprache})": bezeichnung_lokal,
                "Lokale Sachkontobezeichnung (deutsch)": bezeichnung_de,
                "Sachkontonummer neu": "",
                "Sachkontobezeichnung neu": "",
                "Beschreibung neu": "",
                "Score": 0.0,
                "Info": "Keine passende Position in Datenbasis gefunden"
            })
            continue

        def kombi(row_db):
            teile = [
                str(row_db.get("Kontenbezeichnung", "")),
                str(row_db.get("Beschreibung", "")),
                f"Positiv: {row_db.get('Positiv', '')}",
                f"Negativ: {row_db.get('Negativ', '')}",
            ]
            return " ".join([t for t in teile if t and str(t).lower() != 'nan'])

        vergleichstexte = datenbasis_match.apply(kombi, axis=1).tolist()
        datenbasis_embeddings = modell.encode(vergleichstexte, convert_to_tensor=True)
        eingabe_embedding = modell.encode([bezeichnung_de], convert_to_tensor=True)
        scores = util.pytorch_cos_sim(eingabe_embedding, datenbasis_embeddings)[0]
        best_idx = scores.argmax().item()
        bester_score = float(scores[best_idx])

        bester_treffer = {
            "lokale Kontonummer": kontonummer_lokal,
            "alte Positionsnummer": posnummer_alt,
            f"lokale Kontobezeichnung ({sprache})": bezeichnung_lokal,
            "Lokale Sachkontobezeichnung (deutsch)": bezeichnung_de,
            "Sachkontonummer neu": datenbasis_match.iloc[best_idx]["Sachkontonummer"],
            "Sachkontobezeichnung neu": datenbasis_match.iloc[best_idx]["Kontenbezeichnung"],
            "Beschreibung neu": datenbasis_match.iloc[best_idx]["Beschreibung"],
            "Score": round(bester_score, 2),
            "Info": ""
        }
        ergebnis_liste.append(bester_treffer)

        if idx % 10 == 0 or idx == n-1:
            st.progress((idx + 1) / n, f"Matching: {idx+1}/{n}")

    return pd.DataFrame(ergebnis_liste)

# --- Main Workflow ---
if uploaded_file:
    try:
        lokalkonten = pd.read_excel(uploaded_file)
        spalten = ["lokale Kontonummer", "lokale Kontobezeichnung", "alte Positionsnummer"]
        if not all(sp in lokalkonten.columns for sp in spalten):
            st.error(f"Fehlende Spalten im Upload. Erwartet: {spalten}")
        else:
            translator = get_translator(lang_code)
            with st.spinner("Matching läuft ..."):
                ergebnis_df = match_konten(lokalkonten, datenbasis, modell, translator, sprache)
            st.success(f"Fertig! Es wurden {len(ergebnis_df)} Konten gematcht.")
            st.dataframe(ergebnis_df, hide_index=True)

            # Download als Excel
            output = io.BytesIO()
            ergebnis_df.to_excel(output, index=False)
            st.download_button(
                "Ergebnis als Excel herunterladen",
                data=output.getvalue(),
                file_name=f"Konten_Matching_Ergebnis_{sprache}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    except Exception as e:
        st.error(f"Fehler beim Verarbeiten: {e}")
else:
    st.info("Bitte eine Datei hochladen.")
