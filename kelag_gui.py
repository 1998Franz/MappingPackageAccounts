import pandas as pd
from sentence_transformers import SentenceTransformer, util
import streamlit as st

# Für Übersetzung
from transformers import pipeline

st.set_page_config(page_title="Konten-Matching international", layout="centered")
st.title("Automatisiertes Matching: Lokale Konten auf Konzernkonten")

# Sprache auswählen
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

st.info("Bitte lade das Excel mit den lokalen Konten hoch. Erwartete Spalten: 'lokale Kontonummer', 'lokale Kontobezeichnung', 'alte Positionsnummer'.")

# Excel-Upload
uploaded_file = st.file_uploader("Lade dein Excel mit den lokalen Konten hoch", type=["xlsx"])

# Excel-Datenbasis laden
datenbasis_pfad = "Konzernkontenplan_template.xlsx"
datenbasis = pd.read_excel(datenbasis_pfad)

# Modell laden (multilingual, semantisch!)
modell_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
@st.cache_resource
def lade_modell():
    return SentenceTransformer(modell_name)
modell = lade_modell()

# Übersetzer laden (cache für Geschwindigkeit)
@st.cache_resource
def get_translator(src_lang):
    if src_lang == "de":
        # Keine Übersetzung notwendig
        return None
    model_name = f"Helsinki-NLP/opus-mt-{src_lang}-de"
    return pipeline("translation", model=model_name)

if uploaded_file is not None:
    try:
        lokalkonten = pd.read_excel(uploaded_file)
        spalten = ["lokale Kontonummer", "lokale Kontobezeichnung", "alte Positionsnummer"]
        if not all([sp in lokalkonten.columns for sp in spalten]):
            st.error(f"Fehlende Spalten im Upload. Erwartet: {spalten}")
        else:
            # Fortschrittsbalken initialisieren
            progress_bar = st.progress(0)
            n = len(lokalkonten)
            ergebnis_liste = []
            translator = get_translator(lang_code)

            for idx, row in lokalkonten.iterrows():
                posnummer_alt = row["alte Positionsnummer"]
                bezeichnung_lokal = str(row["lokale Kontobezeichnung"])
                kontonummer_lokal = row["lokale Kontonummer"]

                # Übersetzung ins Deutsche (für Ausgabe)
                if translator is not None:
                    try:
                        uebersetzung = translator(bezeichnung_lokal, max_length=128)[0]['translation_text']
                    except Exception:
                        uebersetzung = "(Übersetzung nicht möglich)"
                else:
                    uebersetzung = bezeichnung_lokal  # bereits Deutsch

                # Datenbasis auf diese alte Positionsnummer filtern (alt)
                datenbasis_match = datenbasis[datenbasis.iloc[:,0] == posnummer_alt].copy()
                if datenbasis_match.empty:
                    bester_treffer = {
                        "lokale Kontonummer": kontonummer_lokal,
                        "alte Positionsnummer": posnummer_alt,
                        f"lokale Kontobezeichnung ({sprache})": bezeichnung_lokal,
                        "Lokale Sachkontobezeichnung (deutsch)": uebersetzung,
                        "Sachkontonummer neu": "",
                        "Sachkontobezeichnung neu": "",
                        "Beschreibung neu": "",
                        "Score": 0.0,
                        "Info": "Keine passende Position in Datenbasis gefunden"
                    }
                else:
                    # Vergleichstext der Datenbasis-Konten bauen
                    def kombi(row_db):
                        teile = [
                            str(row_db.get("Kontenbezeichnung", "")),
                            str(row_db.get("Beschreibung", "")),
                            f"Positiv: {row_db.get('Positiv', '')}",
                            f"Negativ: {row_db.get('Negativ', '')}",
                        ]
                        return " ".join([t for t in teile if t and str(t).lower() != 'nan'])
                    datenbasis_match["vergleichstext"] = datenbasis_match.apply(kombi, axis=1)

                    # Semantisches Matching: lokale Bezeichnung (übersetzt oder original) vs. Vergleichstext
                    matching_text = bezeichnung_lokal
                    datenbasis_embeddings = modell.encode(datenbasis_match["vergleichstext"].tolist(), convert_to_tensor=True)
                    eingabe_embedding = modell.encode([matching_text], convert_to_tensor=True)
                    scores = util.pytorch_cos_sim(eingabe_embedding, datenbasis_embeddings)[0]
                    best_idx = scores.argmax().item()
                    bester_score = float(scores[best_idx])

                    bester_treffer = {
                        "lokale Kontonummer": kontonummer_lokal,
                        "alte Positionsnummer": posnummer_alt,
                        f"lokale Kontobezeichnung ({sprache})": bezeichnung_lokal,
                        "Lokale Sachkontobezeichnung (deutsch)": uebersetzung,
                        "Sachkontonummer neu": datenbasis_match.iloc[best_idx]["Sachkontonummer"],
                        "Sachkontobezeichnung neu": datenbasis_match.iloc[best_idx]["Kontenbezeichnung"],
                        "Beschreibung neu": datenbasis_match.iloc[best_idx]["Beschreibung"],
                        "Score": round(bester_score, 2),
                        "Info": ""
                    }
                ergebnis_liste.append(bester_treffer)
                progress_bar.progress((idx + 1) / n)

            ergebnis_df = pd.DataFrame(ergebnis_liste)
            progress_bar.empty()  # Fortschrittsbalken ausblenden

            st.success(f"Fertig! Es wurden {len(ergebnis_df)} Konten gematcht.")
            st.dataframe(ergebnis_df, hide_index=True)

            # Download-Option für Ergebnis
            output_excel = ergebnis_df.to_excel(index=False)
            st.download_button(
                label="Ergebnis als Excel herunterladen",
                data=output_excel,
                file_name=f"Konten_Matching_Ergebnis_{sprache}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    except Exception as e:
        st.error(f"Fehler beim Verarbeiten: {e}")

else:
    st.info("Bitte eine Datei hochladen.")
