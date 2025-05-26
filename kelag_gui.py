import pandas as pd
from sentence_transformers import SentenceTransformer, util
import streamlit as st

st.set_page_config(page_title="Konten-Matching international", layout="centered")
st.title("Automatisiertes Matching: Lokale Konten auf Konzernkonten")

# 1. Sprache auswählen
sprache = st.selectbox(
    "Welche Sprache verwendet dein Upload-Sheet?",
    ["Kroatisch", "Italienisch", "Slowenisch", "Bosnisch", "Rumänisch", "Bulgarisch", "Tschechisch", "Spanisch", "Französisch", "Portugiesisch"]
)

st.info("Bitte lade das Excel mit den lokalen Konten hoch. Erwartete Spalten: 'lokale Kontonummer', 'lokale Kontobeschreibung', 'alte Positionsnummer'.")

# 2. Excel-Upload
uploaded_file = st.file_uploader("Lade dein Excel mit den lokalen Konten hoch", type=["xlsx"])

# 3. Excel-Datenbasis laden
datenbasis_pfad = "Konzernkontenplan_template.xlsx"  # Passe ggf. an
datenbasis = pd.read_excel(datenbasis_pfad)

# Modell laden (multilingual!)
modell_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
@st.cache_resource
def lade_modell():
    return SentenceTransformer(modell_name)
modell = lade_modell()

if uploaded_file is not None:
    try:
        lokalkonten = pd.read_excel(uploaded_file)
        # Überprüfen, ob alle nötigen Spalten vorhanden sind
        spalten = ["lokale Kontonummer", "lokale Kontobeschreibung", "alte Positionsnummer"]
        if not all([sp in lokalkonten.columns for sp in spalten]):
            st.error(f"Fehlende Spalten im Upload. Erwartet: {spalten}")
        else:
            # Ergebnis-DataFrame vorbereiten
            ergebnis_liste = []
            # Über alle lokalen Konten iterieren
            for idx, row in lokalkonten.iterrows():
                posnummer_alt = row["alte Positionsnummer"]
                beschreibung_lokal = str(row["lokale Kontobeschreibung"])
                kontonummer_lokal = row["lokale Kontonummer"]

                # Datenbasis auf diese alte Positionsnummer filtern (alt)
                datenbasis_match = datenbasis[datenbasis.iloc[:,0] == posnummer_alt].copy()
                if datenbasis_match.empty:
                    bester_treffer = {
                        "lokale Kontonummer": kontonummer_lokal,
                        "alte Positionsnummer": posnummer_alt,
                        f"lokale Kontobeschreibung ({sprache})": beschreibung_lokal,
                        "Sachkontonummer neu": "",
                        "Sachkontobezeichnung neu": "",
                        "Beschreibung neu": "",
                        "Score": 0.0,
                        "Info": "Keine passende Position in Datenbasis gefunden"
                    }
                else:
                    # Vergleichstext der Datenbasis-Konten bauen (für das semantische Matching)
                    def kombi(row_db):
                        teile = [
                            str(row_db.get("Kontenbezeichnung", "")),
                            str(row_db.get("Beschreibung", "")),
                            f"Positiv: {row_db.get('Positiv', '')}",
                            f"Negativ: {row_db.get('Negativ', '')}",
                        ]
                        return " ".join([t for t in teile if t and str(t).lower() != 'nan'])
                    datenbasis_match["vergleichstext"] = datenbasis_match.apply(kombi, axis=1)

                    # Multilingualen Vergleich: lokale Beschreibung (in Sprache) vs. Konzern-Kontenbeschreibung (deutsch/englisch/mehrsprachig)
                    datenbasis_embeddings = modell.encode(datenbasis_match["vergleichstext"].tolist(), convert_to_tensor=True)
                    eingabe_embedding = modell.encode([beschreibung_lokal], convert_to_tensor=True)
                    scores = util.pytorch_cos_sim(eingabe_embedding, datenbasis_embeddings)[0]
                    best_idx = scores.argmax().item()
                    bester_score = float(scores[best_idx])

                    # Schreibe Ergebnis
                    bester_treffer = {
                        "lokale Kontonummer": kontonummer_lokal,
                        "alte Positionsnummer": posnummer_alt,
                        f"lokale Kontobeschreibung ({sprache})": beschreibung_lokal,
                        "Sachkontonummer neu": datenbasis_match.iloc[best_idx]["Sachkontonummer"],
                        "Sachkontobezeichnung neu": datenbasis_match.iloc[best_idx]["Kontenbezeichnung"],
                        "Beschreibung neu": datenbasis_match.iloc[best_idx]["Beschreibung"],
                        "Score": round(bester_score, 2),
                        "Info": ""
                    }
                ergebnis_liste.append(bester_treffer)

            ergebnis_df = pd.DataFrame(ergebnis_liste)
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
