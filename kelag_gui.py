import pandas as pd
from sentence_transformers import SentenceTransformer, util
import streamlit as st

st.set_page_config(page_title="Kelag Kontenplan Matching", layout="centered")
st.title("Kelag Konzernkontenplan: Sachkonto-Mapping alt auf neu")

# 1. Excel einlesen
excel_pfad = "Konzernkontenplan_template.xlsx"
df = pd.read_excel(excel_pfad)

# Modell laden (stärkere Variante, mpnet!)
modell_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
@st.cache_resource
def lade_modell():
    return SentenceTransformer(modell_name)
modell = lade_modell()

# --- GUI Inputs ---
konto_art = st.radio(
    "Art des neuen Sachkontos",
    ["Bilanz", "GuV"]
)

untertyp = ""
guv_untertyp = ""
if konto_art == "Bilanz":
    untertyp = st.selectbox(
        "Bilanz-Untertyp auswählen",
        ["Aktiv", "Passiv EK", "Passiv FK"]
    )
elif konto_art == "GuV":
    guv_untertyp = st.selectbox(
        "GuV-Untertyp auswählen",
        ["Ertrag", "Aufwand", "Finanzergebnis", "Ertragsteuerung"]
    )

eingabe_bezeichnung = st.text_input("Bezeichnung des neuen Sachkontos")
eingabe_beschreibung = st.text_area("Beschreibung des neuen Sachkontos", height=100)

if st.button("Sachkonto-Vorschläge berechnen"):
    # 2. Filter nach Kontenart und Untertyp
    if konto_art == "GuV":
        if guv_untertyp == "Ertrag":
            startziffern = ('6',)
            df_filtered = df[df["Sachkontonummer"].astype(str).str.startswith(startziffern)]
            konto_info = "GuV - Ertrag"
        elif guv_untertyp == "Aufwand":
            startziffern = ('7',)
            df_filtered = df[df["Sachkontonummer"].astype(str).str.startswith(startziffern)]
            konto_info = "GuV - Aufwand"
        elif guv_untertyp == "Finanzergebnis":
            startziffern = tuple(str(i) for i in range(80, 86))
            df_filtered = df[df["Sachkontonummer"].astype(str).str[:2].isin(startziffern)]
            konto_info = "GuV - Finanzergebnis"
        elif guv_untertyp == "Ertragsteuerung":
            df_filtered = df[df["Sachkontonummer"].astype(str).str[:2] == '87']
            konto_info = "GuV - Ertragsteuerung"
        else:
            startziffern = ('6', '7', '8', '9')
            df_filtered = df[df["Sachkontonummer"].astype(str).str.startswith(startziffern)]
            konto_info = "GuV"
    else:  # Bilanz
        if untertyp == "Aktiv":
            startziffern = ('1', '2')
            df_filtered = df[df["Sachkontonummer"].astype(str).str.startswith(startziffern)]
            konto_info = "Bilanz - Aktiv"
        elif untertyp == "Passiv EK":
            startziffern = ('3',)
            df_filtered = df[df["Sachkontonummer"].astype(str).str.startswith(startziffern)]
            konto_info = "Bilanz - Passiv EK"
        elif untertyp == "Passiv FK":
            startziffern = ('4', '5')
            df_filtered = df[df["Sachkontonummer"].astype(str).str.startswith(startziffern)]
            konto_info = "Bilanz - Passiv FK"
        else:
            startziffern = ('1', '2', '3', '4', '5')
            df_filtered = df[df["Sachkontonummer"].astype(str).str.startswith(startziffern)]
            konto_info = "Bilanz"

    # 3. Vergleichstext bauen
    def kombiniere_textzeile(row):
        teile = [
            str(row.get("Kontenbezeichnung", "")),
            str(row.get("Beschreibung", "")),
            f"Positiv: {row.get('Positiv', '')}",
            f"Negativ: {row.get('Negativ', '')}",
        ]
        return " ".join([t for t in teile if t and str(t).lower() != 'nan'])
    df_filtered["vergleichstext"] = df_filtered.apply(kombiniere_textzeile, axis=1)

    # 4. Embeddings
    alle_embeddings = modell.encode(df_filtered["vergleichstext"].tolist(), convert_to_tensor=True)
    eingabe_text = f"{eingabe_bezeichnung} {eingabe_beschreibung}"
    eingabe_embedding = modell.encode([eingabe_text], convert_to_tensor=True)

    # 5. Score-Berechnung & Logik
    aehnlichkeit = util.pytorch_cos_sim(eingabe_embedding, alle_embeddings)[0]
    alle_scores = [(i, float(aehnlichkeit[i])) for i in range(len(aehnlichkeit))]
    # Alle mit Score >= 0.50
    relevante = [x for x in alle_scores if x[1] >= 0.50]
    relevante = sorted(relevante, key=lambda x: x[1], reverse=True)
    # Wenn weniger als 5, ergänze weitere (mit den höchsten Werten, aber <0.5), aber NICHT beschränken wenn mehr!
    if len(relevante) < 5:
        rest = [x for x in alle_scores if x not in relevante]
        rest_sorted = sorted(rest, key=lambda x: x[1], reverse=True)
        relevante += rest_sorted[:max(0, 5-len(relevante))]
    # KEINE Begrenzung auf 5! Wenn mehr als 5 mit Score>0.5, alle anzeigen!

    if not relevante:
        st.error("Es konnten keine ähnlichen Sachkonten gefunden werden.")
    else:
        treffer = []
        for idx, score in relevante:
            treffer.append({
                "Score": round(score, 2),
                "Sachkontonummer": df_filtered.iloc[idx]['Sachkontonummer'],
                "Kontenbezeichnung": df_filtered.iloc[idx]['Kontenbezeichnung'],
                "Beschreibung": df_filtered.iloc[idx]['Beschreibung'],
                "Positiv": df_filtered.iloc[idx]['Positiv'],
                "Negativ": df_filtered.iloc[idx]['Negativ'],
                "Position neu": df_filtered.iloc[idx]['Position neu'],
                "Positionsbeschreibung neu": df_filtered.iloc[idx]['Positionsbeschreibung neu'],
            })
        st.success(f"Es werden {len(treffer)} Sachkonten angezeigt. (Alle mit Score >50%, falls weniger als 5, werden die besten weiteren ergänzt.)")
        st.dataframe(pd.DataFrame(treffer), hide_index=True)

        # Download-Link für Excel
        output_path = "Matching_Ergebnis_offline.xlsx"
        pd.DataFrame(treffer).to_excel(output_path, index=False)
        with open(output_path, "rb") as f:
            st.download_button("Ergebnis als Excel herunterladen", f, file_name=output_path)
else:
    st.info("Bitte Bezeichnung und Beschreibung eingeben und auf 'Sachkonto-Vorschläge berechnen' klicken.")
