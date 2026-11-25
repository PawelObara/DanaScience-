from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from rdkit import Chem
from rdkit.Chem import AllChem, Draw

# =====================================================================
# Aplikacja Streamlit: baza pochodnych emitera TADF (wybrane związki po
# usunięciu zbyt podobnych) + rozkłady deskryptorów.
# Plik bazy (Baza_Streamlit_Ready.csv) musi leżeć w tym samym folderze co Main.py.
# =====================================================================

st.set_page_config(page_title="Baza emiterów TADF", layout="wide")

PLIK_BAZY = Path(__file__).parent / "Baza_Streamlit_Ready.csv"

PODSTAWNIKI = {"1": "-CH3", "2": "-F", "3": "-Cl", "4": "-Br", "5": "-I"}
OPIS_LM = {
    "LM1": "1 podstawnik, strona donora",
    "LM2": "1 podstawnik, strona akceptora",
    "LM3": "2 podstawniki, strona donora",
    "LM4": "2 podstawniki, strona akceptora",
}
OPIS_AM = {"AM1": "pozycje orto", "AM2": "pozycje meta", "AM3": "pozycje para"}
OPIS_DONOR = {"D1": "DMAC", "D2": "CBZ (karbazol)", "D3": "TPA"}

OPIS_DESKRYPTOROW = {
    "Hal": "liczba atomów halogenów",
    "MolWt": "masa molowa [g/mol]",
    "HAC": "liczba atomów ciężkich",
    "HA_MolWt": "masa atomów ciężkich [g/mol]",
    "ASA": "powierzchnia cząsteczki (Labute ASA)",
    "Val": "indeks connectivity Chi1v",
    "ASF": "asferyczność (z geometrii 3D)",
}
GRUPY = ["Donor", "Mod_Donor", "Sub_Donor", "Mod_Linker", "Sub_Linker", "Mod_Akceptor", "Sub_Akceptor"]


def opis_podstawnika(kod):
    """DR1 -> 'DR1 (-CH3)' itd."""
    return f"{kod} ({PODSTAWNIKI.get(kod[-1], '?')})"


@st.cache_data
def wczytaj_dane():
    return pd.read_csv(PLIK_BAZY)


@st.cache_data(show_spinner="Liczę geometrię 3D...")
def geometria_xyz(smiles):
    """Wstępna geometria 3D (ETKDG + MMFF) policzona na żądanie ze SMILES-a."""
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    if AllChem.EmbedMolecule(mol, params) < 0:
        params.useRandomCoords = True
        if AllChem.EmbedMolecule(mol, params) < 0:
            return None
    AllChem.MMFFOptimizeMolecule(mol, maxIters=2000)
    return Chem.MolToXYZBlock(mol)


def widok_3d(xyz, wysokosc=450):
    """Interaktywny podgląd 3D (biblioteka 3Dmol.js ładowana z internetu)."""
    html = f"""
    <div id="viewer" style="width:100%;height:{wysokosc}px;position:relative;"></div>
    <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
    <script>
      let v = $3Dmol.createViewer("viewer", {{backgroundColor: "white"}});
      v.addModel(`{xyz}`, "xyz");
      v.setStyle({{}}, {{stick: {{radius: 0.15}}, sphere: {{scale: 0.25}}}});
      v.zoomTo();
      v.render();
    </script>
    """
    components.html(html, height=wysokosc + 10)


df = wczytaj_dane()
DESKRYPTORY = [d for d in OPIS_DESKRYPTOROW if d in df.columns]

st.title("Baza pochodnych emitera TADF")
st.caption(
    "Układy donor – linker – akceptor (DMAC/CBZ/TPA – fenylen – difenylotriazyna) "
    "z modyfikacjami podstawnikowymi. Pokazane są związki wybrane po usunięciu zbyt podobnych "
    "(odcisk Morgana, podobieństwo Tanimoto). Geometria 3D to wstępny wynik MMFF (nie DFT)."
)

# ----------------------------- FILTRY ---------------------------------
st.sidebar.header("Filtry")


def filtr(etykieta, kolumna, opisy=None, podstawnik=False):
    opcje = sorted(df[kolumna].unique())
    if podstawnik:
        formatuj = opis_podstawnika
    elif opisy:
        formatuj = lambda x: f"{x} ({opisy[x]})" if x in opisy else x
    else:
        formatuj = lambda x: x
    return st.sidebar.multiselect(etykieta, opcje, default=opcje, format_func=formatuj)


wybrane_filtry = {
    "Donor": filtr("Donor", "Donor", OPIS_DONOR),
    "Mod_Donor": filtr("Modyfikacja donora", "Mod_Donor"),
    "Sub_Donor": filtr("Podstawnik donora", "Sub_Donor", podstawnik=True),
    "Mod_Linker": filtr("Modyfikacja linkera", "Mod_Linker", OPIS_LM),
    "Sub_Linker": filtr("Podstawnik linkera", "Sub_Linker", podstawnik=True),
    "Mod_Akceptor": filtr("Modyfikacja akceptora", "Mod_Akceptor", OPIS_AM),
    "Sub_Akceptor": filtr("Podstawnik akceptora", "Sub_Akceptor", podstawnik=True),
}

maska = pd.Series(True, index=df.index)
for kolumna, wartosci in wybrane_filtry.items():
    maska &= df[kolumna].isin(wartosci)

przefiltrowane = df[maska].reset_index(drop=True)

tab_baza, tab_desc, tab_info = st.tabs(["Baza cząsteczek", "Rozkład deskryptorów", "O projekcie"])

if przefiltrowane.empty:
    st.warning("Żadna cząsteczka nie spełnia wybranych filtrów.")
    st.stop()

# ----------------------------- ZAKŁADKA BAZA --------------------------
with tab_baza:
    st.subheader(f"Cząsteczki: {len(przefiltrowane)} z {len(df)}")

    kolumny_tabeli = [
        c for c in (
            ["Full_name", "Donor", "Mod_Donor", "Sub_Donor", "Mod_Linker", "Sub_Linker",
             "Mod_Akceptor", "Sub_Akceptor", "Wzor"] + DESKRYPTORY
        ) if c in przefiltrowane.columns
    ]
    widok = przefiltrowane[kolumny_tabeli]

    st.write("Zaznacz kwadracik przy wierszu, aby zobaczyć strukturę (domyślnie pierwszy wiersz).")
    wybrane_wiersze = []
    try:
        zdarzenie = st.dataframe(
            widok, on_select="rerun", selection_mode="single-row", hide_index=True
        )
        wybrane_wiersze = zdarzenie.selection.rows
    except TypeError:
        # starsza wersja Streamlita bez zaznaczania wierszy: wybór z listy
        st.dataframe(widok, hide_index=True)
        nazwa_wybrana = st.selectbox("Wybierz cząsteczkę", widok["Full_name"].head(500))
        wybrane_wiersze = [int(widok.index[widok["Full_name"] == nazwa_wybrana][0])]

    wiersz = przefiltrowane.iloc[wybrane_wiersze[0]] if wybrane_wiersze else przefiltrowane.iloc[0]

    st.download_button(
        "Pobierz przefiltrowaną tabelę (CSV)",
        data=widok.to_csv(index=False).encode("utf-8"),
        file_name="Baza_TADF_przefiltrowana.csv",
        mime="text/csv",
    )

    st.divider()
    st.subheader(wiersz["Full_name"])
    kol_2d, kol_3d = st.columns(2)

    mol = Chem.MolFromSmiles(wiersz["SMILES"])
    with kol_2d:
        st.image(Draw.MolToImage(mol, size=(600, 450)))
        st.markdown(
            f"**Wzór:** {wiersz['Wzor']}  \n"
            f"**Masa molowa:** {wiersz['Masa_Molowa']} g/mol  \n"
            f"**SMILES:** `{wiersz['SMILES']}`"
        )

    with kol_3d:
        if st.checkbox("Pokaż strukturę 3D", value=True):
            xyz = geometria_xyz(wiersz["SMILES"])
            if xyz is None:
                st.error("Nie udało się policzyć geometrii 3D dla tej cząsteczki.")
            else:
                widok_3d(xyz)
                st.download_button(
                    "Pobierz geometrię (XYZ)",
                    data=xyz,
                    file_name=f"{wiersz['Full_name']}.xyz",
                    mime="chemical/x-xyz",
                )

# ----------------------------- ZAKŁADKA DESKRYPTORY -------------------
with tab_desc:
    st.subheader("Rozkład deskryptorów")
    st.caption(
        f"Dane: {len(przefiltrowane)} cząsteczek po zastosowaniu filtrów z lewego panelu "
        "(żeby zobaczyć cały zbiór, zaznacz wszystkie opcje w filtrach)."
    )

    if not DESKRYPTORY:
        st.info("W pliku bazy nie ma kolumn z deskryptorami.")
    else:
        def opis_deskryptora(d):
            return f"{d} – {OPIS_DESKRYPTOROW[d]}"

        # --- histogram ---
        st.markdown("#### Histogram")
        k1, k2, k3 = st.columns(3)
        deskryptor = k1.selectbox("Deskryptor", DESKRYPTORY, format_func=opis_deskryptora)
        grupa = k2.selectbox("Koloruj według", ["brak"] + GRUPY)
        liczba_binow = k3.slider("Liczba przedziałów", 5, 100, 30)

        kolor = alt.Color(f"{grupa}:N") if grupa != "brak" else alt.value("#4c78a8")
        histogram = (
            alt.Chart(przefiltrowane)
            .mark_bar()
            .encode(
                x=alt.X(f"{deskryptor}:Q", bin=alt.Bin(maxbins=liczba_binow), title=opis_deskryptora(deskryptor)),
                y=alt.Y("count()", title="Liczba związków"),
                color=kolor,
            )
            .properties(height=350)
        )
        st.altair_chart(histogram)

        st.markdown("#### Statystyki opisowe")
        st.dataframe(przefiltrowane[DESKRYPTORY].describe().T.round(3))

        # --- wykres rozrzutu ---
        st.markdown("#### Zależność dwóch deskryptorów")
        s1, s2, s3 = st.columns(3)
        os_x = s1.selectbox("Oś X", DESKRYPTORY, index=0, format_func=opis_deskryptora, key="os_x")
        os_y = s2.selectbox(
            "Oś Y", DESKRYPTORY, index=min(1, len(DESKRYPTORY) - 1), format_func=opis_deskryptora, key="os_y"
        )
        grupa2 = s3.selectbox("Koloruj według", ["brak"] + GRUPY, key="grupa2")

        kolor2 = alt.Color(f"{grupa2}:N") if grupa2 != "brak" else alt.value("#4c78a8")
        rozrzut = (
            alt.Chart(przefiltrowane)
            .mark_circle(size=40, opacity=0.6)
            .encode(
                x=alt.X(f"{os_x}:Q", scale=alt.Scale(zero=False)),
                y=alt.Y(f"{os_y}:Q", scale=alt.Scale(zero=False)),
                color=kolor2,
                tooltip=["Full_name", os_x, os_y],
            )
            .properties(height=400)
            .interactive()
        )
        st.altair_chart(rozrzut)

        # --- korelacje ---
        st.markdown("#### Korelacje (Pearson)")
        if len(przefiltrowane) > 2 and len(DESKRYPTORY) > 1:
            korelacje = przefiltrowane[DESKRYPTORY].corr().stack().reset_index()
            korelacje.columns = ["x", "y", "r"]
            mapa = (
                alt.Chart(korelacje)
                .mark_rect()
                .encode(
                    x=alt.X("x:N", title=None),
                    y=alt.Y("y:N", title=None),
                    color=alt.Color("r:Q", scale=alt.Scale(domain=[-1, 1], scheme="redblue")),
                )
            )
            tekst = (
                alt.Chart(korelacje)
                .mark_text()
                .encode(x="x:N", y="y:N", text=alt.Text("r:Q", format=".2f"))
            )
            st.altair_chart((mapa + tekst).properties(height=350))
        else:
            st.info("Za mało cząsteczek, żeby policzyć korelacje.")

# ----------------------------- ZAKŁADKA INFO --------------------------
with tab_info:
    st.markdown(
        """
### Cel projektu
Screening pochodnych emitera TADF typu donor–akceptor. Szukamy układów o małej przerwie
singlet–tryplet (ΔE_ST) i dużej stałej sprzężenia spin–orbita (SOC), czyli o wydajnym odwrotnym
przejściu międzysystemowym (rISC).

### Nazewnictwo cząsteczek
`D1_DM1_DR1_LM1_LR1_A1_AM1_AR1` =
typ donora (D1–D3), modyfikacja donora (DM1–3), podstawnik donora (DR1–5),
modyfikacja linkera (LM1–4), podstawnik linkera (LR1–5),
typ akceptora (A1), modyfikacja akceptora (AM1–3), podstawnik akceptora (AR1–5).

Podstawniki: 1 = -CH3, 2 = -F, 3 = -Cl, 4 = -Br, 5 = -I.

### Skąd ten zbiór
Pełna biblioteka to 13 500 cząsteczek (3 × 3 × 5 donorów, 1 × 3 × 5 akceptorów, 4 × 5 linkerów).
Z niej wylosowano 2000 związków, a następnie usunięto te, których odcisk (Morgan, promień 2, 1024 bity)
był zbyt podobny do już zachowanych (podobieństwo Tanimoto powyżej progu).

### Deskryptory
Hal – liczba halogenów, MolWt – masa molowa, HAC – liczba atomów ciężkich, HA_MolWt – masa atomów ciężkich,
ASA – powierzchnia Labute, Val – indeks Chi1v, ASF – asferyczność (z geometrii 3D z MMFF).

### Wyniki obliczeń DFT
Miejsce na wyniki (ΔE_ST, SOC) po zakończeniu obliczeń w ORCA.
        """
    )
