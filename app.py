import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os
from src.configs.config import Config


# 1. Ładowanie zapisanego potoku modelu
@st.cache_resource
def load_pipeline():
    return joblib.load('model_data.pkl')


# 2. Ładowanie i agregacja danych z oryginalnego CSV do profili krajów
@st.cache_data
def load_country_profiles():
    file_path = 'data/Life Expectancy Data.csv'
    if not os.path.exists(file_path):
        return pd.DataFrame()

    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()

    numeric_means = df.groupby('Country').mean(numeric_only=True).reset_index()
    status_df = df.groupby('Country')['Status'].first().reset_index()

    profiles = pd.merge(numeric_means, status_df, on='Country')
    return profiles


try:
    pipeline = load_pipeline()
    country_profiles = load_country_profiles()

    st.set_page_config(
        page_title="Monitor Długości Życia WHO",
        page_icon="🌍",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.markdown("""
        <style>
        .block-container { padding-top: 2rem; padding-bottom: 2rem; }
        .stTabs [data-baseweb="tab-list"] { gap: 8px; }
        .stTabs [data-baseweb="tab"] { padding-top: 10px; padding-bottom: 10px; }
        </style>
    """, unsafe_allow_html=True)

    st.title("🌍 Inteligentny System Predykcji Długości Życia")
    st.markdown(
        "System ekspercki oparty na modelu Random Forest, wykorzystujący dane Światowej Organizacji Zdrowia (WHO).")
    st.markdown("---")

    # --- SEKCJA: WYBÓR KRAJU ---
    st.subheader("📍 Autouzupełnianie na podstawie danych historycznych")

    if not country_profiles.empty:
        lista_krajow = ["--- Wybierz kraj, aby załadować średnie dane ---"] + sorted(
            country_profiles['Country'].unique().tolist())
        wybrany_kraj = st.selectbox(
            "Wybierz profil państwa",
            lista_krajow,
            help="Wybór profilu spowoduje automatyczne nadpisanie wszystkich suwaków uśrednionymi danymi historycznymi WHO dla wybranego państwa."
        )
    else:
        st.warning("Nie znaleziono pliku `data/Life Expectancy Data.csv`. Autouzupełnianie jest nieaktywne.")
        wybrany_kraj = "--- Wybierz kraj, aby załadować średnie dane ---"

    # --- SŁOWNIK WARTOŚCI DOMYŚLNYCH ---
    def_vals = {
        "Status": "Developing", "Income composition of resources": 0.65, "HIV/AIDS": 0.1,
        "Adult Mortality": 150, "GDP": 5000.0, "percentage expenditure": 12.0,
        "Population": 2000000, "Total expenditure": 5.5, "infant deaths": 25,
        "under-five deaths": 35, "BMI": 25.0, "Measles": 10, "Alcohol": 4.0,
        "Schooling": 12.0, "Hepatitis B": 90, "thinness  1-19 years": 4.5,
        "Polio": 90, "Diphtheria": 90, "thinness 5-9 years": 4.5
    }

    if wybrany_kraj != "--- Wybierz kraj, aby załadować średnie dane ---":
        profil = country_profiles[country_profiles['Country'] == wybrany_kraj].iloc[0]
        def_vals["Status"] = profil["Status"] if pd.notna(profil["Status"]) else "Developing"
        for key in def_vals.keys():
            if key in profil and pd.notna(profil[key]) and key != "Status":
                def_vals[key] = float(profil[key])

    st.markdown("---")
    col_inputs, col_results = st.columns([2, 1], gap="large")

    with col_inputs:
        st.subheader("🔥 Krytyczne czynniki determinujące")
        st.markdown(
            "<small style='color: gray;'>Parametry o najwyższej wadze decyzyjnej w strukturze algorytmu.</small>",
            unsafe_allow_html=True)

        c_top1, c_top2 = st.columns(2)
        with c_top1:
            income_comp = st.slider(
                "📈 Wskaźnik rozwoju zasobów (Income comp.) [🔥+]", 0.0, 1.0,
                min(1.0, max(0.0, float(def_vals["Income composition of resources"]))), step=0.01,
                help="Wskaźnik HDI (Human Development Index) określający stopień zagospodarowania zasobów ludzkich (edukacja, dochód) w skali od 0 do 1. Im wyższy, tym dłuższa oczekiwana długość życia."
            )
        with c_top2:
            hiv_aids = st.slider(
                "🦠 Zgony na HIV/AIDS (0-4 lata) [⚠️-]", 0.0, 55.0,
                float(min(55.0, max(0.0, def_vals["HIV/AIDS"]))), step=0.1,
                help="Liczba zgonów na 1000 żywych urodzeń wśród dzieci w wieku 0-4 lat spowodowanych zakażeniem HIV/AIDS."
            )
            adult_mortality = st.slider(
                "💀 Śmiertelność dorosłych (na 1000 os.) [⚠️-]", 0, 750,
                int(min(750, max(0, def_vals["Adult Mortality"]))), step=1,
                help="Prawdopodobieństwo zgonu osób w wieku od 15 do 60 lat na 1000 mieszkańców tej populacji."
            )

        st.markdown("---")
        st.subheader("📊 Szczegółowe wskaźniki demograficzne i zdrowotne")
        tab1, tab2, tab3 = st.tabs(["💰 Ekonomia", "🏥 Zagrożenia Zdrowotne", "💉 Profilaktyka & Edukacja"])

        with tab1:
            status_index = 0 if def_vals["Status"] == "Developing" else 1
            status_label = st.radio(
                "Status rozwoju państwa (Status)", ["Kraj Rozwijający się", "Kraj Rozwinięty"],
                index=status_index, horizontal=True,
                help="Klasyfikacja geopolityczna kraju według standardów WHO: Kraj Rozwijający się (Developing) lub Rozwinięty (Developed)."
            )
            status = 0 if status_label == "Kraj Rozwijający się" else 1

            c1, c2 = st.columns(2)
            with c1:
                gdp = st.slider(
                    "💵 PKB per capita (USD)", 0, 120000, int(min(120000, max(0, def_vals["GDP"]))), step=100,
                    help="Produkt Krajowy Brutto w przeliczeniu na jednego mieszkańca, wyrażony w dolarach amerykańskich (USD)."
                )
                percentage_exp = st.slider(
                    "🏥 Wydatki na zdrowie (wsk. %)", 0.0, 100.0,
                    float(min(100.0, max(0.0, def_vals["percentage expenditure"]))), step=0.5,
                    help="Procent PKB per capita wydawany na zdrowie publiczne."
                )
            with c2:
                population = st.slider(
                    "👥 Populacja państwa", 0, 300000000, int(min(300000000, max(0, def_vals["Population"]))), step=1000000,
                    help="Całkowita liczba ludności zamieszkująca dany kraj (skala dostosowana do większości państw na świecie)."
                )
                total_exp = st.slider(
                    "🏛️ Budżet zdrowotny rządu (%)", 0.0, 30.0, float(min(30.0, max(0.0, def_vals["Total expenditure"]))), step=0.1,
                    help="Procentowy udział całkowitych wydatków rządu centralnego przeznaczony na opiekę zdrowotną."
                )

        with tab2:
            c3, c4 = st.columns(2)
            with c3:
                infant_deaths = st.slider(
                    "👶 Zgony niemowląt (wskaźnik)", 0, 100, int(min(100, max(0, def_vals["infant deaths"]))),
                    help="Liczba zgonów dzieci poniżej 1 roku życia w przeliczeniu na 1000 urodzeń."
                )
                under_five_deaths = st.slider(
                    "🧒 Zgony dzieci do lat 5 (wskaźnik)", 0, 150, int(min(150, max(0, def_vals["under-five deaths"]))),
                    help="Liczba zgonów dzieci poniżej 5 roku życia w przeliczeniu na 1000 urodzeń."
                )
                bmi = st.slider(
                    "⚖️ Średni wskaźnik BMI populacji", 10.0, 50.0, float(min(50.0, max(10.0, def_vals["BMI"]))), step=0.1,
                    help="Średnia wartość indeksu masy ciała (Body Mass Index) wyznaczona dla populacji dorosłych mieszkańców danego państwa."
                )
            with c4:
                measles = st.slider(
                    "🤒 Przypadki Odry (Measles)", 0, 5000, int(min(5000, max(0, def_vals["Measles"]))), step=50,
                    help="Liczba oficjalnie zarejestrowanych i zgłoszonych przypadków zachorowań na odrę w danym roku."
                )
                alcohol = st.slider(
                    "🍷 Spożycie czystego alkoholu (litry/os.)", 0.0, 25.0, float(min(25.0, max(0.0, def_vals["Alcohol"]))), step=0.1,
                    help="Roczne spożycie czystego, 100-procentowego alkoholu na jednego mieszkańca w wieku 15 lat i więcej (w litrach)."
                )

        with tab3:
            c5, c6 = st.columns(2)
            with c5:
                schooling = st.slider(
                    "🎓 Średnia liczba lat edukacji", 0.0, 25.0, float(min(25.0, max(0.0, def_vals["Schooling"]))), step=0.1,
                    help="Przeciętna liczba lat edukacji szkolnej oraz wyższej, jaką odbywa obywatel danego kraju."
                )
                hepatitis_b = st.slider(
                    "🛡️ Szczepienia: WZW B (%)", 0, 100, int(min(100, max(0, def_vals["Hepatitis B"]))),
                    help="Poziom wyszczepialności (w procentach) przeciwko wirusowemu zapaleniu wątroby typu B (HepB3) wśród dzieci w wieku do jednego roku."
                )
                thinness_1_19 = st.slider(
                    "📉 Wskaźnik chudości (1-19 lat) %", 0.0, 30.0, float(min(30.0, max(0.0, def_vals["thinness  1-19 years"]))), step=0.1,
                    help="Odsetek dzieci i młodzieży w wieku od 1 do 19 lat, których wskaźnik masy ciała (BMI) znajduje się drastycznie poniżej normy (niedożywienie)."
                )
            with c6:
                polio = st.slider(
                    "🛡️ Szczepienia: Polio (%)", 0, 100, int(min(100, max(0, def_vals["Polio"]))),
                    help="Procentowy odsetek dzieci jednorocznych, które otrzymały pełną dawkę szczepienia przeciwko poliomyelitis (Pol3)."
                )
                diphtheria = st.slider(
                    "🛡️ Szczepienia: Błonnica (%)", 0, 100, int(min(100, max(0, def_vals["Diphtheria"]))),
                    help="Poziom wyszczepialności (w procentach) przeciwko błonnicy, tężcowi i krztuścowi (DTP3) wśród dzieci w wieku do 1 roku."
                )
                thinness_5_9 = st.slider(
                    "📉 Wskaźnik chudości (5-9 lat) %", 0.0, 30.0, float(min(30.0, max(0.0, def_vals["thinness 5-9 years"]))), step=0.1,
                    help="Odsetek dzieci w wieku od 5 do 9 lat cierpiących na skrajną niedowagę spowodowaną niedożywieniem."
                )

    # 3. BEZPIECZNE MAPOWANIE SŁOWNIKOWE
    input_dict = {
        "Status": status, "Adult Mortality": adult_mortality, "infant deaths": infant_deaths,
        "Alcohol": alcohol, "percentage expenditure": percentage_exp, "Hepatitis B": hepatitis_b,
        "Measles": measles, "BMI": bmi, "under-five deaths": under_five_deaths, "Polio": polio,
        "Total expenditure": total_exp, "Diphtheria": diphtheria, "HIV/AIDS": hiv_aids,
        "GDP": gdp, "Population": population, "thinness  1-19 years": thinness_1_19,
        "thinness 5-9 years": thinness_5_9, "Income composition of resources": income_comp,
        "Schooling": schooling
    }

    input_data = pd.DataFrame([input_dict])[pipeline.features]

    # 4. SILNIK PRZETWARZANIA
    X_imp = pd.DataFrame(pipeline.prep.imputer.transform(input_data), columns=pipeline.features)

    for col in pipeline.features:
        if col != "Status" and X_imp[col].max() > Config.SKW_LIMIT:
            X_imp[col] = np.log1p(X_imp[col].clip(lower=0))

    X_std = pipeline.prep.scaler.transform(X_imp)
    X_std_df = pd.DataFrame(X_std, columns=pipeline.features)

    X_final = X_std_df[pipeline.remaining].values

    # 5. PREDYKCJA
    predicted_value = pipeline.best_model.predict(X_final)[0]

    # 6. PANEL WYNIKOWY
    with col_results:
        st.markdown("### 🎯 Wynik Obliczeń Modelu")

        st.markdown(f"""
            <div style="background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 100%); padding: 30px; border-radius: 12px; color: white; text-align: center; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); margin-bottom: 20px;">
                <span style="text-transform: uppercase; letter-spacing: 0.1em; font-size: 11px; opacity: 0.85;">
                    Oczekiwana długość życia
                </span>
                <h1 style="font-size: 58px; font-weight: 800; margin: 10px 0; color: white; text-shadow: 2px 2px 4px rgba(0,0,0,0.2);">{predicted_value:.2f} lat</h1>
                <div style="padding: 4px 12px; background: rgba(255,255,255,0.15); border-radius: 20px; display: inline-block; font-size: 13px; font-weight: 500;">
                    Algorytm: {pipeline.model_name} (R²: 0.945)
                </div>
            </div>
        """, unsafe_allow_html=True)

        st.info(
            f"✨ **Filtr Współliniowości (VIF):** Model przetwarza zredukowaną matrycę **{len(pipeline.remaining)} z {len(pipeline.features)}** cech.")

        st.markdown("---")
        st.markdown("### 📊 Znaczenie wskaźników (Feature Importance)")

        if hasattr(pipeline.best_model, "feature_importances_"):
            importances = pipeline.best_model.feature_importances_
            importance_df = pd.DataFrame({
                "Wskaźnik": pipeline.remaining,
                "Waga": importances * 100
            }).sort_values(by="Waga", ascending=False)

            st.dataframe(
                importance_df.style.format({"Waga": "{:.2f}%"}).background_gradient(cmap="Blues"),
                hide_index=True,
                use_container_width=True,
                height=350
            )

except FileNotFoundError:
    st.error(
        f"❌ Błąd krytyczny: Upewnij się, że masz wytrenowany model ('model_data.pkl') oraz dane wejściowe ('data/Life Expectancy Data.csv').")