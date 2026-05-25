import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os
from src.configs.config import Config

# --- KONFIGURACJA STRONY (Musi być jako pierwsza) ---
st.set_page_config(
    page_title="Monitor Długości Życia WHO",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --- 1. ŁADOWANIE ZAPISANEGO POTOKU MODELU ---
@st.cache_resource
def load_pipeline():
    try:
        return joblib.load('model_data.pkl')
    except Exception as e:
        st.error(f"❌ Krytyczny błąd ładowania modelu 'model_data.pkl': {str(e)}")
        return None


# --- 2. ŁADOWANIE I EKSPERCKIE CZYSZCZENIE DANYCH ---
@st.cache_data
def load_country_profiles():
    file_path = 'data/Life Expectancy Data.csv'
    if not os.path.exists(file_path):
        return pd.DataFrame()

    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()

    # --- POPRAWKA: Bezpieczne czyszczenie anomalii bez uszkadzania innych kolumn ---
    # 1. Korekta anomalii BMI (wartości > 38.0 jako średnia populacyjna to błąd zapisu w bazie)
    df['BMI'] = df['BMI'].apply(lambda x: np.nan if x > 38.0 or x < 16.0 else x)

    # 2. Korekta błędu logicznego 'percentage expenditure' (usuwamy tylko abstrakcyjne wartości procentowe > 25%)
    df['percentage expenditure'] = df['percentage expenditure'].apply(lambda x: np.nan if x > 25.0 else x)

    # Agregacja średnich (funkcja mean automatycznie pomija wartości NaN, nie psując kolumny GDP)
    numeric_means = df.groupby('Country').mean(numeric_only=True).reset_index()
    status_df = df.groupby('Country')['Status'].first().reset_index()

    profiles = pd.merge(numeric_means, status_df, on='Country')
    return profiles


pipeline = load_pipeline()
country_profiles = load_country_profiles()

# Custom CSS dla nowoczesnego wyglądu i czcionek zbliżonych do Arial/Sans-serif
st.markdown("""
    <style>
    body { font-family: 'Arial', sans-serif; }
    .block-container { padding-top: 1.5rem; padding-bottom: 1.5rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] { 
        padding: 12px 20px; 
        background-color: #f1f5f9; 
        border-radius: 6px 6px 0px 0px;
        font-weight: 600;
        color: #1e293b !important; 
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] { 
        background-color: #2563eb !important; 
        color: white !important;
    }
    div[data-testid="stMetricValue"] { font-size: 45px; font-weight: 800; color: #1e3a8a; }
    </style>
""", unsafe_allow_html=True)

if pipeline is not None:
    # --- NAGŁÓWEK ---
    st.title("🌍 Medyczny System Ekspercki WHO")
    st.markdown("##### Predykcja Oczekiwanej Długości Życia za pomocą Algorytmów Machine Learning")
    st.markdown("---")

    # --- SEKCJA: WYBÓR KRAJU ---
    st.subheader("📍 Profilowanie Geopolityczne i Demograficzne")

    if not country_profiles.empty:
        lista_krajow = ["--- Wybierz kraj, aby załadować zweryfikowane profile WHO ---"] + sorted(
            country_profiles['Country'].unique().tolist())
        wybrany_kraj = st.selectbox(
            "Wybierz profil państwa narodowego:",
            lista_krajow,
            help="Wczytuje oczyszczone z anomalii, historyczne profile statystyczne Światowej Organizacji Zdrowia."
        )
    else:
        st.warning("Brak pliku źródłowego CSV. System działa w trybie symulacji manualnej.")
        wybrany_kraj = "--- Wybierz kraj, aby załadować zweryfikowane profile WHO ---"

    # --- SŁOWNIK WARTOŚCI DOMYŚLNYCH (Zabezpieczony biologicznie i ekonomicznie) ---
    def_vals = {
        "Status": "Developing", "Income composition of resources": 0.65, "HIV/AIDS": 0.1,
        "Adult Mortality": 150, "GDP": 5000.0, "percentage expenditure": 5.5,
        "Population": 2000000, "Total expenditure": 5.5, "infant deaths": 25,
        "under-five deaths": 35, "BMI": 24.5, "Measles": 10, "Alcohol": 4.0,
        "Schooling": 12.0, "Hepatitis B": 90, "thinness  1-19 years": 4.5,
        "Polio": 90, "Diphtheria": 90, "thinness 5-9 years": 4.5
    }

    # Dynamiczne nadpisywanie słownika profilem
    if wybrany_kraj != "--- Wybierz kraj, aby załadować zweryfikowane profile WHO ---":
        profil = country_profiles[country_profiles['Country'] == wybrany_kraj].iloc[0]
        def_vals["Status"] = profil["Status"] if pd.notna(profil["Status"]) else "Developing"
        for key in def_vals.keys():
            if key in profil and pd.notna(profil[key]) and key != "Status":
                def_vals[key] = float(profil[key])

    st.markdown("---")
    col_inputs, col_results = st.columns([19, 11], gap="large")

    with col_inputs:
        st.subheader("⚙️ Panel Zarządzania Parametrami Populacji")

        # PODZIAŁ NA KATEGORIE ZA POMOCĄ TABS
        tab_main = st.radio("Wybierz domenę operacyjną:",
                            ["🚀 Kluczowe Determinanty", "📊 Ekonomia & Status", "🏥 Zdrowie publiczne & Profilaktyka"],
                            horizontal=True)

        if tab_main == "🚀 Kluczowe Determinanty":
            st.info(
                "💡 Poniższe parametry wykazują najwyższą istotność predykcyjną (Feature Importance) w modelu Random Forest.")

            income_comp = st.slider(
                "📈 Wskaźnik rozwoju zasobów (Income composition) [Korelacja Dodatnia  (+)]", 0.0, 1.0,
                min(1.0, max(0.0, float(def_vals["Income composition of resources"]))), step=0.01,
                help="Wskaźnik HDI określający poziom rozwoju społecznego. Silna korelacja dodatnia z długością życia."
            )

            adult_mortality = st.slider(
                "💀 Śmiertelność dorosłych (na 1000 ludności) [Korelacja Odwrotna (-)]", 0, 750,
                int(min(750, max(0, def_vals["Adult Mortality"]))), step=1,
                help="Liczba zgonów osób w wieku 15-60 lat na 1000 populacji. Im wyższa, tym niższy wynik prognozy."
            )

            hiv_aids = st.slider(
                "🦠 Współczynnik zgonów HIV/AIDS (0-4 lata) [Korelacja Odwrotna (-)]", 0.0, 55.0,
                float(min(55.0, max(0.0, def_vals["HIV/AIDS"]))), step=0.1,
                help="Śmiertelność z powodu HIV/AIDS w przeliczeniu na 1000 żywych urodzeń."
            )

        elif tab_main == "📊 Ekonomia & Status":
            status_index = 0 if def_vals["Status"] == "Developing" else 1
            status_label = st.radio(
                "Status geopolityczny państwa według klasyfikacji WHO",
                ["Kraj Rozwijający się (Developing)", "Kraj Rozwinięty (Developed)"],
                index=status_index, horizontal=True
            )
            status = 0 if status_label == "Kraj Rozwijający się (Developing)" else 1

            c1, c2 = st.columns(2)
            with c1:
                gdp = st.slider(
                    "💵 PKB na mieszkańca (USD)", 0, 120000, int(min(120000, max(0, def_vals["GDP"]))), step=500
                )
                percentage_exp = st.slider(
                    "🏥 Narodowe wydatki na zdrowie (% PKB per capita)", 0.0, 25.0,
                    float(min(25.0, max(0.0, def_vals["percentage expenditure"]))), step=0.1,
                    help="Zweryfikowany, realny procent PKB przeznaczany bezpośrednio na medycynę publiczną."
                )
            with c2:
                population = st.slider(
                    "👥 Liczba ludności państwa", 0, 300000000, int(min(300000000, max(0, def_vals["Population"]))),
                    step=1000000
                )
                total_exp = st.slider(
                    "🏛️ Udział sektora zdrowia w budżecie rządu (%)", 0.0, 25.0,
                    float(min(25.0, max(0.0, def_vals["Total expenditure"]))), step=0.1
                )

        elif tab_main == "🏥 Zdrowie publiczne & Profilaktyka":
            t_sub1, t_sub2, t_sub3 = st.tabs(
                ["🧬 Biometria & Styl Życia", "👶 Epidemiologia Dziecięca", "💉 Odporność Populacyjna"])

            with t_sub1:
                bmi = st.slider(
                    "⚖️ Średnie BMI populacji dorosłej", 15.0, 35.0, float(min(35.0, max(15.0, def_vals["BMI"]))),
                    step=0.1,
                    help="Średni indeks masy ciała obywateli. Skorygowany o błędy statystyczne baz danych WHO."
                )
                alcohol = st.slider(
                    "🍷 Konsumpcja czystego alkoholu (litry/osobę/rok)", 0.0, 25.0,
                    float(min(25.0, max(0.0, def_vals["Alcohol"]))), step=0.1,
                    help="Roczne spożycie czystego, 100-procentowego alkoholu na jednego mieszkańca w wieku 15 lat i więcej (w litrach)."
                )
                schooling = st.slider(
                    "🎓 Przeciętna długość edukacji (lata)", 0.0, 22.0,
                    float(min(22.0, max(0.0, def_vals["Schooling"]))), step=0.5,
                    help="Przeciętna liczba lat edukacji szkolnej oraz wyższej, jaką odbywa obywatel danego kraju."
                )

            with t_sub2:
                c_e1, c_e2 = st.columns(2)
                with c_e1:
                    infant_deaths = st.slider(
                        "👶 Zgony niemowląt / 1000", 0, 100, int(min(100, max(0, def_vals["infant deaths"]))),
                        help="Liczba zgonów dzieci poniżej 1 roku życia w przeliczeniu na 1000 urodzeń."
                    )
                    under_five_deaths = st.slider(
                        "🧒 Zgony dzieci do lat 5 / 1000", 0, 150, int(min(150, max(0, def_vals["under-five deaths"]))),
                        help="Liczba zgonów dzieci poniżej 5 roku życia w przeliczeniu na 1000 urodzeń."
                    )
                with c_e2:
                    thinness_1_19 = st.slider("📉 Chudość młodzieży 1-19 lat (%)", 0.0, 25.0,
                                              float(min(25.0, max(0.0, def_vals["thinness  1-19 years"]))), step=0.1)
                    thinness_5_9 = st.slider("📉 Chudość dzieci 5-9 lat (%)", 0.0, 25.0,
                                             float(min(25.0, max(0.0, def_vals["thinness 5-9 years"]))), step=0.1)
                measles = st.slider("🤒 Przypadki Odry (roczna liczba rejestracji)", 0, 5000,
                                    int(min(5000, max(0, def_vals["Measles"]))), step=50)

            with t_sub3:
                polio = st.slider(
                    "🛡️ Wskaźnik szczepień: Polio (%)", 0, 100, int(min(100, max(0, def_vals["Polio"]))),
                    help="Procentowy odsetek dzieci jednorocznych, które otrzymały pełną dawkę szczepienia przeciwko poliomyelitis (Pol3)."
                )
                diphtheria = st.slider(
                    "🛡️ Wskaźnik szczepień: Błonnica/Tężec (%)", 0, 100, int(min(100, max(0, def_vals["Diphtheria"]))),
                    help="Poziom wyszczepialności (w procentach) przeciwko błonnicy, tężcowi i krztuścowi (DTP3) wśród dzieci w wieku do 1 roku."
                )
                hepatitis_b = st.slider(
                    "🛡️ Wskaźnik szczepień: WZW B (%)", 0, 100, int(min(100, max(0, def_vals["Hepatitis B"]))),
                    help="Poziom wyszczepialności (w procentach) przeciwko wirusowemu zapaleniu wątroby typu B (HepB3) wśród dzieci w wieku do jednego roku."
                )

        # Integracja zmiennych z interfejsu
        inputs_vars = {
            "Status": status if 'status' in locals() else (0 if def_vals["Status"] == "Developing" else 1),
            "Adult Mortality": adult_mortality if 'adult_mortality' in locals() else int(def_vals["Adult Mortality"]),
            "infant deaths": infant_deaths if 'infant_deaths' in locals() else int(def_vals["infant deaths"]),
            "Alcohol": alcohol if 'alcohol' in locals() else float(def_vals["Alcohol"]),
            "percentage expenditure": percentage_exp if 'percentage_exp' in locals() else float(
                def_vals["percentage expenditure"]),
            "Hepatitis B": hepatitis_b if 'hepatitis_b' in locals() else int(def_vals["Hepatitis B"]),
            "Measles": measles if 'measles' in locals() else int(def_vals["Measles"]),
            "BMI": bmi if 'bmi' in locals() else float(def_vals["BMI"]),
            "under-five deaths": under_five_deaths if 'under-five_deaths' in locals() else int(
                def_vals["under-five deaths"]),
            "Polio": polio if 'polio' in locals() else int(def_vals["Polio"]),
            "Total expenditure": total_exp if 'total_exp' in locals() else float(def_vals["Total expenditure"]),
            "Diphtheria": diphtheria if 'diphtheria' in locals() else int(def_vals["Diphtheria"]),
            "HIV/AIDS": hiv_aids if 'hiv_aids' in locals() else float(def_vals["HIV/AIDS"]),
            "GDP": gdp if 'gdp' in locals() else float(def_vals["GDP"]),
            "Population": population if 'population' in locals() else float(def_vals["Population"]),
            "thinness  1-19 years": thinness_1_19 if 'thinness_1_19' in locals() else float(
                def_vals["thinness  1-19 years"]),
            "thinness 5-9 years": thinness_5_9 if 'thinness_5_9' in locals() else float(def_vals["thinness 5-9 years"]),
            "Income composition of resources": income_comp if 'income_comp' in locals() else float(
                def_vals["Income composition of resources"]),
            "Schooling": schooling if 'schooling' in locals() else float(def_vals["Schooling"])
        }


    # --- FUNKCJA PROGNOZUJĄCA (SILNIK ENGINE) ---
    def predict_life_expectancy(data_dict):
        df_in = pd.DataFrame([data_dict])[pipeline.features]
        X_imp = pd.DataFrame(pipeline.prep.imputer.transform(df_in), columns=pipeline.features)
        for col in pipeline.features:
            if col != "Status" and X_imp[col].max() > Config.SKW_LIMIT:
                X_imp[col] = np.log1p(X_imp[col].clip(lower=0))
        X_std = pd.DataFrame(pipeline.prep.scaler.transform(X_imp), columns=pipeline.features)
        X_final = X_std[pipeline.remaining].values
        return float(pipeline.best_model.predict(X_final)[0])


    # Główna predykcja
    predicted_value = predict_life_expectancy(inputs_vars)

    # --- PANEL WYNIKOWY (PRAWA KOLUMNA) ---
    with col_results:
        st.subheader("🎯 Prognoza i Analiza Wrażliwości")

        # GŁÓWNY WIDŻET WYNIKU
        st.markdown(f"""
            <div style="background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%); padding: 25px; border-radius: 12px; color: white; text-align: center; box-shadow: 0 4px 15px rgba(37, 99, 235, 0.2); margin-bottom: 25px;">
                <span style="text-transform: uppercase; letter-spacing: 0.12em; font-size: 11px; opacity: 0.9; font-weight: 600;">
                    Oczekiwana długość życia populacji
                </span>
                <h1 style="font-size: 62px; font-weight: 800; margin: 8px 0; color: white; text-shadow: 2px 2px 4px rgba(0,0,0,0.15);">{predicted_value:.2f} lat</h1>
                <div style="padding: 6px 16px; background: rgba(255,255,255,0.18); border-radius: 30px; display: inline-block; font-size: 12px; font-weight: 600; letter-spacing: 0.05em;">
                    Model: {pipeline.model_name}
                </div>
            </div>
        """, unsafe_allow_html=True)

        # --- DYNAMICZNY SYMULATOR DECYZYJNY (SCENARIUSZE WHAT-IF) ---
        st.markdown("### ⚡ Symulacja Strategii Rozwoju (Co-Jeśli)")
        scenariusz = st.selectbox(
            "Wybierz rządowy program naprawczy:",
            ["Brak modyfikacji (Status Quo)", "Pakiet Modernizacji Służby Zdrowia (+ Wydatki & Edukacja)",
             "Krajowy Program Walki z Zakażeniami (Epidemiologia)", "Strategia Likwidacji Ubóstwa (Ekonomia & HDI)"]
        )

        if scenariusz != "Brak modyfikacji (Status Quo)":
            sim_vars = inputs_vars.copy()
            if scenariusz == "Pakiet Modernizacji Służby Zdrowia (+ Wydatki & Edukacja)":
                sim_vars["percentage expenditure"] = min(25.0, sim_vars["percentage expenditure"] * 1.5)
                sim_vars["Schooling"] = min(22.0, sim_vars["Schooling"] + 2.0)
                sim_vars["Polio"] = min(100, int(sim_vars["Polio"] * 1.1))
                sim_label = "🚀 Efekt zwiększenia budżetu zdrowia i edukacji obywatelskiej"
            elif scenariusz == "Krajowy Program Walki z Zakażeniami (Epidemiologia)":
                sim_vars["HIV/AIDS"] = max(0.0, sim_vars["HIV/AIDS"] * 0.4)
                sim_vars["Adult Mortality"] = max(20, int(sim_vars["Adult Mortality"] * 0.8))
                sim_vars["Diphtheria"] = min(100, int(sim_vars["Diphtheria"] * 1.15))
                sim_label = "🦠 Efekt masowych szczepień i redukcji transmisji wirusów"
            elif scenariusz == "Strategia Likwidacji Ubóstwa (Ekonomia & HDI)":
                sim_vars["Income composition of resources"] = min(1.0,
                                                                  sim_vars["Income composition of resources"] * 1.2)
                sim_vars["thinness  1-19 years"] = max(0.0, sim_vars["thinness  1-19 years"] * 0.5)
                sim_vars["GDP"] = sim_vars["GDP"] * 1.3
                sim_label = "💰 Efekt stymulacji gospodarczej i walki z głodem"

            sim_value = predict_life_expectancy(sim_vars)
            roznica = sim_value - predicted_value
            kolor_roznicy = "#16a34a" if roznica >= 0 else "#dc2626"
            znak = "+" if roznica >= 0 else ""

            st.markdown(f"""
                <div style="background-color: #f8fafc; border-left: 5px solid #2563eb; padding: 15px; border-radius: 4px; margin-top: 10px;">
                    <small style="font-weight: 700; color: #64748b; text-transform: uppercase;">{sim_label}</small>
                    <div style="font-size: 22px; font-weight: 700; color: #1e293b; margin-top: 5px;">
                        Symulowana długość życia: <span style="color: #2563eb;">{sim_value:.2f} lat</span>
                    </div>
                    <div style="font-size: 15px; font-weight: 600; color: {kolor_roznicy}; margin-top: 2px;">
                        Zmiana względem obecnych ustawień: {znak}{roznica:.2f} roku
                    </div>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # --- SEKCJA ISTOTNOŚCI CECH ---
        st.markdown("### 📊 Struktura Decyzyjna Modelu")
        if hasattr(pipeline.best_model, "feature_importances_"):
            importances = pipeline.best_model.feature_importances_
            importance_df = pd.DataFrame({
                "Wskaźnik diagnostyczny": pipeline.remaining,
                "Wpływ na decyzję": importances * 100
            }).sort_values(by="Wpływ na decyzję", ascending=False)

            st.dataframe(
                importance_df.style.format({"Wpływ na decyzję": "{:.2f}%"}).background_gradient(cmap="Blues"),
                hide_index=True,
                use_container_width=True,
                height=280
            )
else:
    st.error("Krytyczny brak wymaganych plików binarnych do uruchomienia aplikacji.")