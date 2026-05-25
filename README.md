# Analysis Life Expectancy (ALE)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white) 
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-013243?style=for-the-badge&logo=Matplotlib&logoColor=white)

Głównym zadaniem modelu jest predykcja średniej długości życia **(Life Expectancy)** w krajach europejskich na podstawie wybranych wskaźników społeczno-ekonomicznych i zdrowotnych. 

Program, po otrzymaniu zestawu danych wejściowych, ma za zadanie oszacować przewidywaną długość życia z określoną dokładnością. Model może służyć do wskazywania krajów, które radzą sobie znacznie lepiej, niż wynikałoby to z ich sytuacji ekonomicznej, co pozwala na dalszą analizę jakości systemów zdrowotnych. 

Model będzie realizował zadanie regresji, ponieważ zmienna określająca długość życia jest wartością ciągłą. 

## Uruchomienie

Aby uruchomić model wystarczy wejść do pliku `main.py`. Natomiast aby skorzystać z aplikacji trzeba jeszcze uruchomić `main.py` a następnie `app.py`.

## Stos technologiczny i biblioteki

Projekt został zbudowany w języku Python i wykorzystuje następujące biblioteki z podziałem na obszary zastosowań:

#### Interfejs użytkownika (UI) i wdrożenie
* **Streamlit** - stworzenie interaktywnej aplikacji webowej i interfejsu użytkownika

#### Analiza, przetwarzanie danych i statystyka
* **Pandas** - manipulacja danymi, analiza struktur DataFrame
* **NumPy** - operacje wektorowe i obliczenia matematyczne
* **Statsmodels** - zaawansowane analizy statystyczne

#### Uczenie maszynowe
* **Scikit-learn** - główne narzędzie do modelowania

#### Wizualizacja danych
* **Matplotlib** - podstawowe wykresy i wykresy statystyczne
* **Seaborn** - zaawansowane i estetyczne wizualizacje danych

#### Inne narzędzia
* **Joblib** - zapisywanie i wczytywanie wytrenowanych modeli maszynowych
* **OS** - wbudowany moduł Pythona do zarządzania ścieżkami plików


### Instalacja
   ```bash
   pip install streamlit pandas numpy statsmodels scikit-learn matplotlib seaborn joblib
   ```

## Link 

Do realizacji projektu wykorzystano zbiór danych udostępniony w serwisie **[Kaggle](https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who)**. Zbiór ten opiera się na danych z Global Health Observatory (GHO) oraz danych ekonomicznych Narodów Zjednoczonych. 

Obejmuje on okres od 2000 do 2015 roku i zawiera informacje o 193 krajach świata. 
