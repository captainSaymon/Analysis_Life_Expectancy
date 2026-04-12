import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from data_fetch import DataFetch


class DataProcessor:
    def __init__(self):
        self.fetch_data()

        self.X = None
        self.y = None

        self.X_train = None
        self.X_val = None
        self.X_test = None

        self.y_train = None
        self.y_val = None
        self.y_test = None

        self.scaler = StandardScaler()

        # hiperparametry
        self.k = 9
        self.random_state = 40

    # Pobranie danych
    def fetch_data(self):
        self.line()
        self.df = DataFetch().fetch_dataset().data.copy()

    # Czyszczenie danych
    def clean_data(self):
        self.line()
        print("\nCzyszczenie danych")

        self.df.replace('?', np.nan, inplace=True)

        self.df.drop_duplicates(inplace=True)

        for col in self.df.columns:
            self.df[col] = pd.to_numeric(self.df[col], errors='ignore')

        print("\nBraki danych:")
        print(self.df.isnull().sum())

        return self
    
    # Obsługa braku wartości
    def handle_missing_values(self):
        self.line()
        print("\nUzupełnianie braków")

        num_cols = self.df.select_dtypes(include=np.number).columns
        for col in num_cols:
            self.df[col].fillna(self.df[col].median(), inplace=True)

        cat_cols = self.df.select_dtypes(include='object').columns
        for col in cat_cols:
            self.df[col].fillna(self.df[col].mode()[0], inplace=True)

        print("Braki po uzupełnieniu:")
        print(self.df.isnull().sum())

        return self

    # Przygotowanie cech
    def prepare_features(self, target_column="Life expectancy "):
        self.line()
        print("\nPrzygotowanie cech")

        self.df = pd.get_dummies(self.df, drop_first=True)

        self.X = self.df.drop(columns=[target_column])
        self.y = self.df[target_column]

        return self
    
