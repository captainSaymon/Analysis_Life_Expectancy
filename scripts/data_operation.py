import pandas as pd
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from data_fetch import DataFetch


class DataProcessor:
    def __init__(self):
        self.df = None

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

        if "Bare Nuclei" in self.df.columns:
            self.df["Bare Nuclei"] = pd.to_numeric(self.df["Bare Nuclei"], errors="coerce")

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

        print("Braki po uzupełnieniu:")
        print(self.df.isnull().sum())

        return self

    # Przygotowanie cech
    def prepare_features(self, target_column=None):
        self.line()
        print("\nPrzygotowanie cech")

        if target_column is None:
            target_column = self.df.columns[-1]

        self.y = self.df[target_column].copy()
        self.X = self.df.drop(columns=[target_column])

        self.X = pd.get_dummies(self.X, drop_first=True)

        self.y = self.y.astype(int)

        return self
    
    # Podział danych
    def split_data(self):
        self.line()
        print("\nPodział danych")

        X_train, X_temp, y_train, y_temp = train_test_split(
            self.X,
            self.y,
            test_size=0.4,
            random_state=self.random_state,
            stratify=None
        )

        X_val, X_test, y_val, y_test = train_test_split(
            X_temp,
            y_temp,
            test_size=0.5,
            random_state=self.random_state,
            stratify=None
        )

        self.X_train, self.X_val, self.X_test = X_train, X_val, X_test
        self.y_train, self.y_val, self.y_test = y_train, y_val, y_test

        print("Train:", X_train.shape)
        print("Val:", X_val.shape)
        print("Test:", X_test.shape)

        return self
    
    # Skalowanie danych
    def scale_data(self):
        self.line()
        print("\nSkalowanie danych")

        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_val = self.scaler.transform(self.X_val)
        self.X_test = self.scaler.transform(self.X_test)
        
        return self
    
    def test_knn(self, k=None):
        self.line()
        print("\nTest KNN")

        if k is None:
            k = self.k

        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(self.X_train, self.y_train)

        y_pred = model.predict(self.X_val)

        acc = accuracy_score(self.y_val, y_pred)

        print(f"K = {k}")
        print(f"Accuracy (val): {acc:.4f}")

        return acc
    
    # Wykaz danych
    def show_basic_info(self):
        print("\nInfo:")
        print(self.df.info())

    # Linie
    def line(self, n=100):
        print('-'*n)


if __name__ == '__main__':
    processor = DataProcessor()
    processor.fetch_data()

    processor.clean_data() \
         .handle_missing_values() \
         .prepare_features() \
         .split_data() \
         .scale_data()
    
    processor.test_knn()