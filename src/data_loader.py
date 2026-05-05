import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from src.configs.config import Config


class DataLoader:
    def __init__(self, path):
        self.path = path

    def load(self):
        print("KROK 1: Wczytywanie danych...")
        df = pd.read_csv(self.path)
        df.columns = [col.strip() for col in df.columns]

        le = LabelEncoder()
        df["Status"] = le.fit_transform(df["Status"])
        df = df.dropna(subset=["Life expectancy"])

        X = df.drop(["Life expectancy", "Country", "Year"], axis=1)
        y = df["Life expectancy"]

        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y,
            test_size=0.4,
            random_state=Config.RANDOM_STATE,
            stratify=X["Status"]
        )

        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=0.5,
            random_state=Config.RANDOM_STATE,
            stratify=X_temp["Status"]
        )

        return X_train, X_val, X_test, y_train, y_val, y_test, X.columns
