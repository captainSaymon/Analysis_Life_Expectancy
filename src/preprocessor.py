import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.impute import IterativeImputer
from src.configs.config import Config


class Preprocessor:
    def __init__(self):
        self.imputer = IterativeImputer(random_state=Config.RANDOM_STATE)
        self.scaler = StandardScaler()

    def impute(self, X_train, X_val, X_test):
        print("\nKROK 2: Imputacja danych...")

        X_train_f = pd.DataFrame(
            self.imputer.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        X_val_f = pd.DataFrame(
            self.imputer.transform(X_val),
            columns=X_val.columns,
            index=X_val.index
        )
        X_test_f = pd.DataFrame(
            self.imputer.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )

        return X_train_f, X_val_f, X_test_f

    def transform_skew(self, X_train, X_val, X_test):
        print("\nKROK 3: Logarytmowanie...")

        X_train.hist(figsize=(12, 10), bins=20, color='skyblue', edgecolor='black')
        plt.suptitle("Rozkłady cech przed transformacją")

        skew_values = X_train.skew()
        to_transform = skew_values[abs(skew_values) > Config.SKW_LIMIT].index.tolist()

        for col in to_transform:
            X_train[col] = np.log1p(X_train[col].clip(lower=0))
            X_val[col] = np.log1p(X_val[col].clip(lower=0))
            X_test[col] = np.log1p(X_test[col].clip(lower=0))

        return X_train, X_val, X_test

    def scale(self, X_train, X_val, X_test):
        print("\nKROK 4: Standaryzacja...")

        X_train_std = self.scaler.fit_transform(X_train)
        X_val_std = self.scaler.transform(X_val)
        X_test_std = self.scaler.transform(X_test)

        return X_train_std, X_val_std, X_test_std
