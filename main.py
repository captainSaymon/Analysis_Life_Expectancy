import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.inspection import permutation_importance

from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

warnings.filterwarnings("ignore", category=RuntimeWarning)


# CONFIG
class Config:
    K_NEIGHBORS = 9
    RANDOM_STATE = 42
    VIF_THRESHOLD = 5
    SKW_LIMIT = 0.75


# DATA LOADER
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


# PREPROCESSOR
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


# VIF REDUCTION
class VIFReducer:
    def reduce(self, X_train_std, feature_names):
        print(f"\nKROK 5: Redukcja VIF (Próg: {Config.VIF_THRESHOLD})...")

        X_df = pd.DataFrame(X_train_std, columns=feature_names)

        while True:
            vif_vals = [
                variance_inflation_factor(X_df.values, i)
                for i in range(len(X_df.columns))
            ]
            max_vif = max(vif_vals)

            if max_vif > Config.VIF_THRESHOLD:
                idx = vif_vals.index(max_vif)
                feat = X_df.columns[idx]
                print(f"Usuwam: {feat} (VIF: {max_vif:.2f})")
                X_df = X_df.drop(columns=[feat])
            else:
                break

        return X_df, X_df.columns


# COOK DISTANCE
class CookAnalyzer:
    def filter_outliers(self, X_train, y_train):
        print("\nKROK 6: Cook distance...")

        X_const = sm.add_constant(X_train)
        ols = sm.OLS(y_train.values, X_const).fit()
        cooks_d = ols.get_influence().cooks_distance[0]

        plt.figure(figsize=(12, 5))
        plt.stem(np.arange(len(cooks_d)), cooks_d, markerfmt=",")
        plt.axhline(y=4/len(X_train), color='orange', linestyle='--')
        plt.title("Odległość Cooka")
        plt.grid(True, alpha=0.3)

        safe_idx = np.where(cooks_d <= 4/len(X_train))[0]

        return X_train[safe_idx], y_train.iloc[safe_idx]


# MODELS
class ModelTrainer:
    def __init__(self):
        self.models = {
            "KNN": KNeighborsRegressor(n_neighbors=Config.K_NEIGHBORS),
            "Linear Regression": LinearRegression(),
            "Decision Tree": DecisionTreeRegressor(random_state=Config.RANDOM_STATE),
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=Config.RANDOM_STATE),
            "SVR": SVR()
        }

    def train_and_select(self, X_train, y_train, X_val, y_val):
        print("\nKROK 7: Porównanie modeli...\n")

        print(f"{'Model':20} | {'R2 (VAL)':>10}")
        print("-" * 35)

        best_model = None
        best_score = -np.inf
        best_name = ""
        scores = {}

        for name, model in self.models.items():
            model.fit(X_train, y_train)
            pred = model.predict(X_val)
            r2 = r2_score(y_val, pred)

            scores[name] = r2
            print(f"{name:20} | {r2:10.4f}")

            if r2 > best_score:
                best_score = r2
                best_model = model
                best_name = name

        print("-" * 35)
        print(f"NAJLEPSZY MODEL: {best_name} ({best_score:.4f})")

        return best_model, best_name, scores


# EVALUATION + VISUALS
class Evaluator:
    def evaluate(self, model, X_test, y_test, model_name):
        print("\nKROK 8: Test końcowy...")

        preds = model.predict(X_test)
        r2 = r2_score(y_test, preds)

        print("=" * 40)
        print(f"NAJLEPSZY MODEL: {model_name}")
        print(f"R2 NA TEST: {r2:.4f}")
        print("=" * 40)

        return preds, r2
    
    def feature_importance(self, model, X_test, y_test, feature_names):
        result = permutation_importance(
            model,
            X_test,
            y_test,
            n_repeats=10,
            random_state=Config.RANDOM_STATE
        )

        sorted_idx = result.importances_mean.argsort()

        plt.figure(figsize=(10, 6))
        plt.barh(
            np.array(feature_names)[sorted_idx],
            result.importances_mean[sorted_idx]
        )
        plt.title("Ważność cech (Permutation Importance)")
        plt.grid(True, alpha=0.3)

    def plots(self, X_train, X_test, y_test, preds, features, model_scores):
        print("\nPROJEKT ZAKOŃCZONY")

        corr_df = pd.DataFrame(X_train, columns=features)

        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", center=0)
        plt.title("Macierz korelacji")

        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, preds, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()],
                 [y_test.min(), y_test.max()], 'r--')
        plt.xlabel("Rzeczywiste")
        plt.ylabel("Przewidziane")
        plt.title("Predykcja vs rzeczywistość")
        plt.grid(True, alpha=0.3)

        plt.figure(figsize=(10, 5))
        plt.bar(model_scores.keys(), model_scores.values())
        plt.xticks(rotation=30)
        plt.ylabel("R2 (VAL)")
        plt.title("Porównanie modeli")
        plt.grid(True, axis='y', alpha=0.3)

        plt.show()


# PIPELINE ORCHESTRATOR
class LifeExpectancyPipeline:
    def __init__(self, path):
        self.loader = DataLoader(path)
        self.prep = Preprocessor()
        self.vif = VIFReducer()
        self.cook = CookAnalyzer()
        self.trainer = ModelTrainer()
        self.eval = Evaluator()

    def run(self):
        X_train, X_val, X_test, y_train, y_val, y_test, features = self.loader.load()

        X_train, X_val, X_test = self.prep.impute(X_train, X_val, X_test)
        y_train_f, y_val_f, y_test_f = y_train, y_val, y_test

        X_train, X_val, X_test = self.prep.transform_skew(X_train, X_val, X_test)
        X_train_std, X_val_std, X_test_std = self.prep.scale(X_train, X_val, X_test)

        X_vif, remaining = self.vif.reduce(X_train_std, features)

        X_train_final = X_vif.values
        X_val_final = pd.DataFrame(X_val_std, columns=features)[remaining].values
        X_test_final = pd.DataFrame(X_test_std, columns=features)[remaining].values

        X_train_final, y_train_final = self.cook.filter_outliers(X_train_final, y_train_f)

        model, name, scores = self.trainer.train_and_select(
            X_train_final, y_train_final,
            X_val_final, y_val_f
        )

        preds, _ = self.eval.evaluate(model, X_test_final, y_test_f, name)

        self.eval.feature_importance(
            model,
            X_test_final,
            y_test_f,
            remaining
        )

        self.eval.plots(
            X_train_final,
            X_test_final,
            y_test_f,
            preds,
            remaining,
            scores
        )


# RUN
if __name__ == "__main__":
    pipeline = LifeExpectancyPipeline("Life Expectancy Data.csv")
    pipeline.run()