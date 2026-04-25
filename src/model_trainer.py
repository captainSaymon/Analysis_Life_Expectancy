import numpy as np

from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from src.configs.config import Config


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