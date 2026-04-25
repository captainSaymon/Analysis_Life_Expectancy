import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import r2_score
from sklearn.inspection import permutation_importance
from src.configs.config import Config


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