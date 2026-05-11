import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

from sklearn.metrics import r2_score
from sklearn.inspection import permutation_importance
from src.configs.config import Config


class Evaluator:
    def evaluate(self, model, X_test, y_test, model_name):
        print("\nKROK 8: Test końcowy i analiza istotności statystycznej...")

        preds = model.predict(X_test)
        r2 = r2_score(y_test, preds)

        print("=" * 50)
        print(f"WYNIK FINALNY DLA: {model_name}")
        print(f"R2 NA TEST: {r2:.4f}")
        print("=" * 50)

        return preds, r2
    
    def report_statistical_significance(self, X_train, y_train, feature_names):
        print("\n\nANALIZA STATYSTYCZNA CECH (Model OLS)")
        
        X_stat = sm.add_constant(pd.DataFrame(X_train, columns=feature_names))
        ols_model = sm.OLS(y_train.values, X_stat).fit()
        
        print(ols_model.summary().tables[1])

    def feature_importance(self, model, X_test, y_test, feature_names):
        result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=Config.RANDOM_STATE)
        sorted_idx = result.importances_mean.argsort()

        plt.figure(figsize=(10, 8))
        plt.barh(np.array(feature_names)[sorted_idx], result.importances_mean[sorted_idx], color='steelblue')
        plt.title(f"Istotność cech dla {type(model).__name__}")
        plt.xlabel("Spadek R2 po wymieszaniu cechy")
        plt.grid(True, alpha=0.3)
        plt.subplots_adjust(left=0.3)
        plt.tight_layout()

    def plot_residuals(self, y_true, y_pred, model_name):
        residuals = y_true - y_pred
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        ax1.scatter(y_true, y_pred, alpha=0.6, color='steelblue', edgecolors='w')
        lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
        ax1.plot(lims, lims, 'r--', alpha=0.75)
        ax1.set_title(f'{model_name}: Dane rzeczywiste vs Predykcja')
        ax1.set_xlabel('Rzeczywista długość życia')
        ax1.set_ylabel('Przewidziana długość życia')

        ax2.scatter(y_pred, residuals, alpha=0.6, color='steelblue', edgecolors='w')
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.set_title(f'{model_name}: Wykres Reszt (Błędów)')
        ax2.set_xlabel('Przewidziane wartości')
        ax2.set_ylabel('Błąd (Residuum)')
        
        plt.tight_layout()

    def plot_model_comparison(self, model_scores):
        names = list(model_scores.keys())
        r2_values = [v['R2'] for v in model_scores.values()]
        
        plt.figure(figsize=(10, 5))
        sns.barplot(x=names, y=r2_values, hue=names, palette='dark:steelblue', legend=False, color='steelblue')
        plt.xticks(rotation=30)
        plt.ylabel("R2 Score")
        plt.title("Porównanie skuteczności modeli")
        plt.grid(True, axis='y', alpha=0.3)

    def plot_correlation_matrix(self, X, feature_names):
            print("\nGenerowanie macierzy korelacji...")
            df_corr = pd.DataFrame(X, columns=feature_names)
            corr_matrix = df_corr.corr()

            plt.figure(figsize=(12, 10))
            sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, fmt='.2f', linewidths=0.5)
            plt.title("Macierz Korelacji Cech")
            plt.tight_layout()