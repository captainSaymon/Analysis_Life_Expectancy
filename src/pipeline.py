import pandas as pd
import matplotlib.pyplot as plt
import joblib

from src.cook import CookAnalyzer
from src.data_loader import DataLoader
from src.evaluator import Evaluator
from src.model_trainer import ModelTrainer
from src.preprocessor import Preprocessor
from src.vif_reducer import VIFReducer


class LifeExpectancyPipeline:
    def __init__(self, path):
        self.loader = DataLoader(path)
        self.prep = Preprocessor()
        self.vif = VIFReducer()
        self.cook = CookAnalyzer()
        self.trainer = ModelTrainer()
        self.eval = Evaluator()

        self.best_model = None
        self.model_name = ""
        self.features = []
        self.remaining = []

    def run(self):
        # 1. Przygotowanie danych
        X_train, X_val, X_test, y_train, y_val, y_test, features = self.loader.load()
        self.features = features

        # 2. Preprocessing
        X_train, X_val, X_test = self.prep.impute(X_train, X_val, X_test)
        X_train, X_val, X_test = self.prep.transform_skew(X_train, X_val, X_test)
        X_train_std, X_val_std, X_test_std = self.prep.scale(X_train, X_val, X_test)

        # 3. Redukcja wymiarów (VIF)
        X_vif, remaining = self.vif.reduce(X_train_std, features)
        self.remaining = remaining

        X_train_final = X_vif.values
        X_val_final = pd.DataFrame(X_val_std, columns=features)[remaining].values
        X_test_final = pd.DataFrame(X_test_std, columns=features)[remaining].values

        # 4. Usuwanie obserwacji odstających (Cook's Distance)
        X_train_final, y_train_final = self.cook.filter_outliers(X_train_final, y_train)

        # 5. Wybór modelu i trening
        self.best_model, self.model_name, all_scores = self.trainer.train_and_select(
            X_train_final, y_train_final, X_val_final, y_val
        )

        # 6. Analiza statystyczna i generowanie wykresów
        self.eval.report_statistical_significance(X_train_final, y_train_final, remaining)

        preds, _ = self.eval.evaluate(self.best_model, X_test_final, y_test, self.model_name)

        # Wywołanie pełnych wizualizacji
        self.eval.plot_residuals(y_test, preds, self.model_name)
        self.eval.feature_importance(self.best_model, X_test_final, y_test, remaining)
        self.eval.plot_model_comparison(all_scores)
        self.eval.plot_correlation_matrix(X_vif.values, remaining)

        # 7. Serializacja całego potoku do pliku binarnego dla Streamlita
        joblib.dump(self, 'model_data.pkl')

        print("\n[SUKCES] PROJEKT ZAKOŃCZONY POMYŚLNIE.")
        print("Wszystkie metryki policzone, a kompletny obiekt zapisano w 'model_data.pkl'.")

        # Fizyczne wyrenderowanie okien z wykresami na ekranie komputera
        print("\nOtwieranie okien z wykresami analitycznymi...")
        plt.show()