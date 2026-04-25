import pandas as pd

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