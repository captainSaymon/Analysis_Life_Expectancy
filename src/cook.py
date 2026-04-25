import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

from sklearn.experimental import enable_iterative_imputer


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