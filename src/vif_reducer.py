import pandas as pd

from statsmodels.stats.outliers_influence import variance_inflation_factor
from src.configs.config import Config


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