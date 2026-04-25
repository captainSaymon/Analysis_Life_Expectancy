import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ==========================================
# 1. KONFIGURACJA
# ==========================================
K_NEIGHBORS = 9
RANDOM_STATE = 42
VIF_THRESHOLD = 5

# ==========================================
# 2. WCZYTANIE DANYCH
# ==========================================
print("KROK 1: Wczytywanie danych...")
df = pd.read_csv('Life Expectancy Data.csv')
df.columns = [col.strip() for col in df.columns]

le = LabelEncoder()
df['Status'] = le.fit_transform(df['Status'])
df = df.dropna(subset=['Life expectancy'])

X = df.drop(['Life expectancy', 'Country', 'Year'], axis=1)
y = df['Life expectancy']

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=RANDOM_STATE, stratify=X['Status']
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=RANDOM_STATE, stratify=X_temp['Status']
)

# ==========================================
# 3. IMPUTACJA
# ==========================================
print("\nKROK 2: Imputacja danych...")
it_imp = IterativeImputer(random_state=RANDOM_STATE)

X_train_f = pd.DataFrame(it_imp.fit_transform(X_train), columns=X.columns, index=X_train.index)
X_val_f = pd.DataFrame(it_imp.transform(X_val), columns=X.columns, index=X_val.index)
X_test_f = pd.DataFrame(it_imp.transform(X_test), columns=X.columns, index=X_test.index)

y_train_f, y_val_f, y_test_f = y_train, y_val, y_test

# ==========================================
# 4. SKOŚNOŚĆ + HISTOGRAMY
# ==========================================
print("\nKROK 3: Logarytmowanie...")

X_train_f.hist(figsize=(12, 10), bins=20, color='skyblue', edgecolor='black')
plt.suptitle("Rozkłady cech przed transformacją")

skew_limit = 0.75
skew_values = X_train_f.skew()
to_transform = skew_values[abs(skew_values) > skew_limit].index.tolist()

for col in to_transform:
    X_train_f[col] = np.log1p(X_train_f[col].clip(lower=0))
    X_val_f[col] = np.log1p(X_val_f[col].clip(lower=0))
    X_test_f[col] = np.log1p(X_test_f[col].clip(lower=0))

# ==========================================
# 5. STANDARYZACJA
# ==========================================
print("\nKROK 4: Standaryzacja...")
scaler = StandardScaler()

X_train_std = scaler.fit_transform(X_train_f)
X_val_std = scaler.transform(X_val_f)
X_test_std = scaler.transform(X_test_f)

# ==========================================
# 6. REDUKCJA VIF
# ==========================================
print(f"\nKROK 5: Redukcja VIF (Próg: {VIF_THRESHOLD})...")

feature_names = X.columns.tolist()
X_vif_df = pd.DataFrame(X_train_std, columns=feature_names)

while True:
    vif_vals = [variance_inflation_factor(X_vif_df.values, i) for i in range(len(X_vif_df.columns))]
    max_vif = max(vif_vals)

    if max_vif > VIF_THRESHOLD:
        idx = vif_vals.index(max_vif)
        feat = X_vif_df.columns[idx]
        print(f"Usuwam: {feat} (VIF: {max_vif:.2f})")
        X_vif_df = X_vif_df.drop(columns=[feat])
    else:
        break

remaining_cols = X_vif_df.columns

X_train_final = X_vif_df.values
X_val_final = pd.DataFrame(X_val_std, columns=feature_names)[remaining_cols].values
X_test_final = pd.DataFrame(X_test_std, columns=feature_names)[remaining_cols].values

# ==========================================
# 7. COOK DISTANCE
# ==========================================
print("\nKROK 6: Cook distance...")

X_constant = sm.add_constant(X_train_final)
ols = sm.OLS(y_train_f.values, X_constant).fit()
cooks_d = ols.get_influence().cooks_distance[0]

plt.figure(figsize=(12, 5))
plt.stem(np.arange(len(cooks_d)), cooks_d, markerfmt=",")
plt.axhline(y=4/len(X_train_final), color='orange', linestyle='--')
plt.title("Odległość Cooka")
plt.grid(True, alpha=0.3)

safe_idx = np.where(cooks_d <= 4/len(X_train_final))[0]

X_train_final = X_train_final[safe_idx]
y_train_final = y_train_f.iloc[safe_idx]

# ==========================================
# 8. PORÓWNANIE MODELI
# ==========================================
print("\nKROK 7: Porównanie modeli...")

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

models = {
    "KNN": KNeighborsRegressor(n_neighbors=K_NEIGHBORS),
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=RANDOM_STATE),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE),
    "SVR": SVR()
}

print(f"{'Model':20} | {'R2 (VAL)':>10}")
print("-" * 35)

best_model = None
best_score = -np.inf
best_name = ""
model_scores = {}

for name, model in models.items():
    model.fit(X_train_final, y_train_final)
    y_pred_val = model.predict(X_val_final)
    r2 = r2_score(y_val_f, y_pred_val)

    model_scores[name] = r2

    print(f"{name:20} | {r2:10.4f}")

    if r2 > best_score:
        best_score = r2
        best_model = model
        best_name = name

print("-" * 35)
print(f"NAJLEPSZY MODEL: {best_name} ({best_score:.4f})")

# ==========================================
# 9. TEST KOŃCOWY
# ==========================================
print("\nKROK 8: Test końcowy...")

y_test_pred = best_model.predict(X_test_final)
final_test_r2 = r2_score(y_test_f, y_test_pred)

print("=" * 40)
print(f"NAJLEPSZY MODEL: {best_name}")
print(f"R2 NA TEST: {final_test_r2:.4f}")
print("=" * 40)

# ==========================================
# 10. WYKRESY
# ==========================================
plt.figure(figsize=(10, 8))
corr_df = pd.DataFrame(X_train_final, columns=remaining_cols)
sns.heatmap(corr_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", center=0)
plt.title("Macierz korelacji")

# Predykcja vs rzeczywistość
plt.figure(figsize=(8, 6))
plt.scatter(y_test_f, y_test_pred, alpha=0.5)
plt.plot([y_test_f.min(), y_test_f.max()],
         [y_test_f.min(), y_test_f.max()],
         'r--')
plt.xlabel("Rzeczywiste")
plt.ylabel("Przewidziane")
plt.title(f"R2 = {final_test_r2:.4f}")
plt.grid(True, alpha=0.3)

# ==========================================
# 11. WAŻNOŚĆ CECH
# ==========================================
from sklearn.inspection import permutation_importance

result = permutation_importance(best_model, X_test_final, y_test_f,
                                n_repeats=10, random_state=RANDOM_STATE)

sorted_idx = result.importances_mean.argsort()

plt.figure(figsize=(10, 6))
plt.barh(remaining_cols[sorted_idx],
         result.importances_mean[sorted_idx])
plt.title("Ważność cech")
plt.grid(True, alpha=0.3)

# ==========================================
# 12. WYKRES PORÓWNANIA MODELI
# ==========================================
plt.figure(figsize=(8, 5))
plt.bar(model_scores.keys(), model_scores.values())
plt.xticks(rotation=30)
plt.ylabel("R2 (VAL)")
plt.title("Porównanie modeli")
plt.grid(True, axis='y', alpha=0.3)
plt.tight_layout()

print("\nPROJEKT ZAKOŃCZONY")
plt.show()