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

# Ignorujemy błędy runtime przy logarytmowaniu - obsłużymy to ręcznie
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ==========================================
# 1. KONFIGURACJA I FUNKCJE POMOCNICZE
# ==========================================
K_NEIGHBORS = 9
RANDOM_STATE = 42

# TWOJA ZMIENNA "SUWAK" DO SELEKCJI DANYCH:
# Niska wartość (np. 2) = silna selekcja, mało cech w macierzy
# Optymalna wartość (od 5 do 10)
# Wysoka wartość (np. 50) = słaba selekcja, dużo cech w macierzy
VIF_THRESHOLD = 5

def test_regression(X_train, X_val, y_train, y_val, label=""):
    model = KNeighborsRegressor(n_neighbors=K_NEIGHBORS)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    r2 = r2_score(y_val, y_pred)
    print(f"--- [R2 Score: {label}] -> {r2:.4f}")
    return r2

# ==========================================
# 2. WCZYTANIE I WSTĘPNE CZYSZCZENIE
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
# 3. OBSŁUGA BRAKÓW DANYCH (IMPUTACJA)
# ==========================================
print("\nKROK 2: Imputacja danych...")
it_imp = IterativeImputer(random_state=RANDOM_STATE)
X_train_f = pd.DataFrame(it_imp.fit_transform(X_train), columns=X.columns, index=X_train.index)
X_val_f = pd.DataFrame(it_imp.transform(X_val), columns=X.columns, index=X_val.index)
X_test_f = pd.DataFrame(it_imp.transform(X_test), columns=X.columns, index=X_test.index)
y_train_f, y_val_f, y_test_f = y_train, y_val, y_test

val_r2 = test_regression(X_train_f, X_val_f, y_train_f, y_val_f, "Po regresji")

# ==========================================
# 4. SKOŚNOŚĆ I WYKRES 1 (HISTOGRAMY)
# ==========================================
print("\nKROK 3: Analiza skośności i logarytmowanie...")
# Poprawione wyświetlanie histogramów
X_train_f.hist(figsize=(12, 10), bins=20, color='skyblue', edgecolor='black')
plt.suptitle("Rozkłady cech przed transformacją")

skew_limit = 0.75
skew_values = X_train_f.skew()
to_transform = skew_values[abs(skew_values) > skew_limit].index.tolist()

for col in to_transform:
    # ZABEZPIECZENIE: log1p nie lubi wartości < -1.
    # Imputer mógł stworzyć ujemne PKB. Przycinamy do 0.
    X_train_f[col] = np.log1p(X_train_f[col].clip(lower=0))
    X_val_f[col] = np.log1p(X_val_f[col].clip(lower=0))
    X_test_f[col] = np.log1p(X_test_f[col].clip(lower=0))

# ==========================================
# 5. STANDARYZACJA
# ==========================================
print("\nKROK 4: Standaryzacja...")
scaler = StandardScaler()
# Po skalerze dane stają się macierzami numpy, więc musimy uważać na formaty
X_train_std = scaler.fit_transform(X_train_f)
X_val_std = scaler.transform(X_val_f)
X_test_std = scaler.transform(X_test_f)

val_r2 = test_regression(X_train_std, X_val_std, y_train_f, y_val_f, "Po standaryzacji")

# ==========================================
# 6. REDUKCJA VIF
# ==========================================
print(f"\nKROK 5: Redukcja VIF (Próg: {VIF_THRESHOLD})...")
feature_names = X.columns.tolist()
X_vif_df = pd.DataFrame(X_train_std, columns=feature_names)

while True:
    vif_vals = [variance_inflation_factor(X_vif_df.values, i) for i in range(len(X_vif_df.columns))]
    max_vif = max(vif_vals)
    if max_vif > VIF_THRESHOLD: # Użycie Twojej zmiennej
        idx_to_drop = vif_vals.index(max_vif)
        feat_name = X_vif_df.columns[idx_to_drop]
        print(f"Usuwam: {feat_name} (VIF: {max_vif:.2f})")
        X_vif_df = X_vif_df.drop(columns=[feat_name])
    else:
        break

# Aktualizacja danych do wersji po VIF
remaining_cols = X_vif_df.columns
X_train_final = X_vif_df.values
X_val_final = pd.DataFrame(X_val_std, columns=feature_names)[remaining_cols].values
X_test_final = pd.DataFrame(X_test_std, columns=feature_names)[remaining_cols].values

# ==========================================
# 7. WYKRES 2 (ODLEGŁOŚĆ COOKA)
# ==========================================
print("\nKROK 6: Analiza Cooka...")
X_constant = sm.add_constant(X_train_final)
ols_model = sm.OLS(y_train_f.values, X_constant).fit()
cooks_d = ols_model.get_influence().cooks_distance[0]

plt.figure(figsize=(12, 5))
plt.stem(np.arange(len(cooks_d)), cooks_d, markerfmt=",")
plt.axhline(y=4/len(X_train_final), color='orange', linestyle='--', label='Próg 4/n')
plt.title("Analiza Odległości Cooka")
plt.legend()

safe_idx = np.where(cooks_d <= 4/len(X_train_final))[0]
X_train_final, y_train_final = X_train_final[safe_idx], y_train_f.iloc[safe_idx]

# ==========================================
# 8. WYNIK KOŃCOWY I WYKRES 3 (KORELACJA)
# ==========================================
print("\n" + "="*40)
final_test_r2 = test_regression(X_train_final, X_test_final, y_train_final, y_test_f, "WYNIK OSTATECZNY")
print("="*40)

plt.figure(figsize=(10, 8))
final_corr_df = pd.DataFrame(X_train_final, columns=remaining_cols)
# sns.heatmap(final_corr_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
# Zmieniona linia z dodanym parametrem center=0
sns.heatmap(final_corr_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", center=0)
plt.title(f"Macierz korelacji cech (VIF Threshold: {VIF_THRESHOLD})")

# ==========================================
# 9. FINALNA ANALIZA I INTERPRETACJA (DOPEŁNIENIE PROJEKTU)
# ==========================================
print("\nKROK 7: Generowanie raportu końcowego i ważności cech...")

# A. Wykres Przewidywane vs Rzeczywiste
model_final = KNeighborsRegressor(n_neighbors=K_NEIGHBORS)
model_final.fit(X_train_final, y_train_final)
y_pred_final = model_final.predict(X_test_final)

plt.figure(figsize=(8, 6))
plt.scatter(y_test_f, y_pred_final, alpha=0.5, color='teal')
plt.plot([y_test_f.min(), y_test_f.max()], [y_test_f.min(), y_test_f.max()], 'r--', lw=2)
plt.xlabel("Rzeczywista Długość Życia")
plt.ylabel("Przewidziana Długość Życia")
plt.title(f"Przewidywania vs Rzeczywistość (R2 = {final_test_r2:.4f})")

# B. Analiza Ważności Cech (metoda permutacyjna dla k-NN)
from sklearn.inspection import permutation_importance

result = permutation_importance(model_final, X_test_final, y_test_f, n_repeats=10, random_state=RANDOM_STATE)
sorted_idx = result.importances_mean.argsort()

plt.figure(figsize=(10, 6))
plt.barh(remaining_cols[sorted_idx], result.importances_mean[sorted_idx], color='salmon')
plt.xlabel("Spadek R2 po usunięciu cechy")
plt.title("Ważność Cech (Które dane najbardziej wpływają na model?)")
plt.tight_layout()

print("PROJEKT ZAKOŃCZONY")

print("\nWyświetlanie wykresów... (Zamknij okna, aby zakończyć program)")

plt.show()