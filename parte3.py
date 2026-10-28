import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge as SklearnRidge
from sklearn.metrics import mean_squared_error
from my_ridge import MyRidge

# 1. Cargar datos limpios
df = pd.read_csv("concrete_data_limpio.csv")
objetivo = "concrete_compressive_strength"
X = df.drop(columns=[objetivo]).values
y = df[objetivo].values

# 2. Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# 3. Estandarizar features (ajustar solo en train)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Escoger alpha por validación (Hold-out / Grilla)
alphas = np.logspace(-4, 4, 50)
train_rmse_list = []
val_rmse_list = []

# Dividir train en sub-entrenamiento y validación para la curva
X_sub_train, X_val, y_sub_train, y_val = train_test_split(X_train_scaled, y_train, test_size=0.20, random_state=42)

for a in alphas:
    modelo = MyRidge(alpha=a)
    modelo.fit(X_sub_train, y_sub_train)
    train_rmse_list.append(modelo.rmse(X_sub_train, y_sub_train))
    val_rmse_list.append(modelo.rmse(X_val, y_val))

# Graficar RMSE vs Alpha
plt.figure(figsize=(8, 5))
plt.plot(alphas, train_rmse_list, label="RMSE Entrenamiento", color="blue")
plt.plot(alphas, val_rmse_list, label="RMSE Validación", color="orange")
plt.xscale("log")
plt.xlabel("Alpha (escala logarítmica)")
plt.ylabel("RMSE")
plt.title("Evolución del RMSE según Alpha (Regularización Ridge)")
plt.legend()
plt.grid(True)
plt.show()

# Seleccionar el mejor alpha (el que tenga menor RMSE de validación)
best_alpha = alphas[np.argmin(val_rmse_list)]
print(f"Mejor Alpha seleccionado: {best_alpha:.4f}")

# 5. Modelo final con el mejor alpha
modelo_final = MyRidge(alpha=best_alpha)
modelo_final.fit(X_train_scaled, y_train)
rmse_prueba_final = modelo_final.rmse(X_test_scaled, y_test)
print(f"RMSE de Prueba del Modelo Final (MyRidge): {rmse_prueba_final:.4f}")

# 6. Verificación estricta con np.allclose frente a sklearn Ridge
sk_ridge = SklearnRidge(alpha=best_alpha)
sk_ridge.fit(X_train_scaled, y_train)

print("\n--- VERIFICACIÓN CON SKLEARN (ESTANDARIZADO) ---")
print("Coeficientes coinciden:", np.allclose(modelo_final.coef_, sk_ridge.coef_))
print("Intercepto coincide:", np.allclose(modelo_final.intercept_, sk_ridge.intercept_))
print("R2 coincide:", np.allclose(modelo_final.score(X_test_scaled, y_test), sk_ridge.score(X_test_scaled, y_test)))
print("RMSE coincide:", np.allclose(modelo_final.rmse(X_test_scaled, y_test), np.sqrt(mean_squared_error(y_test, sk_ridge.predict(X_test_scaled)))))