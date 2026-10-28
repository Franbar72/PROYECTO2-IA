
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score


# ==================================================
# 1. CARGAR LOS DATOS LIMPIOS
# ==================================================

ruta = Path("data/concrete_data_limpio.csv")

if not ruta.exists():
    raise FileNotFoundError(
        f"No se encontro el archivo: {ruta.resolve()}"
    )

df = pd.read_csv(ruta)

print("DATOS CARGADOS")
print("Filas:", df.shape[0])
print("Columnas:", df.shape[1])


# ==================================================
# 2. SEPARAR PREDICTORES Y VARIABLE OBJETIVO
# ==================================================

objetivo = "concrete_compressive_strength"

X = df.drop(columns=[objetivo])
y = df[objetivo]

print("\nVARIABLES PREDICTORAS:", X.shape[1])
print("VARIABLE OBJETIVO:", objetivo)


# ==================================================
# 3. DIVIDIR EN ENTRENAMIENTO Y PRUEBA
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTAMANO DE LOS CONJUNTOS")
print("Entrenamiento:", X_train.shape[0])
print("Prueba:", X_test.shape[0])


# ==================================================
# 4. ENTRENAR UNA REGRESION LINEAL NORMAL
# ==================================================

modelo_lineal = LinearRegression()

modelo_lineal.fit(X_train, y_train)

pred_train_lineal = modelo_lineal.predict(X_train)
pred_test_lineal = modelo_lineal.predict(X_test)

rmse_train_lineal = mean_squared_error(
    y_train, pred_train_lineal
) ** 0.5

rmse_test_lineal = mean_squared_error(
    y_test, pred_test_lineal
) ** 0.5

print("\n========== REGRESION LINEAL NORMAL ==========")
print(f"RMSE de entrenamiento: {rmse_train_lineal:.4f}")
print(f"RMSE de prueba:        {rmse_test_lineal:.4f}")

print(
    f"R2 de entrenamiento:   "
    f"{r2_score(y_train, pred_train_lineal):.4f}"
)
print(
    f"R2 de prueba:          "
    f"{r2_score(y_test, pred_test_lineal):.4f}"
)


# ==================================================
# 5. REGRESION CON CARACTERISTICAS POLINOMICAS
# ==================================================

modelo_polinomico = make_pipeline(
    PolynomialFeatures(degree=4, include_bias=False),
    LinearRegression()
)

modelo_polinomico.fit(X_train, y_train)

pred_train_polinomico = modelo_polinomico.predict(X_train)
pred_test_polinomico = modelo_polinomico.predict(X_test)

rmse_train_polinomico = mean_squared_error(
    y_train, pred_train_polinomico
) ** 0.5

rmse_test_polinomico = mean_squared_error(
    y_test, pred_test_polinomico
) ** 0.5

print("\n========== REGRESION POLINOMICA ==========")
print(f"RMSE de entrenamiento: {rmse_train_polinomico:.4f}")
print(f"RMSE de prueba:        {rmse_test_polinomico:.4f}")

print(
    f"R2 de entrenamiento:   "
    f"{r2_score(y_train, pred_train_polinomico):.4f}"
)
print(
    f"R2 de prueba:          "
    f"{r2_score(y_test, pred_test_polinomico):.4f}"
)


# ==================================================
# 6. COMPARAR LOS RESULTADOS
# ==================================================

resultados = pd.DataFrame({
    "Modelo": [
        "Regresion lineal normal",
        "Regresion polinomica grado 4"
    ],
    "RMSE entrenamiento": [
        rmse_train_lineal,
        rmse_train_polinomico
    ],
    "RMSE prueba": [
        rmse_test_lineal,
        rmse_test_polinomico
    ]
})

print("\n========== COMPARACION DE MODELOS ==========")
print(resultados.round(4).to_string(index=False))


# ==================================================
# 7. INTERPRETAR EL POSIBLE SOBREAJUSTE
# ==================================================

diferencia = rmse_test_polinomico - rmse_train_polinomico

print("\n========== ANALISIS DEL SOBREAJUSTE ==========")

if diferencia > 1:
    print(
        "El modelo polinomico tiene un error de prueba "
        "mayor que su error de entrenamiento."
    )
    print(
        "Esto puede indicar sobreajuste; revisa tambien "
        "la magnitud de ambos errores."
    )
else:
    print(
        "La diferencia entre los RMSE del modelo polinomico "
        "no es grande segun este criterio."
    )
    print(
        "No podemos afirmar que exista un sobreajuste "
        "importante solo con esta comparacion."
    )
