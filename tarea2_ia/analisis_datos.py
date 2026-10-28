
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# ==================================================
# 1. CARGAR EL CONJUNTO DE DATOS
# ==================================================

ruta = Path("data/concrete_data.csv")

if not ruta.exists():
    raise FileNotFoundError(
        f"No se encontro el archivo: {ruta.resolve()}"
    )

df = pd.read_csv(ruta)

print("ARCHIVO CARGADO CORRECTAMENTE")
print("Dimensiones originales:", df.shape)


# ==================================================
# 2. LIMPIAR LOS NOMBRES DE LAS COLUMNAS
# ==================================================

df.columns = df.columns.str.strip()

print("\nNOMBRES DE LAS COLUMNAS:")
print(df.columns.tolist())


# ==================================================
# 3. COMPROBAR LOS TIPOS DE DATOS
# ==================================================

print("\nTIPOS DE DATOS:")
print(df.dtypes)


# ==================================================
# 4. COMPROBAR LOS VALORES AUSENTES
# ==================================================

print("\nVALORES AUSENTES POR COLUMNA:")
print(df.isnull().sum())

total_ausentes = df.isnull().sum().sum()
print("Total de valores ausentes:", total_ausentes)


# ==================================================
# 5. DETECTAR Y ELIMINAR DUPLICADOS EXACTOS
# ==================================================

duplicados = df.duplicated().sum()

print("\nDUPLICADOS EXACTOS EN LOS DATOS ORIGINALES:")
print(duplicados)

df = df.drop_duplicates().copy()

print("\nDIMENSIONES DESPUES DE ELIMINAR DUPLICADOS:")
print(df.shape)

print("\nDUPLICADOS DESPUES DE LA LIMPIEZA:")
print(df.duplicated().sum())


# ==================================================
# 6. MOSTRAR ESTADISTICAS DESCRIPTIVAS
# ==================================================

print("\nRESUMEN ESTADISTICO:")
print(df.describe().round(2))


# ==================================================
# 7. IDENTIFICAR LAS VARIABLES PREDICTORAS Y EL OBJETIVO
# ==================================================

objetivo = "concrete_compressive_strength"

X = df.drop(columns=[objetivo])
y = df[objetivo]

print("\nNUMERO DE VARIABLES PREDICTORAS:", X.shape[1])
print("NUMERO DE VALORES OBJETIVO:", y.shape[0])
print("NOMBRE DE LA VARIABLE OBJETIVO:", objetivo)


# ==================================================
# 8. GUARDAR LOS DATOS LIMPIOS
# ==================================================

ruta_limpia = Path("data/concrete_data_limpio.csv")

df.to_csv(ruta_limpia, index=False)

print("\nDATOS LIMPIOS GUARDADOS EN:")
print(ruta_limpia.resolve())


# ==================================================
# 9. GRAFICAR LA DISTRIBUCION DE LA VARIABLE OBJETIVO
# ==================================================

plt.figure(figsize=(8, 5))

plt.hist(y, bins=20, edgecolor="black")

plt.title("Distribucion de la resistencia del concreto")
plt.xlabel("Resistencia a la compresion")
plt.ylabel("Frecuencia")

plt.tight_layout()
plt.show()


# ==================================================
# 10. RESUMEN FINAL
# ==================================================

print("\nRESUMEN FINAL")
print("Filas originales: 1030")
print("Filas despues de limpiar:", df.shape[0])
print("Columnas:", df.shape[1])
print("Variables predictoras:", X.shape[1])
print("Valores ausentes:", df.isnull().sum().sum())
print("Duplicados restantes:", df.duplicated().sum())