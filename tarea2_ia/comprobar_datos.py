
from pathlib import Path
import pandas as pd

# Buscar el archivo CSV dentro de la carpeta data
ruta = Path("data/concrete_data.csv")

# Comprobar que el archivo existe
if not ruta.exists():
    print("ERROR: No se encuentra el archivo:", ruta)
else:
    # Leer el conjunto de datos
    df = pd.read_csv(ruta)

    print("Archivo cargado correctamente")
    print("\nDimensiones (filas, columnas):")
    print(df.shape)

    print("\nNombres de las columnas:")
    print(df.columns.tolist())

    print("\nPrimeras 5 filas:")
    print(df.head())

    print("\nTipos de datos:")
    print(df.dtypes)
