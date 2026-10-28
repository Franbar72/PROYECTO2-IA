
import numpy as np


class MyRidge:
    """
    Implementacion propia de regresion Ridge usando NumPy.
    """

    def __init__(self, alpha=1.0):
        if alpha < 0:
            raise ValueError("alpha debe ser mayor o igual que cero")

        self.alpha = alpha

        # Atributos que se calculan durante el entrenamiento
        self.coef_ = None
        self.intercept_ = None
        self.X_mean_ = None
        self.y_mean_ = None

    def fit(self, X, y):
        # Convertir los datos a arreglos numericos de NumPy
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)

        # Validar la forma de los datos
        if X.ndim != 2:
            raise ValueError("X debe ser una matriz bidimensional")

        if X.shape[0] != y.shape[0]:
            raise ValueError(
                "X e y deben tener el mismo numero de filas"
            )

        if X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("X no puede estar vacia")

        # Calcular las medias usando solo los datos de entrenamiento
        self.X_mean_ = X.mean(axis=0)
        self.y_mean_ = y.mean()

        # Centrar las variables
        X_centered = X - self.X_mean_
        y_centered = y - self.y_mean_

        # Construir el sistema de ecuaciones de Ridge
        A = (
            X_centered.T @ X_centered
            + self.alpha * np.eye(X.shape[1])
        )

        b = X_centered.T @ y_centered

        # Calcular los coeficientes sin invertir la matriz explicitamente
        self.coef_ = np.linalg.solve(A, b)

        # Recuperar el intercepto sin penalizarlo
        self.intercept_ = (
            self.y_mean_ - self.X_mean_ @ self.coef_
        )

        return self

    def predict(self, X):
        # Comprobar que el modelo ya fue entrenado
        if self.coef_ is None:
            raise ValueError("Debes llamar fit antes de predict")

        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("X debe ser una matriz bidimensional")

        if X.shape[1] != self.coef_.shape[0]:
            raise ValueError(
                "X debe tener el mismo numero de variables "
                "que los datos de entrenamiento"
            )

        # Usar los coeficientes e intercepto aprendidos en fit
        return X @ self.coef_ + self.intercept_

    def score(self, X, y):
        # Calcular R2, como en los modelos de regresion habituales
        y = np.asarray(y, dtype=float).reshape(-1)
        y_pred = self.predict(X)

        if y.shape[0] != y_pred.shape[0]:
            raise ValueError(
                "X e y deben tener el mismo numero de ejemplos"
            )

        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0

        return 1 - ss_res / ss_tot

    def rmse(self, X, y):
        # Calcular la raiz del error cuadratico medio
        y = np.asarray(y, dtype=float).reshape(-1)
        y_pred = self.predict(X)

        if y.shape[0] != y_pred.shape[0]:
            raise ValueError(
                "X e y deben tener el mismo numero de ejemplos"
            )

        return np.sqrt(np.mean((y - y_pred) ** 2))
