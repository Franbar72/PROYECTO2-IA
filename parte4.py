import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso, Ridge as SklearnRidge
from sklearn.model_selection import GridSearchCV

# 1. Encontrar mejores Alpha por GridSearchCV para Ridge y Lasso
alphas = np.logspace(-4, 4, 50)

ridge_cv = GridSearchCV(SklearnRidge(), {'alpha': alphas}, cv=5, scoring='neg_root_mean_squared_error')
ridge_cv.fit(X_train_scaled, y_train)

lasso_cv = GridSearchCV(Lasso(max_iter=10000), {'alpha': alphas}, cv=5, scoring='neg_root_mean_squared_error')
lasso_cv.fit(X_train_scaled, y_train)

best_ridge_alpha = ridge_cv.best_params_['alpha']
best_lasso_alpha = lasso_cv.best_params_['alpha']

# Entrenar modelos finales
ridge_model = SklearnRidge(alpha=best_ridge_alpha).fit(X_train_scaled, y_train)
lasso_model = Lasso(alpha=best_lasso_alpha, max_iter=10000).fit(X_train_scaled, y_train)

# 2. Tabla comparativa de coeficientes
columnas_features = df.drop(columns=[objetivo]).columns
df_coefs = pd.DataFrame({
    "Caracteristica": columnas_features,
    "MyRidge": modelo_final.coef_,
    "Ridge_SK": ridge_model.coef_,
    "Lasso_SK": lasso_model.coef_
})

print("\n========== TABLA COMPARATIVA DE COEFICIENTES ==========")
display(df_coefs.round(4))

# 3. Analizar ceros en Lasso
coefs_cero = np.sum(lasso_model.coef_ == 0)
features_eliminadas = columnas_features[lasso_model.coef_ == 0]
print(f"\nNúmero de coeficientes llevados a cero por Lasso: {coefs_cero}")
print(f"Features eliminadas por Lasso: {list(features_eliminadas)}")

# 4. Caminos de coeficientes (Coefficient Paths)
ridge_coefs_path = []
lasso_coefs_path = []

for a in alphas:
    r = SklearnRidge(alpha=a).fit(X_train_scaled, y_train)
    l = Lasso(alpha=a, max_iter=10000).fit(X_train_scaled, y_train)
    ridge_coefs_path.append(r.coef_)
    lasso_coefs_path.append(l.coef_)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(alphas, ridge_coefs_path)
axes[0].set_xscale("log")
axes[0].set_title("Camino de Coeficientes - Ridge (L2)")
axes[0].set_xlabel("Alpha")
axes[0].set_ylabel("Valor del Coeficiente")
axes[0].grid(True)

axes[1].plot(alphas, lasso_coefs_path)
axes[1].set_xscale("log")
axes[1].set_title("Camino de Coeficientes - Lasso (L1)")
axes[1].set_xlabel("Alpha")
axes[1].set_ylabel("Valor del Coeficiente")
axes[1].grid(True)

plt.tight_layout()
plt.show()