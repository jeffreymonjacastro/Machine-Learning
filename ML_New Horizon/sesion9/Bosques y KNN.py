import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics, datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

digitos = datasets.load_digits()

# Seleccionamos los datos
sample_digitos = digitos.data[:10]

fig, axes = plt.subplots(2,5, figsize=(10,4))

for i, ax in enumerate(axes.flat):
    ax.imshow(sample_digitos[i].reshape(8,8), cmap='binary')
    ax.set_title(f"Digito {i}")
    ax.axis('off')
# plt.show()

x = digitos.data
y = digitos.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=123)

# Creamos el modelo y lo entrenamos con los datos de entrenamiento
modelo = RandomForestClassifier()
modelo.fit(x_train, y_train)

y_pred = modelo.predict(x_test)

cm = metrics.confusion_matrix(y_test, y_pred)
disp = metrics.ConfusionMatrixDisplay(cm, display_labels=digitos.target_names)

# disp.plot(cmap="Blues", values_format="d")
# plt.show()

# print(metrics.classification_report(y_test, y_pred))


########################################

import numpy as np
random_indices = np.random.choice(len(x_test), size=10, replace=False)

fig, axes = plt.subplots(1, 10, figsize=(15, 4))

for i, ax in enumerate(axes):
    index = random_indices[i]
    ax.imshow(x_test[index].reshape(8, 8), cmap='gray')
    ax.set_title(f"Predecir: {y_pred[index]}\nReal:{y_test[index]}")
    ax.axis('off')
# plt.show()


"""
caso 2 random clasification - Billetes
"""

df = pd.read_excel("datasupervisado.xlsx",
                   sheet_name="billeteautenticacion")

x = df.drop("Billete", axis=1)
y = df["Billete"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=123)

modelo = RandomForestClassifier()
modelo.fit(x_train, y_train)

y_pred = modelo.predict(x_test)

cm = metrics.confusion_matrix(y_test, y_pred)
disp = metrics.ConfusionMatrixDisplay(cm, display_labels=y.unique())
# disp.plot(cmap="Blues", values_format="d")
# plt.show()

# print(metrics.classification_report(y_test, y_pred))

feature_importancia_df = pd.DataFrame({
    "Feature": x.columns,
    "Importancia": modelo.feature_importances_})

feature_importancia_df.sort_values(by="Importancia", ascending=False)

import joblib
joblib.dump(modelo, "billete.pkl")
modelo2 = joblib.load("billete.pkl")
nuevadata = pd.read_excel("datasupervisado.xlsx",
                            sheet_name="billetenuevo")
# print(modelo2.predict(nuevadata))


"""
caso 3 - random clasification - vinos
"""

df = pd.read_excel("datasupervisado.xlsx",
                   sheet_name="vino")

x = df.drop("clase", axis=1)
y = df["clase"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=123)

modelo = RandomForestClassifier()
modelo.fit(x_train, y_train)

y_pred = modelo.predict(x_test)

cm = metrics.confusion_matrix(y_test, y_pred)
disp = metrics.ConfusionMatrixDisplay(cm, display_labels=y.unique())
disp.plot(cmap="Blues", values_format="d")
plt.show()

# print(metrics.classification_report(y_test, y_pred))
#
feature_importancia_df = pd.DataFrame({
    "Feature": x.columns,
    "Importancia": modelo.feature_importances_})

feature_importancia_df.sort_values(by="Importancia", ascending=False)

import joblib
joblib.dump(modelo, "vino.pkl")
modelo2 = joblib.load("vino.pkl")
nuevadata = pd.read_excel("datasupervisado.xlsx",
                            sheet_name="vinonuevo")
# print(modelo2.predict(nuevadata))

####################################################

# Seleccion de caracteristicas
valor = 0.07
select_feactures = feature_importancia_df[
    feature_importancia_df["Importancia"] > valor]["Feature"]

x_train_select = x_train[select_feactures]
x_test_select = x_test[select_feactures]

modelo = RandomForestClassifier()
modelo.fit(x_train_select, y_train)

y_pred = modelo.predict(x_test_select)

cm = metrics.confusion_matrix(y_test, y_pred)
disp = metrics.ConfusionMatrixDisplay(cm, display_labels=y.unique())
# disp.plot(cmap="Blues", values_format="d")
# plt.show()

# print(metrics.classification_report(y_test, y_pred))



"""
caso 4 - random regresion - contaminación
"""

from sklearn.ensemble import RandomForestRegressor

df = pd.read_excel("datasupervisado.xlsx",
                   sheet_name="Contaminacion Atmosferica")

x = df.drop("Contaminacion_SO2", axis=1)
y = df["Contaminacion_SO2"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=123)

modelo = RandomForestRegressor()
modelo.fit(x_train, y_train)

y_pred = modelo.predict(x_test)

def metricas(y, y_pred):
    print("Maximo Error:", metrics.max_error(y, y_pred))
    print("Mean Absolute Error:", metrics.mean_absolute_error(y, y_pred))
    print("Mean Squared Error:", metrics.mean_squared_error(y, y_pred))
    print("R2:", metrics.r2_score(y, y_pred))
    print()


# metricas(y_test, y_pred)


from sklearn.model_selection import GridSearchCV

param_grid = {
    "n_estimators": [50, 100, 150],
    "max_depth": [None, 10, 20, 30],
    "min_samples_leaf": [1, 2, 4],
    "min_samples_split": [2, 5, 10],
    "max_features": ["sqrt", "log2"]
}

# grid_search = GridSearchCV(modelo,
#                            param_grid=param_grid,
#                            cv=5,
#                            scoring="neg_mean_squared_error")
#
# grid_search.fit(x_train, y_train)
#
# # print("Mejores hiperparametros:", grid_search.best_params_)
#
# best_modelo = grid_search.best_estimator_
# best_modelo.fit(x_train, y_train)
#
# y_pred = best_modelo.predict(x_test)

# metricas(y_test, y_pred)


"""
caso 5 - random regresion - temperatura
"""

df = pd.read_excel("datasupervisado.xlsx",
                   sheet_name="tiempo")

x = df.drop("Temperature (C)", axis=1)
y = df["Temperature (C)"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=123)

modelo = RandomForestRegressor()
modelo.fit(x_train, y_train)

y_pred = modelo.predict(x_test)

# metricas(y_test, y_pred)

param_grid = {
    "n_estimators": [50, 100, 150],
    "max_depth": [None, 10, 20, 30],
    "min_samples_leaf": [1, 2, 4],
    "min_samples_split": [2, 5, 10],
    "max_features": ["sqrt", "log2"]
}

# grid_search = GridSearchCV(modelo,
#                            param_grid=param_grid,
#                            cv=5,
#                            scoring="neg_mean_squared_error")
#
# grid_search.fit(x_train, y_train)
#
# print("Mejores hiperparametros:", grid_search.best_params_)
#
# best_modelo = grid_search.best_estimator_
# best_modelo.fit(x_train, y_train)
#
# y_pred = best_modelo.predict(x_test)
#
# metricas(y_test, y_pred)


"""
caso 6 - xgbost clasificación
"""

from xgboost import XGBRFClassifier

digitos = datasets.load_digits()
sample_digitos = digitos.data[:10]

fig, axes = plt.subplots(2,5, figsize=(10,4))

for i, ax in enumerate(axes.flat):
    ax.imshow(sample_digitos[i].reshape(8,8), cmap='binary')
    ax.set_title(f"Digito {i}")
    ax.axis('off')
# plt.show()

x = digitos.data
y = digitos.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=123)

modelo = XGBRFClassifier()
modelo.fit(x_train, y_train)

y_pred = modelo.predict(x_test)

cm = metrics.confusion_matrix(y_test, y_pred)
disp = metrics.ConfusionMatrixDisplay(cm, display_labels=digitos.target_names)
# disp.plot(cmap="Blues", values_format="d")
# plt.show()

# print(metrics.classification_report(y_test, y_pred))

import numpy as np
random_indices = np.random.choice(len(x_test), size=10, replace=False)

fig, axes = plt.subplots(1, 10, figsize=(15, 4))

for i, ax in enumerate(axes):
    index = random_indices[i]
    ax.imshow(x_test[index].reshape(8, 8), cmap='gray')
    ax.set_title(f"Predecir: {y_pred[index]}\nReal:{y_test[index]}")
    ax.axis('off')
# plt.show()


from xgboost import XGBClassifier

modelo = XGBClassifier()
modelo.fit(x_train, y_train)

y_pred = modelo.predict(x_test)

cm = metrics.confusion_matrix(y_test, y_pred)
disp = metrics.ConfusionMatrixDisplay(cm, display_labels=digitos.target_names)
disp.plot(cmap="Blues", values_format="d")
plt.show()

print(metrics.classification_report(y_test, y_pred))

