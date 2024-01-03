import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import datasets


"""
knn classificación
"""

from sklearn.neighbors import KNeighborsClassifier

data = pd.read_excel("data8.xlsx", sheet_name="CreditoScore")
data.columns

label_encoder = LabelEncoder()
data["Genero"] = label_encoder.fit_transform(data["Genero"])
data["Educacion"] = label_encoder.fit_transform(data["Educacion"])
data["EstadoCivil"] = label_encoder.fit_transform(data["EstadoCivil"])
data["PropiedadCasa"] = label_encoder.fit_transform(data["PropiedadCasa"])

x = data.drop("PuntajeCredito", axis=1)
y = data["PuntajeCredito"]

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    test_size=0.2, random_state=123)

best_k = None
best_accuracy = 0
for k in range(2, 8):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    # print("K :", k, "--> Accuracy es ", accuracy)

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_k = k

# print(f"El mejor valor de k es {best_k}")

knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)

cm = metrics.confusion_matrix(y_test, y_pred)
disp = metrics.ConfusionMatrixDisplay(cm, display_labels=y.unique())
disp.plot(cmap="Blues", values_format="d")

# print(metrics.classification_report(y_test, y_pred))


"""
knn regresion
"""

from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV

url = "https://raw.githubusercontent.com/rpizarrog/Analisis-Inteligente-de-datos/main/datos/Advertising_Web.csv"

datos = pd.read_csv(url)

datos = datos[['TV', 'Radio', 'Newspaper', 'Web', 'Sales']]

x = datos[['TV', 'Radio', 'Newspaper', 'Web']]
y = datos["Sales"]

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    test_size=0.2, random_state=123)

param_grid={
    "n_neighbors":list(range(2,11)),
    "weights":["uniform", "distance"],
    "metric":["euclidean","manhattan"],
    "algorithm":["auto","ball_tree","kd_tree","brute"]
    }

knn = KNeighborsRegressor()

grid_search = GridSearchCV(knn, param_grid, cv=5,
                           scoring ="neg_mean_squared_error")
grid_search.fit(x_train, y_train)

# print("Mejores Hiperparametros:", grid_search.best_params_)

best_knn = grid_search.best_estimator_
best_knn.fit(x_train, y_train)

y_pred = best_knn.predict(x_test)

mse = metrics.mean_squared_error(y_test, y_pred)
r2 = metrics.r2_score(y_test, y_pred)

# print("MSE:", mse)
# print("R2:", r2)

import joblib
joblib.dump(best_knn, "publicidad.pkl")
modelo = joblib.load("publicidad.pkl")

nuevo = pd.read_excel("data8.xlsx", sheet_name="publicidad")
modelo.predict(nuevo)
