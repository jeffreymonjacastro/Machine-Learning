import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn import datasets

"""
svm clasificacion
"""

from sklearn.svm import SVC

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

best_kernel = None
best_accuracy = 0
kernels = ["linear", "poly", "rbf", "sigmoid"]

for kernel in kernels:
    svm = SVC(kernel=kernel)
    svm.fit(x_train, y_train)
    y_pred = svm.predict(x_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)

    print(f"Kernel: {kernel} --> Accuracy: {accuracy}")
    print(metrics.classification_report(y_test, y_pred))

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_kernel = kernel

print("Mejor Kernel:", best_kernel)
print("Mejor Accuracy:", best_accuracy)

svm = SVC(kernel="linear")
svm.fit(x_train, y_train)
y_pred = svm.predict(x_test)

cm = metrics.cofusion_matrix(y_test, y_pred)
disp = metrics.ConfusionMatrixDisplay(cm, display_labels=y.unique())
disp.plot(cmap="Blues", values_format="d")

print(metrics.classification_report(y_test, y_pred))

feature_importance = abs(svm.coef_)
feature_importance = feature_importance[0]
feature_names = x.columns

importanciaDF = pd.DataFrame({
    "feature": feature_names,
    "importance": feature_importance})

importanciaDF = importanciaDF.sort_values(by="importance", ascending=False)


"""
svm regression
"""

from sklearn.svm import SVR

url = "https://raw.githubusercontent.com/rpizarrog/Analisis-Inteligente-de-datos/main/datos/Advertising_Web.csv"

datos = pd.read_csv(url)

datos = datos[['TV', 'Radio', 'Newspaper', 'Web', 'Sales']]

x = datos[['TV', 'Radio', 'Newspaper', 'Web']]
y = datos["Sales"]

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    test_size=0.2, random_state=123)

param_grid = {
    "kernel": ["linear", "poly", "rbf", "sigmoid"],
    "C": [0.1, 1, 10, 100],
    "epsilon": [0.1, 0.2, 0.3, 0.4],
    }

svm = SVR()

grid_search = GridSearchCV(svm, param_grid, cv=5,
                           scoring ="neg_mean_squared_error")
grid_search.fit(x_train, y_train)

# print("Mejores Hiperparametros:", grid_search.best_params_)

best_svm = grid_search.best_estimator_
best_svm.fit(x_train, y_train)

y_pred = best_svm.predict(x_test)

mse = metrics.mean_squared_error(y_test, y_pred)
r2 = metrics.r2_score(y_test, y_pred)

# print("MSE:", mse)
# print("R2:", r2)

import joblib
joblib.dump(best_svm, "publicidad.pkl")
modelo = joblib.load("publicidad.pkl")

nuevo = pd.read_excel("data8.xlsx", sheet_name="publicidad")
modelo.predict(nuevo)
