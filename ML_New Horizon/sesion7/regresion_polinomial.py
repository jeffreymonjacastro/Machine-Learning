import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib
import statsmodels.api as sm

# funcion para graficar
def grafica(x, y, y_pred):
    plt.scatter(x, y, color="blue")
    plt.plot(x, y_pred, color="red")
    plt.show()

## funcion para imprimir metricas
def metricas(y, y_pred):
    print("Maximo Error:", metrics.max_error(y, y_pred))
    print("Mean Absolute Error:", metrics.mean_absolute_error(y, y_pred))
    print("Mean Squared Error:", metrics.mean_squared_error(y, y_pred))
    print("R2:", metrics.r2_score(y, y_pred))
    print()


df = pd.read_excel("BASEDATOS_MODELO LINEAL.xlsx",
            sheet_name="ELEMENTO")

x = df.drop("RESISTENCIA", axis=1)
y = df["RESISTENCIA"]

# plt.scatter(x,y)
# plt.show()

# modelo lineal
modelo = LinearRegression()
modelo.fit(x,y)
y_pred = modelo.predict(x)

# metricas(y, y_pred)
# grafica(x, y, y_pred)

# elevo al cuadrado
from sklearn.preprocessing import PolynomialFeatures

grado = PolynomialFeatures(degree=2)
x_poly = grado.fit_transform(x)
modelo.fit(x_poly, y)
y_pred = modelo.predict(x_poly)

# metricas(y, y_pred)
# grafica(x, y, y_pred)

# elevo al cubo
grado = PolynomialFeatures(degree=3)
x_poly = grado.fit_transform(x)
modelo.fit(x_poly, y)
y_pred = modelo.predict(x_poly)

# metricas(y, y_pred)
# grafica(x, y, y_pred)

# elevo al 4
grado = PolynomialFeatures(degree=4)
x_poly = grado.fit_transform(x)
modelo.fit(x_poly, y)
y_pred = modelo.predict(x_poly)

# metricas(y, y_pred)
# grafica(x, y, y_pred)

joblib.dump(modelo, "modeloPoly.pkl")
modelo2 = joblib.load("modeloPoly.pkl")
# print(modelo2.predict(x_poly))

"""
caso 4 modelo polinomial
"""

df = pd.read_csv("https://raw.githubusercontent.com/aurea-soriano/ML-Datasets/master/USA_Housing.csv")

x = df.drop(["Price", "Address"], axis=1)
y = df["Price"]

x_train, x_test, y_train, y_test = train_test_split(x, y,
            test_size=0.3, random_state=123)

for grado in range(2,7):
    print("---- Modelo Grado " + str(grado) + " ----")
    modelo = LinearRegression()
    grado = PolynomialFeatures(degree=grado)
    x_train_poly = grado.fit_transform(x_train)
    x_test_poly = grado.fit_transform(x_test)
    modelo.fit(x_train_poly, y_train)
    y_pred = modelo.predict(x_test_poly)
    metricas(y_test, y_pred)

# Nos quedamos con el de grado 2
modelo = LinearRegression()
grado = PolynomialFeatures(degree=2)
x_train_poly = grado.fit_transform(x_train)
x_test_poly = grado.fit_transform(x_test)
modelo.fit(x_train_poly, y_train)
y_pred = modelo.predict(x_test_poly)

joblib.dump(modelo, "modelopolyHouse.pkl")
modelo2 = joblib.load("modelopolyHouse.pkl")
nuevo = pd.read_excel("BASEDATOS_MODELO LINEAL.xlsx",
                      sheet_name="nuevoHouse")
nuevoPoly = grado.fit_transform(nuevo)
print(modelo2.predict(nuevoPoly))