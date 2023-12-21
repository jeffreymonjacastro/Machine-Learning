import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv("https://raw.githubusercontent.com/aurea-soriano/ML-Datasets/master/USA_Housing.csv")

# print(df)

# Visualizar la correlación de todas las columnas entre todas
sns.pairplot(df)
# plt.show()

df = df.drop(["Address"], axis=1)
sns.heatmap(df.corr(), annot=True)
# plt.show()

x = df.drop(["Price"], axis=1)
y = df["Price"]

# dividiendo los datos en entrenamiento y prueba al azar
x_train, x_test, y_train, y_test = train_test_split(x, y,
            test_size=0.3, random_state=123)

modelo = LinearRegression()
modelo.fit(x_train, y_train)

y_pred = modelo.predict(x_test)

df1 = pd.DataFrame({"Actual": y_test.round(2),
                    "Predicción": y_pred.round(2)})

# print(df1)
#
# print("Maximo Error:", metrics.max_error(y_test, y_pred))
# print("Mean Absolute Error:", metrics.mean_absolute_error(y_test, y_pred))
# print("Mean Squared Error:", metrics.mean_squared_error(y_test, y_pred))
# print("R2:", metrics.r2_score(y_test, y_pred))

# joblib es una biblioteca de Python que proporciona funciones para guardar y cargar objetos de Python
import joblib
joblib.dump(modelo, "modeloHouse.pkl")
modelo2 = joblib.load("modeloHouse.pkl")
nuevo = pd.read_excel("BASEDATOS_MODELO LINEAL.xlsx",
                      sheet_name="nuevoHouse")
modelo2.predict(nuevo)


import statsmodels.api as sm
x_train = sm.add_constant(x_train)
modelo = sm.OLS(y_train, x_train)
resultados = modelo.fit()
# print(resultados.summary())


"""
caso 2
"""

df = pd.read_excel("BASEDATOS_MODELO LINEAL.xlsx",
                      sheet_name="SUELDO")

sns.pairplot(df)
# plt.show()

sns.heatmap(df.corr(), annot=True)
# plt.show()

# no se considera salario_actual porque es el que se quiere predecir
x = df.drop(["id", "salario_actual"], axis=1)
y = df["salario_actual"]

x_train, x_test, y_train, y_test = train_test_split(x, y,
            test_size=0.25, random_state=100)

modelo = LinearRegression()
modelo.fit(x_train, y_train)

y_pred = modelo.predict(x_test)

df1 = pd.DataFrame({"Actual": y_test.round(2),
                    "Predicción": y_pred.round(2)})

# print(df1)

# print("Maximo Error:", metrics.max_error(y_test, y_pred))
# print("Mean Absolute Error:", metrics.mean_absolute_error(y_test, y_pred))
# print("Mean Squared Error:", metrics.mean_squared_error(y_test, y_pred))
# print("R2:", metrics.r2_score(y_test, y_pred))

joblib.dump(modelo, "modeloSueldo.pkl")
modelo2 = joblib.load("modeloSueldo.pkl")
nuevo = pd.read_excel("BASEDATOS_MODELO LINEAL.xlsx",
                      sheet_name="nuevoSUELDO")
# print(modelo2.predict(nuevo))

x_train = sm.add_constant(x_train)
modelo = sm.OLS(y_train, x_train)
modelo = modelo.fit()
# print(modelo.summary())


""" no considerar la columna sexo """

x = df.drop(["id", "salario_actual", "sexo"], axis=1)
y = df["salario_actual"]

x_train, x_test, y_train, y_test = train_test_split(x, y,
            test_size=0.25, random_state=100)

modelo = LinearRegression()
modelo.fit(x_train, y_train)

y_pred = modelo.predict(x_test)

df1 = pd.DataFrame({"Actual": y_test.round(2),
                    "Predicción": y_pred.round(2)})

# print(df1)

# print("Maximo Error:", metrics.max_error(y_test, y_pred))
# print("Mean Absolute Error:", metrics.mean_absolute_error(y_test, y_pred))
# print("Mean Squared Error:", metrics.mean_squared_error(y_test, y_pred))
# print("R2:", metrics.r2_score(y_test, y_pred))

joblib.dump(modelo, "modeloSueldo.pkl")
modelo2 = joblib.load("modeloSueldo.pkl")
nuevo = pd.read_excel("BASEDATOS_MODELO LINEAL.xlsx",
                      sheet_name="nuevoSUELDO")
# print(modelo2.predict(nuevo))

x_train = sm.add_constant(x_train)
modelo = sm.OLS(y_train, x_train)
modelo = modelo.fit()
print(modelo.summary())