## CLASIFICACIÓN

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

url = "https://raw.githubusercontent.com/JoaquinAmatRodrigo/Estadistica-machine-learning-python/master/data/spam.csv"

datos = pd.read_csv(url)

x = datos.drop("type", axis=1)
y = datos["type"]

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                test_size=0.2, random_state=100)

modelo = LogisticRegression()
modelo.fit(x_train, y_train)
y_pred = modelo.predict(x_test)

# cm = metrics.confusion_matrix(y_test, y_pred)
# metrics.ConfusionMatrixDisplay(cm, display_labels=y.unique()).plot()
# plt.show()

# print(metrics.classification_report(y_test, y_pred)

import joblib
joblib.dump(modelo, "modeloCorreo.pkl")
modelo2 = joblib.load("modeloCorreo.pkl")
# print(modelo2.predict(x_test))


"""
caso 2
No hay data :C
"""

# df = pd.read_excel("datalineal.xlsx", sheet_name="salud")
#
# x = df.drop(columns="DEATH_EVENT")
# y = df["DEATH_EVENT"]
#
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25,
#                                                     random_state=23)
#
# modelo = LogisticRegression()
# modelo.fit(x_train, y_train)
# y_pred = modelo.predict(x_test)
#
# cm = metrics.confusion_matrix(y_test, y_pred)
# metrics.ConfusionMatrixDisplay(cm, display_labels=y.unique()).plot()
#
# print(metrics.classification_report(y_test, y_pred))


"""
caso 3
"""

iris = sns.load_dataset("iris")
# print(iris)

x = iris.drop("species", axis=1)
y = iris["species"]

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                test_size=0.25, random_state=100)

modelo = LogisticRegression()
modelo.fit(x_train, y_train)
y_pred = modelo.predict(x_test)

cm = metrics.confusion_matrix(y_test, y_pred)
metrics.ConfusionMatrixDisplay(cm, display_labels=y.unique()).plot()
# plt.show()

# print(metrics.classification_report(y_test, y_pred))

nueva_data = pd.DataFrame({
    'sepal_length': [6.2,5.5,4.9],
    'sepal_width': [2.8,3.4,3.1],
    'petal_length': [4.8,.5,1.6],
    'petal_width': [1.8,0.2,0.2]
})

# print(modelo.predict(nueva_data))

joblib.dump(modelo, "irisLogistico.pkl")
modelo2 = joblib.load("irisLogistico.pkl")

# nuevo = pd.read_excel("datalineal.xlsx", sheet_name="iris")
# print(modelo2.predict(nuevo))



"""
caso 4 arbol de clasificacion
"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree, export_text
iris = sns.load_dataset("iris")

x = iris.drop("species", axis=1)
y = iris["species"]

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                test_size=0.25, random_state=100)

modelo = DecisionTreeClassifier()
modelo.fit(x_train, y_train)

y_pred = modelo.predict(x_test)

# # print(export_text(modelo,
#                   feature_names=x.columns.tolist(),
#                   show_weights=True))

plt.figure(figsize=(12,8))
plot_tree(modelo,
          feature_names=x.columns.tolist(),
          class_names=modelo.classes_,
          filled=True)
# plt.show()

cm = metrics.confusion_matrix(y_test, y_pred)
metrics.ConfusionMatrixDisplay(cm, display_labels=y.unique()).plot()
# plt.show()

# print(metrics.classification_report(y_test, y_pred))

#########################################################

joblib.dump(modelo, "irisArbol.pkl")
modelo2 = joblib.load("irisArbol.pkl")

# nuevo = pd.read_excel("datalineal.xlsx", sheet_name="iris")
# print(modelo2.predict(nuevo))

#########################################################

# featura_importancia = modelo.feature_importances_
#
# # for feature,importancia in zip(x.columns.tolist(), featura_importancia):
# #     print(f'{feature} : {importancia}')
#
# valor = 0.05
# x = importancia_feature = x.columns[featura_importancia > valor]
# x_importante = x[importancia_feature]
#
# x_train_importante, x_test_importante = train_test_split(x_importante, test_size=0.3, random_state=100)
#
# modelo = DecisionTreeClassifier()
# modelo.fit(x_train, y_train)
#
# y_pred = modelo.predict(x_test)
#
# cm = metrics.confusion_matrix(y_test, y_pred)
# metrics.ConfusionMatrixDisplay(cm, display_labels=y.unique()).plot()


"""
caso 5
"""

url = "https://raw.githubusercontent.com/XavierCarrera/Tutorial-Machine-Learning-Arboles/main/Fish.csv"
df = pd.read_csv(url)

x = df.drop("Species", axis=1)
y = df["Species"]

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                test_size=0.3, random_state=100)

modelo = DecisionTreeClassifier()
modelo.fit(x_train, y_train)

y_pred = modelo.predict(x_test)

# print(export_text(modelo,
#                   feature_names=x.columns.tolist()))

plt.figure(figsize=(12,8))
plot_tree(modelo,
          feature_names=x.columns.tolist(),
          class_names=modelo.classes_,
          filled=True)
# plt.show()

cm = metrics.confusion_matrix(y_test, y_pred)
metrics.ConfusionMatrixDisplay(cm, display_labels=y.unique()).plot()
# plt.show()

# print(metrics.classification_report(y_test, y_pred))

from sklearn.model_selection import GridSearchCV

param_grid = {
    "criterion": ["gini", "entropy"],
    "max_depth": [None, 5, 10, 15, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}

modelo = DecisionTreeClassifier()
grid_search = GridSearchCV(modelo, param_grid=param_grid, cv=5)
grid_search.fit(x_train, y_train)

best_model = grid_search.best_estimator_
# print(best_model)
# print(export_text(best_model,
#                     feature_names=x.columns.tolist()))

plt.figure(figsize=(10,8))
plot_tree(best_model,
          feature_names=x.columns.tolist(),
          class_names=df["Species"].unique(),
          filled=True)
# plt.show()

y_pred = best_model.predict(x_test)

cm = metrics.confusion_matrix(y_test, y_pred)
metrics.ConfusionMatrixDisplay(cm, display_labels=y.unique()).plot()
# plt.show()

print(metrics.classification_report(y_test, y_pred))


"""
caso 6
"""

from sklearn.tree import DecisionTreeRegressor

url = "https://raw.githubusercontent.com/rpizarrog/Analisis-Inteligente-de-datos/main/datos/Advertising_Web.csv"

datos = pd.read_csv(url)

datos[['TV', 'Radio', 'Newspaper', 'Web', 'Sales']]

x = datos.drop("Sales", axis=1)
y = datos["Sales"]

x_train, x_test, y_train, y_test = train_test_split(x, y,
                    test_size=0.25, random_state=100)

param_grid = {
    "max_depth": [None, 3, 5, 7, 6],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}

grid_search = GridSearchCV(DecisionTreeRegressor(),
                        param_grid=param_grid,
                        scoring="neg_mean_squared_error",
                        cv=5)

grid_search.fit(x_train, y_train)

mejor_hiperparametro = grid_search.best_params_

modelo_mejorado = DecisionTreeRegressor(**mejor_hiperparametro)
modelo_mejorado.fit(x_train, y_train)
y_pred = modelo_mejorado.predict(x_test)

print(export_text(modelo_mejorado,
                    feature_names=x.columns.tolist()))

plt.figure(figsize=(12,6))
plot_tree(modelo_mejorado,
            feature_names=x.columns.tolist(),
            filled=True)
# plt.show()

def metricas(y, y_pred):
    print("Maximo Error:", metrics.max_error(y, y_pred))
    print("Mean Absolute Error:", metrics.mean_absolute_error(y, y_pred))
    print("Mean Squared Error:", metrics.mean_squared_error(y, y_pred))
    print("R2:", metrics.r2_score(y, y_pred))
    print()

# metricas(y_test, y_pred)

joblib.dump(modelo_mejorado, "modeloPublicidad.pkl")
modelo2 = joblib.load("modeloPublicidad.pkl")

