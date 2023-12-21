## Regresión lineal
# Importar la biblioteca pandas y se le asigna el alias 'pd'
import pandas as pd

# Importar módulos específicos de scikit-learn
# En este caso, importamos la submódulo 'metrics' que contiene funciones para evaluar el rendimiento del modelo
from sklearn import metrics

# Importar la biblioteca seaborn y se le asigna el alias 'sns'
# Seaborn es una biblioteca de visualización de datos basada en Matplotlib que proporciona una interfaz de alto nivel para gráficos atractivos y informativos.
import seaborn as sns

# Importar el módulo pyplot de Matplotlib y se le asigna el alias 'plt'
# pyplot proporciona funciones para crear gráficos y visualizaciones interactivas
from matplotlib import pyplot as plt

# Importar la función train_test_split de scikit-learn
# Esta función se utiliza para dividir un conjunto de datos en conjuntos de entrenamiento y prueba
from sklearn.model_selection import train_test_split

# Importar la clase LinearRegression de scikit-learn
# LinearRegression se utiliza para entrenar modelos de regresión lineal
from sklearn.linear_model import LinearRegression

# Importar la biblioteca statsmodels y se le asigna el alias 'sm'
# statsmodels es una biblioteca de Python que proporciona clases y funciones para la estimación de muchos modelos estadísticos diferentes, así como para realizar pruebas estadísticas y explorar datos estadísticos
import statsmodels.api as sm


df = pd.read_excel("BASEDATOS_MODELO LINEAL.xlsx",
                   sheet_name="WOMEN")

plt.scatter(df["Altura"], df["Peso"])
# plt.show()

# sns.heatmap(df.corr(), annot=True)
# plt.show()

x = df[["Altura"]]
y = df["Peso"]

## Datos de entrenamiento y prueba
# train_test_split divide los datos en dos subconjuntos: entrenamiento y prueba
# El parámetro test_size especifica el tamaño del conjunto de prueba
# El parámetro random_state establece una semilla para generar números aleatorios
x_train, x_test, y_train, y_test = train_test_split(x, y,
            test_size=0.2, random_state=123)

## Crear el modelo lineal
modelo = LinearRegression()

## Entrenar el modelo
modelo.fit(x_train, y_train)

## Predecir
y_pred = modelo.predict(x_test)

## Evaluar el modelo
df1 = pd.DataFrame({"Actual": y_test.round(2),
                    "Predicción": y_pred.round(2)})

print(x_test["Altura"])

plt.scatter(x_test, y_test, color="blue")
plt.plot(x_test, y_pred, color="red")
plt.show()

print("Maximo Error:", metrics.max_error(y_test, y_pred))
print("Mean Absolute Error:", metrics.mean_absolute_error(y_test, y_pred))
print("Mean Squared Error:", metrics.mean_squared_error(y_test, y_pred))
print("R2:", metrics.r2_score(y_test, y_pred))

## Modelo con statsmodels
x_train = sm.add_constant(x_train)
modelo = sm.OLS(y_train, x_train)
modelo = modelo.fit()
# print(modelo.summary())
