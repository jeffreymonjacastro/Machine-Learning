## Para establecer un directorio de trabajo
# import os
# os.chdir(".")
# os.getcwd()

import pandas as pd
import numpy as np

data = pd.read_csv("cinema.csv",
                   dtype={"title_year": str})

# object: cadena
# float64: decimal
# int64: entero

# print(data.info())

# Eliminar filas con al menos una celda vacias
# data2 = data.dropna()
# print(data2.info())

# cambiar filas vacias por 2023
data3 = data[:]
data3["title_year"] = data3["title_year"].fillna(value=2023)
# print(data3["title_year"])

#completar celda vacía por el valor siguiente
data4 = data[:]
data4["title_year"] = data4["title_year"].fillna(method="bfill")
# print(data4["title_year"])

# Cambiar los años mayor a 2012 por 2023
# primero cambiar la columna a numérica
data4["title_year"] = data4["title_year"].astype(int)
data4.loc[data4["title_year"] > 2012, "title_year"] = 2023
# print(data4["title_year"])

# Eliminar las filas que tenga celdas vacías en año
data = data.dropna(subset={"title_year"})
# print(data.info())

# Rellenar los datos nulos por la media
data["duration"] = data["duration"].fillna(data.duration.mean())
# print(data.duration[190:220])

# Cambiar los vacíos por una cadena
# Tail(n) -> últimos n registros
data["content_rating"] = data["content_rating"].fillna("No se")
# print(data["content_rating"].tail(15))

# Cambiar nombre de las columna
data = data.rename(columns={
    "director_name": "director",
    "movie_title": "TituloPelicula"
})
# print(data.info())

# Cambiar valores a mayúsculas
data["TituloPelicula"] = data["TituloPelicula"].str.upper()
# print(data["TituloPelicula"])

## Leer otra data
import datetime as dt
df = pd.read_csv("ride_new.csv")
# print(df.info())

## Convertir a category
# describe() -> ver detalladamente los promedios de una columna
df["user_type"] = df["user_type"].astype("category")
# print(df["user_type"].describe())

## Eliminar un substring con strip()
df["duration"] = df["duration"].str.strip(" minutes")
df["duration"] = df["duration"].astype(int)
# print(df["duration"].describe())

## Cambiar con condición
df.loc[df["tire_sizes"] == 27, "tire_sizes"] = 29
df["tire_sizes"] = df["tire_sizes"].astype("category")

## Cambiar una columna a fecha
df["ride_date"] = pd.to_datetime(df["ride_date"])
# print(df["ride_date"])

## Crear nuevas columnas
df["año"] = df["ride_date"].dt.year
df["mes"] = df["ride_date"].dt.month
# print(df.info())

## Leer otra data
df = pd.read_csv("denuncias.csv")
# print(df.columns)

## Eliminar las columnas
df = df.drop(['id', 'idcarpeta', 'ao', 'mes', 'clasificaciondelito', 'lon', 'lat', 'geopoint'], axis=1)
# print(df.columns)

## Convertir a tipo fecha
df["fechahecho"] = pd.to_datetime(df["fechahecho"])

## Ver delitos registrados después de las 8 pm
# print(df[df["fechahecho"].dt.hour >= 20])

df["delito"] = df["delito"].str.capitalize()
df["categoria"] = df["categoria"].str.capitalize()
df["tipopersona"] = df["tipopersona"].str.capitalize()
df["calidadjuridica"] = df["calidadjuridica"].str.capitalize()

# print(df.count())

## value_counts() es como un group_by, cuenta la catidad de registros que se repiten
df["sexo"] = df["sexo"].fillna("No se especifica")
# print(df["sexo"].value_counts(dropna=False))

## Obtener datos mediante filtro de fecha
df = df[df["fechahecho"].dt.year >= 2016]
# print(df["fechahecho"].dt.year.value_counts())

## Obtener data mediante filtro de edad
df = df.loc[df["edad"] <= 100]
# print(df["edad"].value_counts())

## Reemplazar los datos de una columna
# print(df["calidadjuridica"].value_counts())

reemplazo = {
    "Fallecido": "Cadaver",
    "Menor víctima": "Victima",
    "Victima  niño": "Victima",
    "Lesionado  adolescente": "Lesionado",
    "Victima  adolescente": "Victima"
}

df["calidadjuridica"] = df["calidadjuridica"].replace(reemplazo)
# print(df["calidadjuridica"].value_counts())


## Leer otra data
data = pd.read_csv("dataset_banco.csv")
# print(data.info())

data = data.dropna()
# print(data.info())

data = data.drop_duplicates()
# print(data.info())

data = data[data["age"] <= 100]
# print(data.info())

data = data[data["duration"] > 0]
# print(data.info())

data["job"] = data["job"].str.lower()
data["job"] = data["job"].str.replace("admin.", "administrative",
                          regex=False)
# print(data["job"].value_counts())

data["marital"] = data["marital"].str.lower()
data["marital"] = data["marital"].str.replace("div.","divorced",
                                              regex=False)
# print(data["marital"].value_counts())

data["education"] = data["education"].str.lower()
data[data["education"] == "sec."] = "secondary"
data[data["education"] == "unk"] = "unknown"
# print(data["education"].value_counts())