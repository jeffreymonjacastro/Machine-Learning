# import os
# os.chdir("")
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

import datetime as dt
df = pd.read_csv("ride_new.csv")
# print(df.info())

# Convertir a category
# describe() -> ver detalladamente los promedios de una columna
df["user_type"] = df["user_type"].astype("category")
# print(df["user_type"].describe())

# Eliminar un substring con strip()
df["duration"] = df["duration"].str.strip(" minutes")
df["duration"] = df["duration"].astype(int)
# print(df["duration"].describe())

# 
df.loc[df["tire_sizes"] == 27, "tire_sizes"] = 29
df["tire_sizes"] = df["tire_sizes"].astype("category")
