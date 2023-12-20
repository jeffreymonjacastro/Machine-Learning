import os
os.chdir("./SESION 6")
os.getcwd()

import pandas as pd
import numpy as np

data = pd.read_csv("cinema.csv",
                   dtype={"title_year": str})

# object: cadena
# float64: decimal
# int64: entero

print(data.info())

