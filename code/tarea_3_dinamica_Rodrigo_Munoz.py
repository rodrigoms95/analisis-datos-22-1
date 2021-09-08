# Calcula la temperatura de una parcela en
# diferentes niveles de presión.

import os


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt


path_r = os.getcwd() + "/resultados/dinamica/"
nombre = "Rodrigo_Munoz"

# Si no existe la carpeta, la crea.
if not os.path.exists(path_r):
    os.mkdir(path_r)


# Valores para el aire.
K = 273.15
R = 287.05
c_p = 1004

# Se crea la tabla con los rangos de
# presión y la temperatura inicial.
df = pd.DataFrame({"P": np.arange(100, 1010, 10)[::-1],
    "T": [20 + K] + [np.nan] * 90})

# Se calcula la temperatura.
df.loc[1:, "T"] = ( df.at[0, "T"] *
    ( df.loc[1:, "P"] / df.at[0, "P"]  ) ** (R / c_p) )

df["T"] -= K

df.to_csv(path_r + "tabla_" + nombre + ".csv", index = False)


# Se grafica.
ax = df.plot("T", "P")

# Ajustes de la gráfica.
ax.invert_yaxis()
ax.set_title("Variación de temperatura de una parcela\n"
    + "a diferentes niveles de presión", fontsize = 16)
ax.grid(axis = "y")
ax.grid(axis = "x", which = "both")
ax.legend([])
ax.set_xlabel("Temperatura [°C]")
ax.set_ylabel("Presión [hPa]")
ax.set_xlim(-130, 20)
ax.set_ylim(1000, 100)
ax.xaxis.set_major_locator(
    plt.MultipleLocator(20))
ax.xaxis.set_minor_locator(
    plt.MultipleLocator(10))

plt.savefig(path_r + "gráfica_" + nombre + ".png")
