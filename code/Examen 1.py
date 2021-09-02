# Examen 1 de Análisis de datos atmosféricos.
# CCA - UNAM - maestría.
# 2 de septiembre de 2021.

import os


import pandas as pd
import numpy as np

from scipy import stats

from matplotlib import pyplot as plt


# Datos para los ejercicios 6, 7, 8.

path_d = os.getcwd() + "/datos/"
path_r = os.getcwd() + "/resultados/Examen 1/"
fname = "A.3_Wilks.csv"

# Si no existe la carpeta, la crea.
if not os.path.exists(path_r):
    os.mkdir(path_r)

df = pd.read_csv(path_d + fname, index_col = "Year")


# Ejercicio 6.
# Anomalía estandarizada.

df["z Temp"] = stats.zscore(df["Temperature"])

ax = df["z Temp"].plot(legend = True)
( df["z Temp"].where(df["El Niño"] == True)
    .plot(linestyle = "", marker = "o", 
    color = "blue", legend = True, ax = ax) )

#plt.title("Anomalía Estandarizada de la Temperatura",
#    fontsize = 16)
plt.xlabel("Año")
plt.ylabel("Anomalía estandarizada")
plt.legend(["_", "Años Niño"])
plt.grid(axis = "y")

ax.xaxis.set_major_formatter("{x:.0f}")

plt.savefig(path_r + "Ejercicio_6.png")
plt.close()


# Ejercicio 7.
# Dispersión.

ax = ( df.where(df["El Niño"] == True).plot
    .scatter("Temperature", "Pressure", c = "blue") )
( df.where(df["El Niño"] == False).plot
    .scatter("Temperature", "Pressure", c = "red", ax = ax) )

#plt.title("Dispersión Temperatura - Presión",
#    fontsize = 16)
plt.xlabel("Temperatura")
plt.ylabel("Presión")
plt.legend(["Años Niño", "_"])

plt.savefig(path_r + "Ejercicio_7.png")
plt.close()

# Ejercicio 8.
# Correlaciones. 

corr = [df.loc[:, ["Temperature","Pressure"]].corr().iat[1, 0],
    df.loc[:, ["Temperature","Pressure"]].corr("spearman").iat[1, 0]]

tipo = ["Pearson:  ", "Spearman: "]

with open(path_r + "examen 1.txt", "w", encoding = "utf-8") as f:
    for i in range(len(tipo)):
        f.write("Correlación de " + tipo[i]
            + f"{corr[i]:.3f}\n")
    f.write("\n")



# Datos ejercicio 9.

fname = "SN_m_tot_V2.0.csv"

sunspot = "Monthly Sunspots"
cols = ["Year", "Month", "Year Fraction", 
    sunspot, "Monthly Mean Std",
    "Number of Observations",
    "Definitive/provisional marker"]

sol = pd.read_csv(path_d + fname, sep = ";",
    names = cols, index_col = cols[0])

# Se convierte el valor -1 a np.nan
sol = sol.where(sol > -1, np.nan)


# Ejercicio 9.a.
# Histograma.

k = np.ceil(1 + 3.3 *
    np.log10(sol.shape[0]) )

fig, ax = plt.subplots()
n, bins, patches = ax.hist(sol[sunspot],
    bins = k.astype(int))

#plt.title("Manchas solares mensuales - Histograma",
#    fontsize = 16)
plt.xlabel("Número de manchas")
plt.ylabel("Frecuencia")
plt.grid(axis = "y", which = "both")

ax.xaxis.set_major_locator(plt.MultipleLocator(50))
ax.xaxis.set_minor_locator(plt.MultipleLocator(25))
ax.yaxis.set_major_locator(plt.MultipleLocator(200))
ax.yaxis.set_minor_locator(plt.MultipleLocator(50))

with open(path_r + "examen 1.txt", "a", encoding = "utf-8") as f:
    f.write("Ejercicio 9.a.\n")
    f.write("k recomendado: " + k.astype(str) + "\n")
    f.write( "ancho de intervalo: " + 
        f"{bins[1]:.2f}\n")
    f.write("\n")

plt.savefig(path_r + "Ejercicio_9_a.png")
plt.close()


# Ejercicio 9.b.
# Estadísticos.

with open(path_r + "examen 1.txt", "a", encoding = "utf-8") as f:
    f.write("Ejercicio 9.a.\n")
    f.write("Media:   " + f"{sol[sunspot].mean():.2f}\n")
    f.write("Mediana: " + f"{sol[sunspot].median():.2f}\n")
    f.write("Moda:    " + f"{bins[1] / 2:.2f}\n")
    f.write("IQR:     " +
        f"{sol[sunspot].quantile(0.75) - sol[sunspot].quantile(0.25):.2f}\n")
    f.write("sesgo:   " +
        f"{sol[sunspot].skew():.2f}\n")
    f.write("\n") 


# Ejercicio 9.c.
# Box plot.

sol[sunspot].plot.box()

#plt.title("Manchas solares mensuales - Box plot",
#    fontsize = 16)
plt.xticks([])
plt.ylabel("Número de manchas")

plt.savefig(path_r + "Ejercicio_9_c.png")
plt.close()


# Ejercicio 9.d.
# Frecuencia acumulada.

fig, ax = plt.subplots()
# Criterio de Weibull.
ax.plot(sol[sunspot].sort_values(),
    range(1, sol.shape[0] + 1) /
    np.float64(sol.shape[0] + 1)
    )

#plt.title("Manchas solares - Frecuencia acumulada",
#    fontsize = 16)
plt.grid(True, which = "both")
plt.xlabel("Número de manchas")
plt.ylabel("Frecuencia acumulada")

ax.xaxis.set_major_locator(plt.MultipleLocator(50))
ax.xaxis.set_minor_locator(plt.MultipleLocator(10))
ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))
ax.yaxis.set_minor_locator(plt.MultipleLocator(0.05))

plt.savefig(path_r + "Ejercicio_9_d.png")
plt.close()


# Ejercicio 9.f.
# Media móvil.

sol["rolling mean"] = sol[sunspot].rolling(13).mean()

sol[[sunspot, "rolling mean"]].plot()

#plt.title("Manchas solares",
#    fontsize = 16)
plt.xlabel("Año")
plt.ylabel("Número de manchas")
plt.grid(axis = "y")
plt.legend(["Promedio mensual", "Media móvil a 13 meses"])

plt.savefig(path_r + "Ejercicio_9_f.png")
plt.close()

# Ejercicio 9.g.
# Autocorrelación.

sol_anual = sol[sunspot].groupby("Year").mean().to_frame()
sol_anual.rename(columns = {"Monthly Sunspots": "Sunspots"},
    inplace = True)

pd.plotting.autocorrelation_plot(
        sol_anual, marker = "o").set_xlim([1, 40])

#plt.title("Manchas solares anuales - Autocorrelación",
#    fontsize = 16)
plt.xlabel("Retraso")
plt.ylabel("Autocorrelación")
plt.grid(axis = "x")

plt.savefig(path_r + "Ejercicio_9_g.png")
plt.close()
