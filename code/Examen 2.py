# Examen 2 de Análisis de datos atmosféricos.
# CCA - UNAM - maestría.
# 28 de septiembre de 2021.

import os


import pandas as pd
import numpy as np

from scipy import stats

from matplotlib import pyplot as plt


path_r = os.getcwd() + "/resultados/Examen 2/"
path_d = os.getcwd() + "/datos/"

# Si no existe la carpeta, la crea.
if not os.path.exists(path_r):
    os.mkdir(path_r)


# Ejercicio 1
print( "Ejercicio 1" )
print( f"{stats.binom.pmf( 2, 18, 0.1 ):.4f}" )
print()


# Ejercicio 2
print( "Ejercicio 2" )
print( f"{stats.uniform.sf( ( 8 - 0 ) / ( 20 - 0 ) ):.4f}" )
print()


# Ejercicio 3
print( "Ejercicio 3" )
print( f"a. {stats.poisson.pmf( 2, 2.3 ):.4f}" )
print( f"b. {stats.poisson.pmf( 10, 2.3 * 5 ):.4f}" )
print( f"c. {stats.poisson.sf( 0, 2.3 * 2 ):.4f}" )
print()


# Ejercicio 4
print( "Ejercicio 4" )
print( f"{stats.expon.ppf( 0.9, scale = 140 / np.log(2) ):.2f}" )
print()


# Ejercicio 5

mu = 65
sigma = 8

print( "Ejercicio 5" )
print( f"a. {stats.norm.sf( 61, mu, sigma ):.4f}" )
a = ( stats.norm.cdf( 69, mu, sigma )
    - stats.norm.cdf( 63, mu, sigma ) )
print( f"b. {a:.4f}" )
print( f"c. {stats.norm.cdf( 70, mu, sigma ):.4f}" )
print( f"d. {stats.norm.sf( 75, mu, sigma ):.4f}" )
print()

# Gráfica inciso a.
fig, ax = plt.subplots()
x1 = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 1000)
y1 = stats.norm.pdf(x1, mu, sigma)
x2 = np.linspace(61, mu + 3 * sigma, 1000)
y2 = stats.norm.pdf(x2, mu, sigma)
ax.plot(x1, y1)
ax.fill_between(x2, y2)
ax.set_title("P{X > 61}",
    fontsize = 16)
ax.set_xlabel("Peso [kg]")
ax.set_ylabel("P")
ax.set_xlim(mu - 3 * sigma, mu + 3 * sigma)
ax.set_ylim(0)
plt.savefig(path_r + "Ejercicio_5_a.png")
plt.close()

# Gráfica inciso b.
fig, ax = plt.subplots()
x1 = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 1000)
y1 = stats.norm.pdf(x1, mu, sigma)
x2 = np.linspace(63, 69, 1000)
y2 = stats.norm.pdf(x2, mu, sigma)
ax.plot(x1, y1)
ax.fill_between(x2, y2)
ax.set_title("P{63 < X < 69}",
    fontsize = 16)
ax.set_xlabel("Peso [kg]")
ax.set_ylabel("P")
ax.set_xlim(mu - 3 * sigma, mu + 3 * sigma)
ax.set_ylim(0)
plt.savefig(path_r + "Ejercicio_5_b.png")
plt.close()

# Gráfica inciso c.
fig, ax = plt.subplots()
x1 = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 1000)
y1 = stats.norm.pdf(x1, mu, sigma)
x2 = np.linspace(mu - 3 * sigma, 70, 1000)
y2 = stats.norm.pdf(x2, mu, sigma)
ax.plot(x1, y1)
ax.fill_between(x2, y2)
ax.set_title("P{X < 70}",
    fontsize = 16)
ax.set_xlabel("Peso [kg]")
ax.set_ylabel("P")
ax.set_xlim(mu - 3 * sigma, mu + 3 * sigma)
ax.set_ylim(0)
plt.savefig(path_r + "Ejercicio_5_c.png")
plt.close()

# Gráfica inciso d.
fig, ax = plt.subplots()
x1 = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 1000)
y1 = stats.norm.pdf(x1, mu, sigma)
x2 = np.linspace(75, mu + 3 * sigma, 1000)
y2 = stats.norm.pdf(x2, mu, sigma)
ax.plot(x1, y1)
ax.fill_between(x2, y2)
ax.set_title("P{X > 75}",
    fontsize = 16)
ax.set_xlabel("Peso [kg]")
ax.set_ylabel("P")
ax.set_xlim(mu - 3 * sigma, mu + 3 * sigma)
ax.set_ylim(0)
plt.savefig(path_r + "Ejercicio_5_d.png")
plt.close()


# Ejercicio 6
print( "Ejercicio 6" )
print( f"a. {stats.binom.pmf( 0, 50, .02 ):.4f}" )
print( f"b. {stats.binom.pmf( 1, 50, .02 ):.4f}" )
print( f"{stats.binom.pmf( 2, 50, .02 ):.4f}" )
print( f"c. {stats.binom.sf( 2, 50, .02 ):.4f}" )
print( f"d. {50 * 0.02:.0f}" )
print()


# Ejercicio 7
print( "Ejercicio 7" )
a = stats.expon.sf( 21, 20, 0.5 )
print( f"a. {a:.4f}" )
b = stats.binom.pmf( 0, 15, a )
c = stats.binom.pmf( 1, 15, a )
d = stats.binom.pmf( 2, 15, a )
e = b + c + d
print( f"b. {b:.4f} + {c:.4f} "
    f" + {d:.4f} = {e:.4f}" )
print()


# Ejercicio 4.3
print( "Ejercicio 4.3" )
print( f"b. {stats.poisson.sf( 0, 1 / 18 ):.4f}" )
print( f"c. {stats.poisson.sf( 0, 13 / 23 ):.4f}" )
print()


# Ejercicio 4.7

fname = "A.3_Wilks.csv"

df = pd.read_csv(path_d + fname, index_col = "Year")

# Ajuste de distribución.
mu, sigma = stats.norm.fit(df["Temperature"])

print("Ejercicio 4.7")
print("a.")
print(f"mu: {mu:.2f} °C")
print(f"sigma: {sigma:.2f} °C")
print(f"max  : {df['Temperature'].min():.2f}")
print(f"min  : {df['Temperature'].max():.2f}")
print("b.")
print(f"mu: {mu * 9 / 5 + 32:.2f} °F")
print(f"sigma: {sigma * 9 / 5:.2f} °F")
print()

# Gráfica de histograma y distribución.
fig = plt.figure()

min = 23
max = 27
delta = 0.5
ax = df["Temperature"].hist(
    bins = np.arange(min, max + delta, delta),
    density = True )

x = np.linspace( min,
    max, 1000 )
y = stats.norm.pdf(x, mu, sigma)

ax.plot(x, y)

ax.set_title("Temperatura durante junio en Guayaquil",
    fontsize = 16)
ax.legend(["Distribución", "Muestra"])
ax.set_xlabel("Temperatura [°C]")
ax.set_ylabel("P")
ax.set_xlim( min, max)
ax.set_ylim(0)
plt.savefig(path_r + "Ejercicio_4.7_c.png")
plt.close()


# Ejercicio 4.10

fname = "Table 4.8.csv"

df = pd.read_csv(path_d + fname, index_col = "Year")

# Ajuste de distribución.
alpha, loc, beta = stats.gamma.fit(
    df["Precipitation"], floc = 0 )

print( "Ejercicio 4.10" )
print("a.")
print(f"alpha: {alpha:.2f}")
print(f"loc  : {loc:.2f}")
print(f"beta : {beta:.2f} in")
print(f"max  : {df['Precipitation'].min():.2f}")
print(f"min  : {df['Precipitation'].max():.2f}")
print("b.")
print(f"alpha: {alpha:.2f}")
print(f"beta : {beta * 25.4:.2f} mm")
print()

# Gráfica de histograma y distribución.
fig = plt.figure()

min = 0.5
max = 8.5 
delta = 1
ax = df["Precipitation"].hist(
    bins = np.arange(min, max + delta, delta),
    density = True )

x = np.linspace( 0,
    max, 1000 )
y = stats.gamma.pdf(x, alpha, loc, beta)

ax.plot(x, y)

ax.set_title("Precipitación durante julio en Ithaca",
    fontsize = 16)
ax.legend(["Distribución", "Muestra"])
ax.set_xlabel("Precipitación [in]")
ax.set_ylabel("P")
ax.set_xlim( 0, max)
ax.set_ylim(0)
plt.savefig(path_r + "Ejercicio_4.10_c.png")
plt.close()


# Ejercicio 4.11
print( "Ejercicio 4.11" )
print("a.")
print(f"p_30: {stats.gamma.ppf(0.3, alpha, loc, beta):.2f}")
print(f"p_70: {stats.gamma.ppf(0.7, alpha, loc, beta):.2f}")

print("b.")
median = stats.gamma.ppf(0.7, alpha, loc, beta)
mean_s = df["Precipitation"].mean()
print(f"median       :  {median:.2f}")
print(f"sample mean  :  {mean_s:.2f}")
print(f"mean - median: {mean_s - median:.2f}")

print("c.")
print(f"{stats.gamma.sf(7, alpha, loc, beta):.2f}")
print()


# Ejercicio 4.16

fname = "A.1_Wilks.csv"

temp = ["Canandaigua - Min Temp", "Canandaigua - Max Temp"]

df = pd.read_csv(path_d + fname, index_col = "Date")

# Normal bivariada.
# Se obtienen los parámetros.
mu_x = df[temp[0]].mean()
mu_y = df[temp[1]].mean()
sigma_x = df[temp[0]].std()
sigma_y = df[temp[1]].std()
rho = df[temp].corr()
cov = df[temp].cov()

print("Ejercicio 4.16")
print("a.")
print("mu_x    = " f"{mu_x:.1f}")
print("mu_y    = " f"{mu_y:.1f}")
print("sigma_x = " f"{sigma_x:.2f}")
print("sigma_y = " f"{sigma_y:.2f}")
print("rho     = " f"{rho.iat[1, 0]:.2f}")
print("cov     = " f"{cov.iat[1, 0]:.1f}")

# Distribución condicional.
x = 0
y = 20

# Parámetros condicionales.
mu_y_x = ( mu_y +( rho.iat[1, 0] * sigma_y *
    ( x - mu_x ) ) / sigma_x )
sigma_y_x = sigma_y * np.sqrt(
    1 - rho.iat[1, 0] ** 2 )

print("b.")
print("mu_y_x = " f"{mu_y_x:.2f}")
print("sigma_y_x = " f"{sigma_y_x:.2f}")
p_cond = stats.norm.cdf(y, mu_y_x, sigma_y_x)
print(f"{p_cond:.4f}")
print()


# Ejercicio 4.19
print( "Ejercicio 4.19" )
a = stats.weibull_min.cdf( 10, 1.2, scale = 7.4 )
b = stats.weibull_min.cdf( 20, 1.2, scale = 7.4 )
print( f"{b:.4f} - {a:.4f} = {b - a:.4f}" )
print()
