# Examen 3 de Análisis de datos atmosféricos.
# ICACC - UNAM - maestría.
# 2 de noviembre de 2021.

import os

import pandas as pd
import numpy as np

from scipy import stats

from matplotlib import pyplot as plt

from statsmodels.stats.diagnostic import lilliefors
from statsmodels.stats.stattools import durbin_watson

path_r = os.getcwd() + "/resultados/Examen 3/"

# Si no existe la carpeta, la crea.
if not os.path.exists(path_r):
    os.mkdir(path_r)


# Ejercicio 1
# Prueba de hipótesis para la resta
# de medias bajo independencia.

# Datos.
path_d = os.getcwd() + "/datos/"
fname = "A.3_Wilks.csv"
df = pd.read_csv(path_d + fname, index_col = "Year")

# Separamos los conjuntos.
df_Nino = ( df.where( df["El Niño"] )
    .dropna() )["Temperature"]
df_Nina = ( df.where( ~df["El Niño"] )
    .dropna() )["Temperature"]

# Clave para distinguir los datos
o = df_Nino
a = df_Nina

# Estadísticos.
m_o = o.mean()
s_o = o.std()
n_o = o.shape[0]
v_o = s_o ** 2 / n_o
m_a = a.mean()
s_a = a.std()
n_a = a.shape[0]
v_a = s_a ** 2 / n_a
m_d = np.abs( m_o - m_a )
v_d = v_o + v_a
s_d = np.sqrt( v_d )

# Estadístico de prueba.
z = m_d / np.sqrt( v_d )

# Valor p.
p = stats.norm.cdf(z)

# Intervalo de confianza.
alpha = 0.5
crit = stats.norm.ppf( 1- alpha / 2 )
error = crit * s_d
conf = [ m_d - error, m_d + error ]

print("Ejercicio 1")
print("o -> Niño")
print("a -> Niña")
print("d -> Delta, o - a")
print(f"mu_o:    {m_o:.3f}")
print(f"sigma_o:  {s_o:.3f}")
print(f"var_o:    {v_o:.3f}")
print(f"n_o:      {n_o}")
print(f"mu_a:    {m_a:.3f}")
print(f"sigma_a:  {s_a:.3f}")
print(f"var_a:    {v_a:.3f}")
print(f"n_a:      {n_a}")
print(f"mu_d:     {m_d:.3f}")
print(f"sigma_d:  {s_d:.3f}")
print(f"var_d:    {v_d:.3f}")
print(f"z:        {z:.3f}")
print("1.a: prueba de hipótesis.")
print("Prueba bilateral")
print(f"p:     {p:.6f}")
print("1.b: Intervalo de confianza.")
print(f"alpha:    {alpha:.3f}")
print(f"z_crit:  {crit:.3f}")
print(f"error:   {error:.3f}")
print(f"confianza: [ {conf[0]:.3f}"
    f" , {conf[1]:.3f} ]")


# Ejercicio 2
# Prueba de hipótesis para la resta
# de medias con correlación serial.

# Datos.
path = os.getcwd() + "/datos/A.1_Wilks.csv"
df = pd.read_csv(path)[["Ithaca - Min Temp",
    "Canandaigua - Min Temp"]]

# Delta de temperatura.
# Positivo: Canandaigua
#  es mayor que Ithaca.
df["delta"] = ( df["Canandaigua - Min Temp"]
    - df["Ithaca - Min Temp"] )
var = "delta"

# Estadísticos de Delta.
mu = df[var].mean()
sigma = df[var].std()
rho1 = df[var].autocorr()
n = df.shape[0]

# Ajuste por correlación serial.
n_prim = n * ( 1 - rho1 ) / ( 1 + rho1 )

# Varianza con ajuste.
var = sigma ** 2 / n_prim

# Estadístico de prueba.
z = mu / np.sqrt(var)

# Valor p.
p_bi = stats.norm.cdf(z)
p_uni = 1 - ( 1 - p_bi ) / 2 

print("Ejercicio 2")
print(f"mu:      {mu:.3f}")
print(f"sigma:   {sigma:.3f}")
print(f"rho1:    {rho1:.3f}")
print(f"n:      {n:.3f}")
print(f"n_prim: {n_prim:.3f}")
print(f"var:     {var:.3f}")
print(f"z:       {z:.3f}")
print("2.a: prueba bilateral.")
print(f"p:    {p_bi:.6f}")
print("2.b: prueba unilateral.")
print(f"p:    {p_uni:.6f}")


# Ejercicio 3

# Datos
path_d = os.getcwd() + "/datos/"
fname = "Table 4.8.csv"
df = pd.read_csv(path_d + fname,
    index_col = "Year")
var = "Precipitation"

# Ajuste de parámetros.
mu, sigma = stats.norm.fit( df[var] )
# Parámetros estimados.
params = 2

# 3.a: Chi-square
# Histograma de datos observados.
bins_lim = [ 0, 1.75, 3, 4, 5.5,
    df[var].max() ]
n_obs, bins = np.histogram( 
    df[var], bins = bins_lim )

# Se discretizan las distribuciones continuas.
prob_norm = np.array( [
    stats.norm.cdf(bins_lim[1], mu, sigma),
    stats.norm.cdf(bins_lim[2], mu, sigma) -
    stats.norm.cdf(bins_lim[1], mu, sigma),
    stats.norm.cdf(bins_lim[3], mu, sigma) -
    stats.norm.cdf(bins_lim[2], mu, sigma),
    stats.norm.cdf(bins_lim[4], mu, sigma) -
    stats.norm.cdf(bins_lim[3], mu, sigma),
    stats.norm.sf(bins_lim[4], mu, sigma) 
    ] )
n_norm = n_obs.sum() * prob_norm

# Graficamos los datos y las distribuciones.
fig, ax = plt.subplots()

df[var].hist( bins = bins_lim,
    density = True, ax = ax )

x = np.linspace(0, df[var].max(), 1000)
y = stats.norm.pdf(x, mu, sigma)

ax.plot(x, y)

ax.set_title("Prueba Chi-square",
    fontsize = 16)
ax.set_xlabel("Precipitación [mm]")
ax.set_ylabel("P")
ax.legend(["Normal", "Histograma"])
ax.set_xlim(0, bins_lim[-1])
ax.set_ylim(0)

fig.savefig(path_r + "Ejercicio_3_a.png")

# Prueba chi-square.
alpha_test = 0.05
clases = bins.shape[0] - 1
nu = clases - params - 1
crit = stats.chi.ppf(1 - alpha_test, nu)

AUTO = False
#AUTO = True
if AUTO:
    # Automática.
    chi_test = stats.chisquare(
        n_obs, n_norm, ddof = params)
    chi = chi_test.statistic
    p = chi_test.pvalue
else:
    # Manual.
    chi = ( ( n_obs - n_norm ) ** 2
        / n_norm ).sum()
    p = stats.chi.sf(chi, nu)

print("Chi-square")
print(f"Chi : {chi:.2f}")
print(f"p   : {p:.4f}")
print(f"crit: {crit:.4f}")

# 3.b. Lilliefors
# Prueba Kolmogorov-Smirnov.

# Tamaño de la muestra.
n = df[var].shape[0]

# Frecuencia acumulada.
# Criterio de Weibull.
F = ( range( 1, n + 1 ) /
    np.float64( n + 1 ) )

# Frecuencia acumulada de
# distribución normal.
x = df[var].sort_values()
F_n = stats.norm.cdf(
    x, mu, sigma )

# Lilliefors critical value.
crit = 0.161
if AUTO:
    # Automático.
    # Smirnov test statistic.
    D_s, p = lilliefors( df[var] )
else:
    # Manual.
    # Smirnov test statistic.
    D = np.abs( F - F_n )
    D_s = D.max()
    alpha = 0.05

print("")
print("Lilliefors")
print(f"n   : {n}")
print(f"D_s :  {D_s:.3f}")
if AUTO:
    print(f"p   :  {p:.3f}")
print(f"crit:  {crit:.3f}")

# Graficamos las distribuciones.
x_n = np.linspace( df[var].min(),
    df[var].max(), 1000 )
y_n = stats.norm.cdf( x_n, mu, sigma )

fig, ax = plt.subplots()
ax.plot(x_n, y_n)

# Grficamos la frecuencia acumulada.
ax.plot(df[var].sort_values(),
    F, drawstyle = "steps")

# Graficamos los intervalos de confianza.
# Distribución Kolmogorov-Smirnov
ax.plot( x, F - crit,
    drawstyle = "steps", color = "red" )
ax.plot( x, F + crit,
    drawstyle = "steps", color = "red" )

ax.set_title("Distibución Normal \n"
    "Prueba Kolmogorov-Smirnov",
    fontsize = 16)
ax.set_xlabel("Precipitación [in]")
ax.set_ylabel("P")
ax.legend(["Normal",
    "Datos", "Intervalo de\nconfianza"])
ax.set_xlim( df[var].min(), df[var].max() )
ax.set_ylim(0, 1)

fig.savefig(path_r + "Ejercicio_3_b.png")


# Ejercicio 4
# Prueba Wilcoxon-Mann-Whitney.

ej = "4. Prueba Wilcoxon-Mann-Wthiney"

# Datos.
path_d = os.getcwd() + "/datos/"
fname = "A.3_Wilks.csv"
df = pd.read_csv(path_d + fname, index_col = "Year")

# Ordenamos los valores.

df.sort_values("Pressure", inplace = True)
# Creamos la columan de rangos y un
# índice ascendiente para verificar
# que el rango esté bien calculado. 
df.reset_index(inplace = True)
df.index += 1
df.reset_index(inplace = True)
df.rename(columns = {"index": "rank"},
    inplace = True)
df.index += 1

# Revisamos los valores repetidos.
df["Repeated"] = df.duplicated("Pressure")
for i in df.itertuples():
    if i.Repeated == True:
        # Hay tres valores repetidos.
        if df.loc[i[0] - 1,
            "Repeated"] == True:
            # Promedio de los rangos.
            for j in range(i[0] - 2,  i[0] + 1):
                df.loc[j, "rank"] = i.rank - 1
        # Hay dos valore1☼ repetidos.
        else:
            # Promedio de los rangos.
            for j in range(i[0] - 1,  i[0] + 1):
                df.loc[j, "rank"] = (
                    2 * i.rank - 1 ) / 2

# Separamos los conjuntos.
df_Nino = ( df.where( df["El Niño"] )
    .dropna() )
df_Nina = ( df.where( ~df["El Niño"] )
    .dropna() )

# Clave para distinguir los datos
o = df_Nino
a = df_Nina

# Tamaños de las muestras.
n_1 = o.shape[0]
n_2 = a.shape[0]

# Suma de los rangos.
R_1 = o["rank"].sum()
R_2 = a["rank"].sum()

# Estadístico de prueba.
U_1 = R_1 - n_1 / 2 * ( n_1 + 1 )
U_2 = R_2 - n_2 / 2 * ( n_2 + 1 )

# Parámetros de la distribución nula.
m_U = n_1 * n_2 / 2
s_U = np.sqrt( n_1 * n_2 *
    ( n_1 + n_2 + 1 ) / 12 )

# Estadístico de prueba normalizado.
z_U = ( U_1 - m_U ) / s_U

# Nivel de significancia.
alpha_test = 0.05

# Valor crítico.
z_crit = stats.norm.ppf(1- alpha_test / 2)

# Prueba automatizada.
MWU = stats.mannwhitneyu(
    o["Pressure"], a["Pressure"] )

print(ej)
print(f"n_1: {n_1}")
print(f"n_2: {n_2}")
print(f"R_1: {R_1:.1f}")
print(f"R_2: {R_2:.1f}")
print(f"U_1: {U_1:.2f}")
print(f"U_2: {U_2:.2f}")
print(f"m_U: {m_U:.2f}")
print(f"s_U: {s_U:.2f}")
print(f"z_U: {z_U:.3f}")
print(f"z_c: {-z_crit:.3f}")

print("\nAuto")
print(f"U: {MWU.statistic:.2f}")
print(f"p: {MWU.pvalue:.3f}")

df.drop(columns = ["Year", "Temperature",
    "Precipitation", "Repeated"])


# Ejercicio 5
# Prueba chi-square.

# Datos
path_d = os.getcwd() + "/datos/"
fname = "Datos_HurrDays_1900-1983.csv"
df = pd.read_csv(path_d + fname,
    sep = "\s+", header = 0,
    names = ["Year", "Hurr_days"],
    index_col = "Year")
v = "Hurr_days"

# Estadísticos
mean = df[v].mean()
std = df[v].std()

print("Estadísticos")
print( f"mean     : {mean:.2f}" )
print( f"std      : {std:.2f}" )

# Ajuste de parámetros.
# Distribución continua gamma.
alpha, zeta, beta = stats.gamma.fit(
    df[v], loc = 0 )
# Distribución discreta binomial negativa.
p = mean / std ** 2
k = mean ** 2 / (std ** 2 - mean)
# Parámetros estimados.
params = 2

print("")
print("Parámetros")
print("Gamma")
print(f"alpha: {alpha:.2f}")
print(f"beta : {beta:.2f}")
print("Binomial negativa")
print(f"p: {p:.2f}")
print(f"k: {k:.2f}")

# 3.a: Chi-square
# Histograma de datos observados.
bins_lim = [ 0, 7.5, 15.5, 23.5, 31.5,
    39.5, 47.5, df[v].max() ]
n_obs, bins = np.histogram( 
    df[v], bins = bins_lim )

# Se discretizan las distribuciones continuas.
prob_gamma = np.array( [
    stats.gamma.cdf(bins_lim[1], alpha, zeta, beta),
    stats.gamma.cdf(bins_lim[2], alpha, zeta, beta) -
    stats.gamma.cdf(bins_lim[1], alpha, zeta, beta),
    stats.gamma.cdf(bins_lim[3], alpha, zeta, beta) -
    stats.gamma.cdf(bins_lim[2], alpha, zeta, beta),
    stats.gamma.cdf(bins_lim[4], alpha, zeta, beta) -
    stats.gamma.cdf(bins_lim[3], alpha, zeta, beta),
    stats.gamma.cdf(bins_lim[5], alpha, zeta, beta) -
    stats.gamma.cdf(bins_lim[4], alpha, zeta, beta),
    stats.gamma.cdf(bins_lim[6], alpha, zeta, beta) -
    stats.gamma.cdf(bins_lim[5], alpha, zeta, beta),
    stats.gamma.sf( bins_lim[6], alpha, zeta, beta) 
    ] )
n_gamma = n_obs.sum() * prob_gamma

# Histograma de la distribución discreta
prob_nbinom = np.array( [
    stats.nbinom.cdf(bins_lim[1], k, p),
    stats.nbinom.cdf(bins_lim[2], k, p) -
    stats.nbinom.cdf(bins_lim[1], k, p),
    stats.nbinom.cdf(bins_lim[3], k, p) -
    stats.nbinom.cdf(bins_lim[2], k, p),
    stats.nbinom.cdf(bins_lim[4], k, p) -
    stats.nbinom.cdf(bins_lim[3], k, p),
    stats.nbinom.cdf(bins_lim[5], k, p) -
    stats.nbinom.cdf(bins_lim[4], k, p),
    stats.nbinom.cdf(bins_lim[6], k, p) -
    stats.nbinom.cdf(bins_lim[5], k, p),
    stats.nbinom.sf( bins_lim[6], k, p) 
    ] )
n_nbinom = n_obs.sum() * prob_nbinom

# Graficamos los datos y las distribuciones.
fig, ax = plt.subplots()

df[v].hist( bins = bins_lim,
    density = True,
    ax = ax )

x_gamma = np.linspace(0, bins_lim[-1], 1000)
y_gamma = stats.gamma.pdf(
    x_gamma, alpha, zeta, beta)

x_nbinom = np.arange(0,
    np.floor( df[v].max() ), 1)
y_nbinom = stats.nbinom.pmf(
    x_nbinom, k, p)

ax.plot(x_gamma, y_gamma)
ax.stem( x_nbinom, y_nbinom,
    linefmt = "C3-",
    markerfmt = "C3o" )

ax.set_title("Prueba Chi-square",
    fontsize = 16)
ax.set_xlabel("Precipitación [mm]")
ax.set_ylabel("P")
ax.legend(["Gamma", "Histograma",
 "Binomial\nnegativa"])
ax.set_xlim(0, bins_lim[-1])
ax.set_ylim(0)

# Prueba chi-square.
alpha_test = 0.05
clases = bins.shape[0]
nu = clases - params - 1
crit = stats.chi.ppf(1 - alpha_test, nu)

AUTO = False
#AUTO = True
if AUTO:
    # Automática.
    chi_test_gamma = stats.chisquare(
        n_obs, n_gamma, ddof = params)
    chi_gamma = chi_test_gamma.statistic
    p_gamma = chi_test_gamma.pvalue
    chi_test_nbinom = stats.chisquare(
        n_obs, n_nbinom, ddof = params)
    chi_nbinom = chi_test_nbinom.statistic
    p_nbinom = chi_test_nbinom.pvalue
else:
    # Manual.
    chi_gamma = ( ( n_obs - n_gamma ) ** 2
        / n_gamma ).sum()
    chi_nbinom = ( ( n_obs - n_nbinom ) ** 2
        / n_nbinom ).sum()
    p_gamma = stats.chi.sf(chi_gamma, nu)
    p_nbinom = stats.chi.sf(chi_nbinom, nu)

print("")
print("Chi-square")
print("Gamma")
print(f"Chi : {chi_gamma:.2f}")
print(f"p   : {p_gamma:.4f}")
print("Binomial Negativa")
print(f"Chi : {chi_nbinom:.2f}")
print(f"p   : {p_nbinom:.4f}")
print(f"crit: {crit:.4f}")

fig.savefig(path_r + "Ejercicio_5.png")


# Ejercicio 6

# Datos.
path  = os.getcwd() + "/datos/"
fname = "data_madison_precip.txt"
v   = "Precipitación"

# Se lee el archivo .dat
# y se ajusta su formato.
df = pd.read_table(path + fname,
    names = [v], sep = "\s+")
df.index.set_names(["Year", "Month", "Day"],
    inplace = True)

# Se e☼cogen los datos de
# junio mayores a 0.1 in.
df = df.xs(6, level = "Month")
df = df.where(df > 0.1).dropna()

# Estadísticos
mean = df[v].mean()
median = df[v].median()
mode = df[v].mode()[0]
std = df[v].std()
var = df[v].var()
skew = df[v].skew()
quartiles = [ df[v].quantile(0.25), 
    df[v].quantile(),
    df[v].quantile(0.75)]
IQR = ( df[v].quantile(0.75)
    - df[v].quantile(0.25) )
rango = df[v].max() - df[v].min()

print("Estadísticos")
print( f"mean     : {mean:.2f}" )
print( f"median   : {median:.2f}" )
print( f"mode     : {mode:.2f}" )
print( f"std      : {std:.2f}" )
print( f"var      : {var:.2f}" )
print( f"skewness : {skew:.2f}" )
print( f"quartiles: {quartiles[0]:.2f}, "
    f"{quartiles[1]:.2f}, {quartiles[2]:.2f}" )
print( f"IQR      : {IQR:.2f}" )
print( f"range    : {rango:.2f}" )

# Ajuste de parámetros.
alpha, zeta, beta = stats.gamma.fit(
    df[v])#, loc = 0)

print("")
print("Distribución Gamma")
print(f"alpha: {alpha:.3f}")
print(f"beta : {beta:.3f}")

# Graficamos los datos y las distribuciones.
fig, ax = plt.subplots()
n_obs, bins, patches = plt.hist( 
    df[v], density = True)

x = np.linspace(0, df[v].max(), 1000)
y = stats.gamma.pdf(x, alpha, zeta, beta)
ax.plot(x, y)

ax.set_title("Precipitación en Madison\n"
    "Valores de junio mayores a 0.1 in",
    fontsize = 16)
ax.set_xlabel("Precipitación [mm]")
ax.set_ylabel("P")
ax.legend(["Gamma", "Histograma"])
ax.set_ylim(0, 2)
ax.set_xlim(bins[0], bins[-1])

fig.savefig(path_r + "Ejercicio_6_hist.png")

fig, ax = plt.subplots()
df.boxplot(ax = ax)

ax.set_title("Precipitación en Madison\n"
    "Valores de junio mayores a 0.1 in",
    fontsize = 16)
ax.set_xlabel("Madison - junio")
ax.set_ylabel("Precipitación [mm]")

fig.savefig(path_r + "Ejercicio_6_boxplot.png")

# Prueba Kolmogorov-Smirnov - Lilliefors.

# Tamaño de la muestra.
n = df[v].shape[0]

# Frecuencia acumulada.
# Criterio de Weibull.
F = ( range( 1, n + 1 ) /
    np.float64( n + 1 ) )

# Frecuencia acumulada de
# distribución normal.
x = df[v].sort_values()
F_n = stats.gamma.cdf(
    alpha, zeta, beta )

# Lilliefors critical value.
crit = 1.05 / np.sqrt( n )

AUTO = True
#AUTO = False
if AUTO:
    # Automático.
    # Smirnov test statistic.
    D_s, p = lilliefors( df[v] )
else:
    # Manual.
    # Smirnov test statistic.
    D = np.abs( F - F_n )
    D_s = D.max()
    alpha_test = 0.05

print("")
print("Lilliefors")
print(f"n   : {n}")
print(f"D_s :   {D_s:.3f}")
print(f"crit:   {crit:.3f}")
if AUTO:
    print(f"p   :   {p:.3f}")

# Graficamos las distribuciones.
x_n = np.linspace( df[v].min(),
    df[v].max(), 1000 )
y_n = stats.gamma.cdf(
    x_n, alpha, zeta, beta )
fig, ax = plt.subplots()
ax.plot(x_n, y_n)

# Graficamos la frecuencia acumulada.
ax.plot(df[v].sort_values(), F,
    drawstyle = "steps", linewidth = 0.75)

# Graficamos los intervalos de confianza.
# Distribución Kolmogorov-Smirnov
ax.plot( x, F - crit, drawstyle = "steps",
    color = "red", linewidth = 0.75 )
ax.plot( x, F + crit, drawstyle = "steps",
    color = "red", linewidth = 0.75 )

ax.set_title("Distibución Normal \n"
    "Prueba Kolmogorov-Smirnov",
    fontsize = 16)
ax.set_xlabel("Precipitación [in]")
ax.set_ylabel("P")
ax.legend(["Normal",
    "Datos", "Intervalo de\nconfianza"])
ax.set_xlim( df[v].min(), df[v].max() )
ax.set_ylim(0, 1)

fig.savefig(path_r + "Ejercicio_Lilliefors.png")



# Regresión lineal

# Problema 1

ej = "1. Profundidad - Oxígeno"

x = np.array( [ 15, 20, 30,
    40, 50, 60, 70 ] )
y = np.array( [ 6.5, 5.6, 5.4,
    6.0, 4.6, 1.4, 0.1 ] )

# Cantidad de predictores.
k = 1

alpha_test = 0.05

# Regresión lineal.
lin_reg = stats.linregress(x, y)
b = lin_reg.intercept
a = lin_reg.slope

# Y gorro.
y_reg = b + a * x

# Residuales.
res = y - y_reg

# ANOVA
n = x.shape[0]
SST = ( ( y - y.mean() ) ** 2 ).sum()
SSR = ( ( y_reg - y.mean() ) ** 2 ).sum()
SSE = ( ( y - y_reg ) ** 2 ).sum()
Se = np.sqrt(SSE / (y.shape[0] - 2 ))
R = SSR / SST
F = SSR / Se ** 2
F_crit = stats.f.ppf( 1 - alpha_test, 1,
    n - 1 - k )
p = stats.f.cdf( F, 1, n - 1 - k )
t_crit = stats.t.ppf( 1 - alpha_test / 2,
    n - 1 - k)
t_sb = ( t_crit * Se / np.sqrt(
    ( ( x - x.mean() ) ** 2 ).sum()
    * ( n - 1 - k ) ) )
durbin = durbin_watson(res)

print(ej)
print(f"n            : {n}")
print(f"a            : {a:.4f}")
print(f"b            : {b:.4f}")
print(f"X_mean       : {x.mean():.4f}")
print(f"Y_mean       : {y.mean():.4f}")
print(f"SST          : {SST:.4f}")
print(f"SSR          : {SSR:.4f}")
print(f"SSE          : {SSE:.4f}")
print(f"RMSE         : {Se:.4f}")
print(f"R            : {R:.4f}")
print(f"F            : {F:.4f}")
print(f"F_crit       : {F_crit:.4f}")
print(f"p            : {p:.4f}")
print(f"t_crit       : {t_crit:.4f}")
print(f"t_sb         : {t_sb:.4f}")
print(f"Durbin-Watson: {durbin:.4f}")
print("\nProfundidad con 3.5 mg/l: "
    f"{b + a * 3.5:.2f}")

# Intervalo de confianza para la estimación.
y_lim_1 = y_reg + t_crit * Se
y_lim_2 = y_reg - t_crit * Se

# Intervalo de confianza para la pendiente.
y_t_1 = ( ( lin_reg.slope - t_sb )
    * ( x - x.mean() ) + y.mean() )
y_t_2 = ( ( lin_reg.slope + t_sb )
    * ( x - x.mean() ) + y.mean() )

# Intervalo de confianza para la media.
s_m = np.sqrt ( Se ** 2 * ( 1 / n +
    ( x - x.mean() ) ** 2 /
    ( ( x - x.mean() ) ** 2 ).sum() ) )
y_m_1 = y_reg + t_crit * s_m
y_m_2 =  y_reg - t_crit * s_m

# Residuales - x
fig, ax = plt.subplots()
ax.scatter(x, res, s = 30)
ax.plot(x, np.zeros_like(x),
    color = "black", linewidth = 1.5)

ax.set_title(ej + "Residuales",
    fontsize = 18)
ax.set_xlabel("x")
ax.set_ylabel("residual")
ax.legend(["_", "residuales"])

fig.savefig(path_r + "Ejercicio_reg_1_res.png")

# Q-Q plot
fig, ax = plt.subplots()
qqplot = stats.probplot(res, plot = ax)
ax.get_lines()[0].set_markersize(5)

ax.set_title(ej + "\nResiduales - Q-Q Plot",
    fontsize = 18)
ax.set_xlabel("normal")
ax.set_ylabel("residual")
ax.legend(["Residuales", "Normal"])

fig.savefig(path_r + "Ejercicio_reg_QQ.png")

print(f"Filliben Q-Q plot test: {qqplot[1][2]:.4f}")

# Se grafican los valores.
fig, ax = plt.subplots()
ax.scatter(x, y, s = 30)
ax.plot(x, y_reg, color = "r", linewidth = 1.5)

# Graficamos el intervalo de confianza.
ax.plot( x, y_lim_1, color = "g",
    linewidth = 1.5 )
ax.plot( x, y_lim_2, color = "g",
    linewidth = 1.5 )
ax.plot( x, y_t_1, color = "black",
    linewidth = 1.5 )
ax.plot( x, y_t_2, color = "black",
    linewidth = 1.5 )
ax.plot( x, y_m_1, color = "brown",
    linewidth = 1.5 )
ax.plot( x, y_m_2, color = "brown",
    linewidth = 1.5 )

ax.set_title(ej,
    fontsize = 18)
ax.set_xlabel("profundidad [m]")
ax.set_ylabel("oxígeno [mg/l]")
ax.legend( ["Regresión lineal", "_",
     "Confianza - media",
    "_", "Confianza - estimación",
    "_", "Confianza - pendiente",
    "Datos" ] )
ax.set_ylim(0, 11)

fig.savefig(path_r + "Ejercicio_reg_1.png")


# Problema 2

x = np.arange( 1, 11, 1 )
y = np.array( [ 0.64, 1.93, 2.21, 2.71, 2.18,
    1.43, 1.01, 0.79, 1.32, 2.8 ] )

# Nivel de significancia.
alpha_test = 0.05
t_crit = stats.t.ppf(1 - alpha_test / 2,
    n - k - 1)

# 2.a.
ej = "2.a. Polinomio de primer grado\n"
YY = np.array([y]).T
x_1 = x
XX = np.array((np.ones_like(x_1), x_1)).T

XX_sq = XX.T @ XX
XX_inv = np.linalg.inv(XX_sq)
BB = XX_inv @ ( XX.T @ YY )
BB.shape

y_reg = ( BB[0,0] + BB[1,0] * x_1 )

# Cantidad de predictores.
k = 1

# Residuales.
res = y - y_reg

# ANOVA
n = x.shape[0]
SST = ( ( y - y.mean() ) ** 2 ).sum()
SSR = ( ( y_reg - y.mean() ) ** 2 ).sum()
SSE = ( ( y - y_reg ) ** 2 ).sum()
Se = np.sqrt(SSE / (y.shape[0] - 2 ))
R = SSR / SST
F = SSR / Se ** 2
F_crit = stats.f.ppf( 1 - alpha_test,
    1, n - 1 - k )
p = stats.f.cdf( F, 1, n - 1 - k )
durbin = durbin_watson(res)

print(ej)
print(f"n            : {n}")
print(f"a_1          : {BB[1,0]:.4f}")
print(f"b            : {BB[0,0]:.4f}")
print(f"X_mean       : {x.mean():.4f}")
print(f"Y_mean       : {y.mean():.4f}")
print(f"SST          : {SST:.4f}")
print(f"SSR          : {SSR:.4f}")
print(f"SSE          : {SSE:.4f}")
print(f"RMSE         : {Se:.4f}")
print(f"R            : {R:.4f}")
print(f"F            : {F:.4f}")
print(f"F_crit       : {F_crit:.4f}")
print(f"p            : {p:.4f}")
print(f"Durbin-Watson: {durbin:.4f}")

# Intervalo de confianza
y_lim_1 = y_reg + t_crit * Se
y_lim_2 = y_reg - t_crit * Se

# Residuales - x
fig, ax = plt.subplots()
ax.scatter(x, res, s = 15)
ax.plot(x, np.zeros_like(x),
    color = "black", linewidth = 1)

ax.set_title(ej + "Residuales - X",
    fontsize = 18)
ax.set_xlabel("x")
ax.set_ylabel("residual")
ax.legend(["_", "residuales"])

fig.savefig( path_r +
    "Ejercicio_reg_2_a_res.png" )

# Q-Q plot
fig, ax = plt.subplots()
qqplot = stats.probplot(res, plot = ax)
ax.get_lines()[0].set_markersize(5)

ax.set_title(ej + "Residuales - Q-Q Plot",
    fontsize = 18)
ax.set_xlabel("normal")
ax.set_ylabel("residual")
ax.legend(["Residuales", "Normal"])

fig.savefig( path_r +
    "Ejercicio_reg_2_a_QQ.png" )

print(f"Filliben Q-Q plot test: {qqplot[1][2]:.4f}")

# Se grafican los valores.
fig, ax = plt.subplots()
ax.scatter(x, y, s = 15)
ax.plot(x, y_reg, color = "r", linewidth = 0.5)

# Graficamos el intervalo de confianza.
ax.fill_between(x, y_lim_1, y_lim_2,
    color = "r", alpha = 0.1)

ax.set_title(ej + "X - Y",
    fontsize = 18)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend( ["Regresión\nlineal",
    "Datos", "Intervalo\nde confianza" ] )

fig.savefig( path_r +
    "Ejercicio_reg_2_a_reg.png" )

# 2.b.
ej = "2.b. Polinomio de grado 2\n"
YY = np.array([y]).T
x_1 = x
x_2 = x ** 2
XX = np.array((np.ones_like(x_1), x_1,
    x_2)).T

XX_sq = XX.T @ XX
XX_inv = np.linalg.inv(XX_sq)
BB = XX_inv @ ( XX.T @ YY )
BB.shape

y_reg = ( BB[0,0] + BB[1,0] * x_1
    + BB[2,0] * x_2 )

# Cantidad de predictores.
k = 2

# Residuales.
res = y - y_reg

# ANOVA
n = x.shape[0]
SST = ( ( y - y.mean() ) ** 2 ).sum()
SSR = ( ( y_reg - y.mean() ) ** 2 ).sum()
SSE = ( ( y - y_reg ) ** 2 ).sum()
Se = np.sqrt(SSE / ( n - k - 1 ))
R = SSR / SST
F = SSR / Se ** 2
F_crit = stats.f.ppf( 1 - alpha_test,
    1, n - 1 - k )
p = stats.f.cdf( F, 1, n - 1 - k )
durbin = durbin_watson(res)

print(ej)
print(f"n            : {n}")
print(f"a_1          : {BB[1,0]:.4f}")
print(f"a_2          : {BB[2,0]:.4f}")
print(f"b            : {BB[0,0]:.4f}")
print(f"X_mean       : {x.mean():.4f}")
print(f"Y_mean       : {y.mean():.4f}")
print(f"SST          : {SST:.4f}")
print(f"SSR          : {SSR:.4f}")
print(f"SSE          : {SSE:.4f}")
print(f"RMSE         : {Se:.4f}")
print(f"R            : {R:.4f}")
print(f"F            : {F:.4f}")
print(f"F_crit       : {F_crit:.4f}")
print(f"p            : {p:.4f}")
print(f"Durbin-Watson: {durbin:.4f}")

# Intervalo de confianza
y_lim_1 = y_reg + t_crit * Se
y_lim_2 = y_reg - t_crit * Se

# Residuales - x
fig, ax = plt.subplots()
ax.scatter(x, res, s = 15)
ax.plot(x, np.zeros_like(x),
    color = "black", linewidth = 1)

ax.set_title(ej + "Residuales - X",
    fontsize = 18)
ax.set_xlabel("x")
ax.set_ylabel("residual")
ax.legend(["_", "residuales"])

fig.savefig( path_r +
    "Ejercicio_reg_2_b_res.png" )

# Q-Q plot
fig, ax = plt.subplots()
qqplot = stats.probplot(res, plot = ax)
ax.get_lines()[0].set_markersize(5)

ax.set_title(ej + "Residuales - Q-Q Plot",
    fontsize = 18)
ax.set_xlabel("normal")
ax.set_ylabel("residual")
ax.legend(["Residuales", "Normal"])

print(f"Filliben Q-Q plot test: {qqplot[1][2]:.4f}")

fig.savefig( path_r +
    "Ejercicio_reg_2_b_QQ.png" )

# Se grafican los valores.
fig, ax = plt.subplots()
ax.scatter(x, y, s = 15)
ax.plot(x, y_reg, color = "r", linewidth = 0.5)

# Graficamos el intervalo de confianza.
ax.fill_between(x, y_lim_1, y_lim_2,
    color = "r", alpha = 0.1)

ax.set_title(ej + "X - Y",
    fontsize = 18)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend( ["Regresión\nlineal",
    "Datos", "Intervalo\nde confianza" ] )

fig.savefig( path_r +
    "Ejercicio_reg_2_b_reg.png" )

# 2.c.
ej = "2.c. Polinomio de grado 3 \n"
YY = np.array([y]).T
x_1 = x
x_2 = x ** 2
x_3 = x ** 3
XX = np.array((np.ones_like(x_1), x_1,
    x_2, x_3)).T

XX_sq = XX.T @ XX
XX_inv = np.linalg.inv(XX_sq)
BB = XX_inv @ ( XX.T @ YY )
BB.shape

y_reg = ( BB[0,0] + BB[1,0] * x_1 
    + BB[2,0] * x_2 + BB[3,0] * x_3
    )

# Cantidad de predictores.
k = 3

# Residuales.
res = y - y_reg

# ANOVA
n = x.shape[0]
SST = ( ( y - y.mean() ) ** 2 ).sum()
SSR = ( ( y_reg - y.mean() ) ** 2 ).sum()
SSE = ( ( y - y_reg ) ** 2 ).sum()
Se = np.sqrt(SSE / ( n - k - 1 ))
R = SSR / SST
F = SSR / Se ** 2
F_crit = stats.f.ppf( 1 - alpha_test,
    1, n - 1 - k )
p = stats.f.cdf( F, 1, n - 1 - k )
durbin = durbin_watson(res)

print(ej)
print(f"n            : {n}")
print(f"a_1          : {BB[1,0]:.4f}")
print(f"a_2          : {BB[2,0]:.4f}")
print(f"a_3          : {BB[3,0]:.4f}")
print(f"b            : {BB[0,0]:.4f}")
print(f"X_mean       : {x.mean():.4f}")
print(f"Y_mean       : {y.mean():.4f}")
print(f"SST          : {SST:.4f}")
print(f"SSR          : {SSR:.4f}")
print(f"SSE          : {SSE:.4f}")
print(f"RMSE         : {Se:.4f}")
print(f"R            : {R:.4f}")
print(f"F            : {F:.4f}")
print(f"F_crit       : {F_crit:.4f}")
print(f"p            : {p:.4f}")
print(f"Durbin-Watson: {durbin:.4f}")

# Intervalo de confianza
y_lim_1 = y_reg + t_crit * Se
y_lim_2 = y_reg - t_crit * Se

# Residuales - x
fig, ax = plt.subplots()
ax.scatter(x, res, s = 15)
ax.plot(x_1, np.zeros_like(x),
    color = "black", linewidth = 1)

ax.set_title(ej + "Residuales - X",
    fontsize = 18)
ax.set_xlabel("x")
ax.set_ylabel("residual")
ax.legend(["_", "residuales"])

fig.savefig( path_r +
    "Ejercicio_reg_2_c_res.png" )

# Q-Q plot
fig, ax = plt.subplots()
qqplot = stats.probplot(res, plot = ax)
ax.get_lines()[0].set_markersize(5)

ax.set_title(ej + "Residuales - Q-Q Plot",
    fontsize = 18)
ax.set_xlabel("normal")
ax.set_ylabel("residual")
ax.legend(["Residuales", "Normal"])

fig.savefig( path_r +
    "Ejercicio_reg_2_a_QQ.png" )

print(f"Filliben Q-Q plot test: {qqplot[1][2]:.4f}")

# Se grafican los valores.
fig, ax = plt.subplots()
ax.scatter(x, y, s = 15)
ax.plot(x, y_reg, color = "r", linewidth = 0.5)

# Graficamos el intervalo de confianza.
ax.fill_between(x, y_lim_1, y_lim_2,
    color = "r", alpha = 0.1)

ax.set_title(ej + "X - Y",
    fontsize = 18)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend( ["Regresión\nlineal",
    "Datos", "Intervalo\nde confianza" ] )

fig.savefig( path_r +
    "Ejercicio_reg_2_c_reg.png" )

# 2.d.
ej = "2.d. Polinomio de grado 6 \n"
YY = np.array([y]).T
x_1 = x
x_2 = x ** 2
x_3 = x ** 3
x_4 = x ** 4
x_5 = x ** 5
x_6 = x ** 6
XX = np.array((np.ones_like(x_1), x_1,
    x_2, x_3, x_4, x_5, x_6)).T

XX_sq = XX.T @ XX
XX_inv = np.linalg.inv(XX_sq)
BB = XX_inv @ ( XX.T @ YY )
BB.shape

y_reg = ( BB[0,0] + BB[1,0] * x_1 
    + BB[2,0] * x_2 + BB[3,0] * x_3
    + BB[4,0] * x_4 + BB[5,0] * x_5 
    + BB[6,0] * x_6 )

# Cantidad de predictores.
k = 6

# Residuales.
res = y - y_reg

# ANOVA
n = x.shape[0]
SST = ( ( y - y.mean() ) ** 2 ).sum()
SSR = ( ( y_reg - y.mean() ) ** 2 ).sum()
SSE = ( ( y - y_reg ) ** 2 ).sum()
Se = np.sqrt(SSE / ( n - k - 1 ))
R = SSR / SST
F = SSR / Se ** 2
F_crit = stats.f.ppf( 1 - alpha_test,
    1, n - 1 - k )
p = stats.f.cdf( F, 1, n - 1 - k )
durbin = durbin_watson(res)

print(ej)
print(f"n            : {n}")
print(f"a_1          : {BB[1,0]:.4f}")
print(f"a_2          : {BB[2,0]:.4f}")
print(f"a_3          : {BB[3,0]:.4f}")
print(f"a_4          : {BB[4,0]:.4f}")
print(f"a_5          : {BB[5,0]:.4f}")
print(f"a_6          : {BB[6,0]:.4f}")
print(f"b            : {BB[0,0]:.4f}")
print(f"X_mean       : {x.mean():.4f}")
print(f"Y_mean       : {y.mean():.4f}")
print(f"SST          : {SST:.4f}")
print(f"SSR          : {SSR:.4f}")
print(f"SSE          : {SSE:.4f}")
print(f"RMSE         : {Se:.4f}")
print(f"R            : {R:.4f}")
print(f"F            : {F:.4f}")
print(f"F_crit       : {F_crit:.4f}")
print(f"p            : {p:.4f}")
print(f"Durbin-Watson: {durbin:.4f}")

# Intervalo de confianza
y_lim_1 = y_reg + t_crit * Se
y_lim_2 = y_reg - t_crit * Se

# Residuales - x
fig, ax = plt.subplots()
ax.scatter(x, res, s = 15)
ax.plot(x, np.zeros_like(x),
    color = "black", linewidth = 1)

ax.set_title(ej + "Residuales - X",
    fontsize = 18)
ax.set_xlabel("x")
ax.set_ylabel("residual")
ax.legend(["_", "residuales"])

fig.savefig( path_r +
    "Ejercicio_reg_2_d_res.png" )

# Q-Q plot
fig, ax = plt.subplots()
qqplot = stats.probplot(res, plot = ax)
ax.get_lines()[0].set_markersize(5)

ax.set_title(ej + "Residuales - Q-Q Plot",
    fontsize = 18)
ax.set_xlabel("normal")
ax.set_ylabel("residual")
ax.legend(["Residuales", "Normal"])

fig.savefig( path_r +
    "Ejercicio_reg_2_a_QQ.png" )

print(f"Filliben Q-Q plot test: {qqplot[1][2]:.4f}")

# Se grafican los valores.
fig, ax = plt.subplots()
ax.scatter(x, y, s = 15)
ax.plot(x, y_reg, color = "r", linewidth = 0.5)

# Graficamos el intervalo de confianza.
ax.fill_between(x, y_lim_1, y_lim_2,
    color = "r", alpha = 0.1)

ax.set_title(ej + "X - Y",
    fontsize = 18)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend( ["Regresión\nlineal",
    "Datos", "Intervalo\nde confianza" ] )

fig.savefig( path_r +
    "Ejercicio_reg_2_d_reg.png" )


# Wilks

# Ejercicio 7.1
# Regresión lineal simple.

# Datos.
path_d = os.getcwd() + "/datos/"
fname = "A.3_Wilks.csv"
df = pd.read_csv(path_d + fname, index_col = "Year")

x = df["Pressure"]
y = df["Temperature"]

k = 1

ej = "7.1. Pressure - Temperature"

alpha_test = 0.05
z = stats.norm.ppf(1 - alpha_test / 2)

# Regresión lineal.
lin_reg = stats.linregress(x, y)
a = lin_reg.intercept
b = lin_reg.slope

# Y gorro.
y_reg = a + b * x

# Residuales.
res = y - y_reg

# ANOVA
n = x.shape[0]
SST = ( ( y - y.mean() ) ** 2 ).sum()
SSR = ( ( y_reg - y.mean() ) ** 2 ).sum()
SSE = ( ( y - y_reg ) ** 2 ).sum()
Se = np.sqrt(SSE / (y.shape[0] - 2 ))
R = SSR / SST
F = SSR / Se ** 2
F_crit = stats.f.ppf( 1 - alpha, 1,
    n - 1 - k )
p = stats.f.cdf( F, 1, n - 1 - k )
t_crit = stats.t.ppf(
    1 - alpha_test/2, n - 1 - k )
S_b = Se / np.sqrt(
    ( ( x - x.mean() ) ** 2 )
    .sum() * (n - 1 - k) )
t_sb = ( t_crit * S_b )
durbin = durbin_watson(res)

# Sginificancia de la pendiente.
ratio_b = b / S_b

# Intervalo de confianza para una estimación.
x_0 = 1013
sm = np.sqrt ( Se ** 2 * ( 1 + 1 / n +
    ( x_0 - x.mean() ) ** 2 /
    ( ( x - x.mean() ) ** 2 ).sum() ) )
prob_sm = 1 - ( 1 - stats.t.cdf(
    1 / sm , n - 1 - k ) ) * 2
prob_se = 1 - ( 1 - stats.t.cdf(
    1 / sm , n - 1 - k ) ) * 2

print(ej)
print(f"n            : {n}")
print(f"a            : {a:.4f}")
print(f"b            : {b:.4f}")
print(f"X_mean       : {x.mean():.4f}")
print(f"Y_mean       : {y.mean():.4f}")
print(f"SST          : {SST:.4f}")
print(f"SSR          : {SSR:.4f}")
print(f"SSE          : {SSE:.4f}")
print(f"RMSE         : {Se:.4f}")
print(f"R            : {R:.4f}")
print(f"F            : {F:.4f}")
print(f"F_crit       : {F_crit:.4f}")
print(f"p            : {p:.4f}")
print(f"t_crit       : {t_crit:.4f}")
print(f"t_sb         : {t_sb:.4f}")
print(f"Durbin-Watson: {durbin:.4f}")
print("\nProfundidad con 3.5 mg/l: "
    f"{b + a * 3.5:.2f}")
print(f"e. {prob_sm:.4f}")
print(f"f. {prob_se:.4f}")
print(f"Siginificancia pendiente: {ratio_b}")

# Intervalo de confianza para la estimación.
y_lim_1 = y_reg + z * Se
y_lim_2 = y_reg - z * Se

# Intervalo de confianza para la pendiente.
y_t_1 = ( ( lin_reg.slope - t_sb )
    * ( x - x.mean() ) + y.mean() )
y_t_2 = ( ( lin_reg.slope + t_sb )
    * ( x - x.mean() ) + y.mean() )

# Intervalo de confianza para la media.
s_m = np.sqrt ( Se ** 2 * ( 1 / n +
    ( x - x.mean() ) ** 2 /
    ( ( x - x.mean() ) ** 2 ).sum() ) )
y_m_1 = y_reg + t_crit * s_m
y_m_2 =  y_reg - t_crit * s_m
y_m_1.name = "y_m_1"
y_m_2.name = "y_m_2"
err_m = pd.concat( (x, y_m_1, y_m_2),
    axis = 1 ).sort_values(x.name)

# Residuales - x
fig, ax = plt.subplots()
ax.scatter(x, res, s = 30)
ax.plot(x, np.zeros_like(x),
    color = "black", linewidth = 1.5)

ax.set_title(ej + "\nResiduales",
    fontsize = 18)
ax.set_xlabel("x")
ax.set_ylabel("residual")
ax.legend(["_", "residuales"])

fig.savefig( path_r +
    "Ejercicio_Wilks_7_1_res.png" )

# Q-Q plot
fig, ax = plt.subplots()
qqplot = stats.probplot(res, plot = ax)
ax.get_lines()[0].set_markersize(5)

ax.set_title(ej + "\nResiduales - Q-Q Plot",
    fontsize = 18)
ax.set_xlabel("normal")
ax.set_ylabel("residual")
ax.legend(["Residuales", "Normal"])

print(f"Filliben Q-Q plot test: {qqplot[1][2]:.4f}")

fig.savefig( path_r +
    "Ejercicio_Wilks_7_1_QQ.png" )

# Se grafican los valores.
fig, ax = plt.subplots()
ax.scatter(x, y, s = 30)
ax.plot(x, y_reg, color = "r", linewidth = 1.5)

# Graficamos el intervalo de confianza.
ax.plot( x, y_lim_1, color = "g",
    linewidth = 1.5 )
ax.plot( x, y_lim_2, color = "g",
    linewidth = 1.5 )
ax.plot( x, y_t_1, color = "black",
    linewidth = 1.5 )
ax.plot( x, y_t_2, color = "black",
    linewidth = 1.5 )
ax.plot( err_m[x.name], err_m["y_m_1"],
    color = "brown", linewidth = 1.5 )
ax.plot( err_m[x.name], err_m["y_m_2"],
    color = "brown", linewidth = 1.5 )

ax.set_title(ej,
    fontsize = 18)
ax.set_xlabel(x.name + " [hPa]")
ax.set_ylabel(y.name + " [°C]")
ax.legend( ["Regresión lineal", "_",
     "Confianza - media",
    "_", "Confianza - estimación",
    "_", "Confianza - pendiente",
    "Datos" ] )
ax.set_ylim(21)

fig.savefig( path_r +
    "Ejercicio_Wilks_7_1_reg.png" )


# Ejercicio 7.2
# Tabla ANOVA.

ej = "7.2. Tabla ANOVA."

n = 27
k = 1
SST = 318.2874
SSR = 316.6065
se = np.sqrt( ( SST - SSR )
    / ( n - k - 1 ) )
b = 0.69

# Desviación de la pendiente.
sb = se * b / np.sqrt( SSR )

# Probabilidad de estar dentro del
# rango de la estimación.
prob = 1 - ( 1 - stats.t.cdf(
    0.2 / se, n - k - 1 ) ) * 2

print(ej)
print(f"prob: {prob:.3f}")
print(f"sb  : {sb:.2f}")
print(f"prueba pendiente: {b / sb:.2f}")


# Ejercicio 7.4
# Regresión no lineal.

# Datos.
path_d = os.getcwd() + "/datos/"
fname = "A.3_Wilks.csv"
df = pd.read_csv(path_d + fname,
    index_col = "Year")
data = df.loc[1956]

ej = "7.4. Regresión no lineal."

# Nivel de significancia.
alpha_test = 0.95
# Mean squared error.
mse = 0.701

# Regresión no lineal.
pre = np.exp( 499.4 - 0.512 * data["Pressure"]
    + 0.796 * data["Temperature"] ) - 1
k = 2

# Valor crítico, t de Student.
t_crit = stats.t.ppf( 1 - alpha_test/2,
    y.shape[0] - 1 - k )

# Intervalo de confianza para la estimación.
conf = [ pre - t_crit * mse,
    pre + t_crit * mse ]

print(ej)
print(f"pre : {pre:.2f}")
print(f"conf: [ {conf[0]:.2f} "
    f", {conf[1]:.2f} ]")
