import matplotlib as mpl
mpl.use('pgf')
mpl.rcParams.update({
    'pgf.preamble': r'\usepackage{siunitx}',
})

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from uncertainties import unumpy
import scipy.constants.constants as const
from uncertainties import ufloat

data = np.genfromtxt('content/Messung1.txt', unpack=True)
#plot1
x0 = 2
p0 = 1013
p = data[2]
x = x0 * (p/p0)
y = data[0]/120
n= (74271/120)/2

# Fitvorschrift
def f(x, A, B):
    return A*x+B      #jeweilige Fitfunktion auswaehlen:


params, covar = curve_fit(f, x[:-4], y[:-4])            #eigene Messwerte hier uebergeben
uparams = unumpy.uarray(params, np.sqrt(np.diag(covar)))
for i in range(0, len(uparams)):
    print(chr(ord('A') + i), "=" , uparams[i])
print()

lin = np.linspace(x[0], x[-3], 1000)
plt.plot(lin, f(lin, *params), "xkcd:orange", label=r'Regression' )

plt.plot(x, y, 'bx', label=r'Messwerte')
plt.axhline(y = n, label=r'1/2 Zählrate')
plt.axvline(x = 0.93, color='r', label=r'mittlere Reichweite')
plt.ylabel('Zählrate $Z$ / 1/s')
plt.xlabel('effektive Länge $x$ / cm')
plt.legend(loc='best')
plt.grid(True, which='both')
plt.savefig('build/counts1.pdf')
plt.clf()
#plot2
E1 = 4/363
E = data[1] * E1

# Fitvorschrift
def g(x, S, T):
    return S*x + T      #jeweilige Fitfunktion auswaehlen:

params, covar = curve_fit(g, x, E)            #eigene Messwerte hier uebergeben
uparams = unumpy.uarray(params, np.sqrt(np.diag(covar)))
for i in range(0, len(uparams)):
    print(chr(ord('S') + i), "=" , uparams[i])
print()

plt.plot(x, g(x, *params), "xkcd:orange", label=r'Regression' )


plt.plot(x, E, 'bx', label=r'Messwerte')
plt.ylabel('Energie $E$ / MeV')
plt.xlabel('effektive Länge $x$ / cm')
plt.legend(loc='best')
plt.grid(True, which='both')
plt.savefig('build/energie1.pdf')
plt.clf()
data1 = np.genfromtxt('content/Messung2.txt', unpack=True)
#plot3
x1 = 1.5
p = data1[2]
x = x1 * (p/p0)
y = data1[0]/120
n1= (148139/120)/2

# Fitvorschrift
def f(x, A, B):
    return A*x+B      #jeweilige Fitfunktion auswaehlen:


params, covar = curve_fit(f, x[:-4], y[:-4])            #eigene Messwerte hier uebergeben
uparams = unumpy.uarray(params, np.sqrt(np.diag(covar)))
for i in range(0, len(uparams)):
    print(chr(ord('A') + i), "=" , uparams[i])
print()

lin = np.linspace(x[0], 5 , 1000)
plt.plot(lin, f(lin, *params), "xkcd:orange", label=r'Regression' )

plt.plot(x, y, 'bx', label=r'Messwerte')
plt.axhline(y = n, label=r'1/2 Zählrate')
plt.axvline(x = 3.12, color='r', label=r'mittlere Reichweite')
plt.ylabel('Zählrate $Z$ / 1/s')
plt.xlabel('effektive Länge $x$ / cm')
plt.legend(loc='best')
plt.grid(True, which='both')
plt.savefig('build/counts2.pdf')
plt.clf()
#plot4
E1 = 4/448
E = data1[1] * E1

# Fitvorschrift
def g(x, S, T):
    return S*x + T      #jeweilige Fitfunktion auswaehlen:

params, covar = curve_fit(g, x, E)            #eigene Messwerte hier uebergeben
uparams = unumpy.uarray(params, np.sqrt(np.diag(covar)))
for i in range(0, len(uparams)):
    print(chr(ord('S') + i), "=" , uparams[i])
print()

plt.plot(x, g(x, *params), "xkcd:orange", label=r'Regression' )


plt.plot(x, E, 'bx', label=r'Messwerte')
plt.ylabel('Energie $E$ / MeV')
plt.xlabel('effektive Länge $x$ / cm')
plt.legend(loc='best')
plt.grid(True, which='both')
plt.savefig('build/energie2.pdf')
plt.clf()

#poisson
T = 10
counts = np.genfromtxt("content/poisson.txt", unpack = True)
zaehlrate = counts / T
plt.hist(zaehlrate)
plt.xlabel(r"Counts")
plt.ylabel(r"$\mathrm{H\"aufigkeit}$")
plt.legend(loc='best')
plt.savefig("PlotPoisson.pdf")
plt.clf()
