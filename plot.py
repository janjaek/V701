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
import math

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

mittelwert = np.mean(zaehlrate)
varianz = np.var(zaehlrate)

print("zaehlrateMittelwert = " + str(round(mittelwert, 3)))
print("varianz = " + str(round(varianz, 3)))

balkenzahl = 8
breite = (np.max(zaehlrate) - np.min(zaehlrate)) / balkenzahl
balken = np.array([])

for balkenNummer in range(1, balkenzahl + 1, 1):
        untereSchranke = np.min(zaehlrate) + (balkenNummer - 1) * breite
        obereSchranke = np.min(zaehlrate) + balkenNummer * breite
        balken = np.append(balken, [0])
        for i in range(0, np.size(zaehlrate), 1):
                if (zaehlrate[i] >= untereSchranke) & (zaehlrate[i] < obereSchranke):
                        balken[balkenNummer - 1] += 1
        print("Bereich " + str(balkenNummer) + ": \t" + str(round(untereSchranke, 3)) + "\t bis \t" + str(round(obereSchranke, 3)) + "\t: " + str(balken[balkenNummer - 1]))

k = np.arange(0, balkenzahl, 1)
poissonLambda = 4.
poissonBalken = np.array([])
kPoisson = np.arange(0, balkenzahl + 2, 1)
for i in kPoisson:
        poissonBalken = np.append(poissonBalken, [poissonLambda ** i / math.factorial(i) *
math.exp(-poissonLambda)])


plt.clf()
plt.grid()

plt.bar(k + 1, balken / np.sum(balken), color = "r", width = .4, label=r'Messwerte')
plt.bar(kPoisson + .4, poissonBalken, color = "b", width = .4,
label=r'Piossonverteilung')

#plt.xlim([0, 10])
#plt.ylim([0, .25])
plt.xlabel(r"$\mathrm{H\"aufigkeitsbereich}$")
plt.ylabel(r"$\mathrm{rel.\, H\"aufigkeit}$")
#plt.xlabel("my xlabel")
#plt.ylabel("my ylabel")
plt.legend(loc='best')

#plt.legend(["Gaussfunktion", "Messwerte"], "upper right")

plt.savefig("build/PlotPoisson.pdf")

#gauß
time = 10 #in Sekunden
counts = np.loadtxt('content/poisson.txt', unpack=True)
counts = counts/time

mittelwert = np.mean(counts)
varianz = np.var(counts)

print("zaehlrateMittelwert = " + str(round(mittelwert, 3)))
print("varianz = " + str(round(varianz, 3)))

balkenzahl = 8
breite = (np.max(counts) - np.min(counts)) / balkenzahl
balken = np.array([])

for balkenNummer in range(1, balkenzahl + 1, 1):
        untereSchranke = np.min(counts) + (balkenNummer - 1) * breite
        obereSchranke = np.min(counts) + balkenNummer * breite
        balken = np.append(balken, [0])
        for i in range(0, np.size(counts), 1):
                if (counts[i] >= untereSchranke) & (counts[i] < obereSchranke):
                        balken[balkenNummer - 1] += 1
        print("Bereich " + str(balkenNummer) + ": \t" + str(round(untereSchranke, 3)) + "\t bis \t" + str(round(obereSchranke, 3)) + "\t: " + str(balken[balkenNummer - 1]))

k = np.arange(0, balkenzahl, 1)

x = np.arange(-2, 10, .001)
varianz = 2
mittelwert = 4.5

gaussFunktion = lambda x, A: A / (varianz * np.sqrt(2 * np.pi)) * np.e ** (- .5 *
((x - mittelwert) / varianz) ** 2)

koeffizienten, unsicherheit = curve_fit(gaussFunktion, balkenNummer, balken, maxfev
= 1000)


plt.clf()
plt.grid()

plt.plot(x, gaussFunktion(x, 1), "b-", label=r'Gaußfunktion')
plt.bar(k + 1, balken / np.sum(balken), color = "r", width = .8,label=r'Messwerte')

#plt.hist([zaehlrate], facecolor='green')


#plt.xlim([0, 10])
#plt.ylim([0, .25])
plt.xlabel(r"$\mathrm{H\"aufigkeitsbereich}$")
plt.ylabel(r"$\mathrm{rel.\, H\"aufigkeit}$")
plt.legend(loc='best')


plt.savefig("build/PlotGauss.pdf")
