import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import PchipInterpolator

R_EARTH = 6371.0  # km


def earth_profile(x):
    fi = pd.read_csv("data/EARTH_MODEL_PREM.dat", sep=" ")
    r = fi["radius"]
    rho = fi["density"]
    ye = fi["Ye"]
    f = PchipInterpolator(r * R_EARTH, ye * rho)
    return f(x)


def diff_earth_profile():
    fi = pd.read_csv("data/EARTH_MODEL_PREM.dat", sep=" ")
    r = fi["radius"]
    rho = fi["density"]
    ye = fi["Ye"]
    f = PchipInterpolator(r * R_EARTH, ye * rho)
    return f.derivative()


"""
def f(x):
	return np.sin(x)

def ddf(x):
	return np.cos(x)
"""


def h(df, dx, x0, xf):
    xx = x0
    x = []
    diff_f = diff_earth_profile()
    while xx < xf:
        hh = df / 2 / (1 + np.abs(diff_f(np.array([xx])))) ** 2
        hh = min(hh, 100 * dx)
        hh = max(hh, 0.1 * dx)
        xx += hh
        x.append(xx)
    return np.asarray(x)


df = 10e-1
dx = 1e3


x_ada = h(df, dx, 0, R_EARTH)
print(x_ada.size)
rho = earth_profile(x_ada)


x_lin = np.linspace(0, R_EARTH, 1000)

fi = pd.read_csv("data/EARTH_MODEL_PREM.dat", sep=" ")
r = np.array(fi["radius"])
rho = fi["density"]
ye = fi["Ye"]
r0 = r[-1]
rho0 = 99
count = 0
for i in range(len(r) - 2, -1, -1):
    if np.abs(rho[i] - rho0) / rho0 > 0.075:
        print(f'"thickness": {R_EARTH*(r0-r[i])}, "e_density": {rho[i]*ye[i]}')
        count = count + 1
        r0 = r[i]
        rho0 = rho[i]
print(count)

plt.plot(r * R_EARTH, ye * rho, "-")
plt.plot(x_ada, earth_profile(x_ada), ".", color="orange")
plt.show()


"""
x_ada = h(df, dx, 0, 10)
print(x_ada.size)

x_lin = np.linspace(0, 10, 1000)

plt.plot(x_lin, f(x_lin), "-")
plt.plot(x_ada, f(x_ada), ".", color="orange")
plt.show()
"""
