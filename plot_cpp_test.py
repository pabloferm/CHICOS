import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

f = pd.read_csv("test_cpp.dat", sep=",")

Pee = f["P(mu->e)"]
Pemu = f["P(mu->mu)"]
Petau = f["P(mu->tau)"]
E = f["E(GeV)"]
L = f["L(km)"]

#plt.hist(Pee, bins=100)
plt.plot(E, Pee)
plt.plot(E, Pemu)
plt.plot(E, Petau)
#plt.xscale("log")
plt.show()