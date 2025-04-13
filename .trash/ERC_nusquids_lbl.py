# coding: utf-8
# # $\nu$-SQuIDS Demo: Welcome!

import nuSQuIDS as nsq
import time

import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np

energy_values = np.geomspace(0.01,10.0,10000)

start_exact = time.time()

# Welcome to the $\nu$-SQuIDS demo. In 
# this notebook we will demostrate some of the functionalities of the $\nu$-SQuIDS' python bindings. All of the calculations performed here can also be done in the C++ interface. Enjoy :)! Carlos, Jordi & Chris.

# # The Basics: single energy mode

# #### Basic definitions

# To start, like in the C++ case, we need to create a $\nu$-SQuIDS object. To begin this demonstration we will create a simple single energy three flavor neutrino oscillation calculator. Thus we just need to specify the number of neutrinos (3) and if we are dealing with neutrinos or antineutrinos.


nuSQ = nsq.nuSQUIDS(3,nsq.NeutrinoType.neutrino)


# nuSQuIDS inputs should be given in natural units. In order to make this convenient we have define a units class called *Const*. We can instanciate it as follows


units = nsq.Const()


# As in the C++ $\nu$-SQuIDS interface one can propagate the neutrinos in various environments (see the documentation for further details), and the user can create and include their own environments. To start a simple case, lets consider oscillactions in <strong> Vacuum </strong>
"""

nuSQ.Set_Body(nsq.Vacuum())


# Since we have specify that we are considering vacuum propagation, we must construct - as in the C++ interface - a *trayectory* inside that object. This can be done using the `Track` property of the given `Body`. Each `Body` will have its on `Track` subclass and its constructors. We can set and construct a <strong>vacuum trayectory</strong> in the following way:

nuSQ.Set_Track(nsq.Vacuum.Track(100.0*units.km))

# Next we have to set the <strong>neutrino energy</strong>, which can be done as follows

nuSQ.Set_E(1.0*units.GeV)

# Now we have to tell $\nu$-SQuIDS what is the **initial neutrino state** and if such state is given in the **flavor or mass basis**. We can do this using the `Set_initial_state` function and providing it with a *list* and a *string*. If the string is *flavor* then list must contain [$\phi_e$,$\phi_\mu$,$\phi_\tau$], similarly if the string is *mass* it the list must specify $\phi_i$. Lets set the initial state to $\nu_\mu$.

nuSQ.Set_initial_state(np.array([0.,1.,0.]),nsq.Basis.flavor)

# Finally we can tell $\nu$-SQuIDS to perform the calculation. In case one (or all) of the above parameters is not set $\nu$-SQuIDS will throw an exception and tell you to fix it, but if you have defined everything then it will evolve the given state.


nuSQ.EvolveState()

# After this runs $\nu$-SQuIDS has evolved the state and store it in memory. Now there are lots of things you can ask $\nu$-SQuIDS to do. What is the flavor composition now?

[nuSQ.EvalFlavor(i) for i in range(3)]

# #### Writing and Reading the state

# $\nu$-SQuIDS knows everything about the neutrino state at the current moment, it also knows what we did with it so far, where it went, what mixing parameters were used, etc. It would be convenient to store this information. One way of doing this is to **save the $\nu$-SQuIDS status**, we can do this in the following way

nuSQ.WriteStateHDF5("current_state.hdf5")

# Everything that is in the $\nu$-SQuIDS object is now in that file. We can use that file to create a new $\nu$-SQuIDS object and do another calculation, we can stop the calculation midway and use it to restart, we can explore that file with other analysis tools, etc. In particular, the `ReadStateHDF5` will return us to the given configuration.

nuSQ.ReadStateHDF5("current_state.hdf5")


# #### Simple Plot

# Lets use the current tool to try to calculate $P(\nu_\mu \to \nu_e)$ as a function of energy. We can do the following

energy_values = np.linspace(1,10,40)
nu_mu_to_nu_e = []
for Enu in energy_values:
    nuSQ.Set_E(Enu*units.GeV)
    nuSQ.Set_initial_state(np.array([0.,1.,0.]),nsq.Basis.flavor)
    nuSQ.EvolveState()
    nu_mu_to_nu_e.append(nuSQ.EvalFlavor(0))

plt.figure(figsize = (8,6))
plt.xlabel(r"$E_\nu [{\rm GeV}]$")
plt.ylabel(r"$P(\nu_\mu \to \nu_e)$")
plt.plot(energy_values,nu_mu_to_nu_e, lw = 2, color = 'blue')

# #### Changing the oscillation parameters

# This is a nice plot. But we do not see an *oscillation* like curve. The reason for this is that the oscillation parameters do not produce an oscillation pattern in this $L/E$ scale. **$\nu$-SQuIDs has some predefined oscillation mixing angles ($\theta_{ij}$) and mass splittings ($\Delta m^2_{ij}$)**, which we have taken from the most recents fits. Perhaps you want to change this parameter to investigate what happens, this can be done easily using the `Set` function.

nuSQ.Set_SquareMassDifference(1,2.0e-1) # sets dm^2_{21} in eV^2.


# We can then, again, do a plot with this simple script

energy_values = np.linspace(1,10,100)
nu_mu_to_nu_e = []
for Enu in energy_values:
    nuSQ.Set_E(Enu*units.GeV)
    nuSQ.Set_initial_state(np.array([0.,1.,0.]),nsq.Basis.flavor)
    nuSQ.EvolveState()
    nu_mu_to_nu_e.append(nuSQ.EvalFlavor(0))

plt.figure(figsize = (8,6))
plt.xlabel(r"$E_\nu [{\rm GeV}]$")
plt.ylabel(r"$P(\nu_\mu \to \nu_e)$")
plt.plot(energy_values,nu_mu_to_nu_e, lw = 2, color = 'blue')


# We can also try to modify the mixing angles (see how to do this in detail in the documentation), for example

nuSQ.Set_MixingAngle(0,1,1.2) # sets \theta_{12} in radians.
nuSQ.Set_MixingAngle(0,2,0.3) # sets \theta_{23}} in radians.
nuSQ.Set_MixingAngle(1,2,0.4) # sets \theta_{23}} in radians.

energy_values = np.linspace(1,10,100)
nu_mu_to_nu_e = []
for Enu in energy_values:
    nuSQ.Set_E(Enu*units.GeV)
    nuSQ.Set_initial_state(np.array([0.,1.,0.]),nsq.Basis.flavor)
    nuSQ.EvolveState()
    nu_mu_to_nu_e.append(nuSQ.EvalFlavor(0))

plt.figure(figsize = (8,6))
plt.xlabel(r"$E_\nu [GeV]$")
plt.ylabel(r"$P(\nu_\mu \to \nu_e)$")
plt.plot(energy_values,nu_mu_to_nu_e, lw = 2, color = 'blue')

# We can now go back to the defaults, which are the values given by Gonzalez-Garcia et al. (arXiv:1409.5439)

nuSQ.Set_MixingParametersToDefault()

"""
# #### Changing where the neutrino propagation takes place

# As in the C++ implementation we can change the `Body` by means of the `Set_Body` function and in similar way we can change the `Track`. Lets do an atmospheric oscillation example =).

nuSQ = nsq.nuSQUIDS(3,nsq.NeutrinoType.neutrino)
nuSQ.Set_Body(nsq.ConstantDensity(13.0,0.5))
nuSQ.Set_Track(nsq.ConstantDensity.Track(295.0*units.km))
nuSQ.Set_rel_error(1.0e-17)
nuSQ.Set_abs_error(1.0e-17)

nuSQ.Set_MixingParametersToDefault()
nuSQ.Set_SquareMassDifference(2,2.5e-3)
nuSQ.Set_MixingAngle(1,2,0.6)
nuSQ.Set_CPPhase(0,2,0.0)

nu_mu_to_nu_e = []
for Enu in energy_values:
    nuSQ.Set_E(Enu*units.GeV)
    nuSQ.Set_initial_state(np.array([0.,1.,0.]),nsq.Basis.flavor)
    nuSQ.EvolveState()
    nu_mu_to_nu_e.append(nuSQ.EvalFlavor(0))
end_exact = time.time()
time_exact = end_exact - start_exact
print(f"{time_exact}")
plt.plot(energy_values,nu_mu_to_nu_e, lw = 2, color = 'blue',label = r"$\nu_\mu$")
plt.legend(fancybox = True, fontsize = 13, loc='upper center', bbox_to_anchor=(0.5, 0.75))
plt.xscale("log")
#fig.tight_layout()
plt.show()
