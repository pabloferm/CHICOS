import numpy as np
import sys

##### EARTH #####
# Earth radius
R_EARTH = 6371.0  # km

# 7.5% difference Earth profile based on the PREM model
sevenpct_earth_profile = [
    {"thickness": 31.854999999999322, "e_density": 1.6754352665999999},
    {"thickness": 382.26000000000033, "e_density": 1.8546798908999995},
    {"thickness": 286.6950000000003, "e_density": 2.1810499762936897},
    {"thickness": 573.3899999999998, "e_density": 2.3478402215199994},
    {"thickness": 668.9549999999999, "e_density": 2.5239652574251483},
    {"thickness": 764.52, "e_density": 2.713891441004218},
    {"thickness": 191.13000000000017, "e_density": 4.616852676057599},
    {"thickness": 541.5350000000001, "e_density": 4.977871343539198},
    {"thickness": 764.52, "e_density": 5.3606627520767995},
    {"thickness": 955.6500000000001, "e_density": 5.945453401104},
]

# 2% difference Earth profile based on the PREM model
twopct_earth_profile = [
    {"thickness": 31.854999999999322, "e_density": 1.6754352665999999},
    {"thickness": 222.9850000000009, "e_density": 1.713426706},
    {"thickness": 127.42000000000012, "e_density": 1.7511445189999997},
    {"thickness": 31.854999999999322, "e_density": 1.8546798908999995},
    {"thickness": 63.71000000000006, "e_density": 1.8944836094999995},
    {"thickness": 63.71000000000006, "e_density": 1.9342873280999995},
    {"thickness": 95.56500000000008, "e_density": 1.9750968219999998},
    {"thickness": 63.71000000000006, "e_density": 2.1810499762936897},
    {"thickness": 159.27500000000015, "e_density": 2.229267825940521},
    {"thickness": 159.27500000000086, "e_density": 2.2759576081510398},
    {"thickness": 191.12999999999874, "e_density": 2.33016987538241},
    {"thickness": 191.13000000000017, "e_density": 2.38263527509352},
    {"thickness": 191.13000000000017, "e_density": 2.4336011979687497},
    {"thickness": 191.13000000000017, "e_density": 2.4833150346924797},
    {"thickness": 222.9849999999995, "e_density": 2.5400620582290654},
    {"thickness": 222.98500000000018, "e_density": 2.59583442684625},
    {"thickness": 222.98500000000018, "e_density": 2.6510249877882086},
    {"thickness": 222.98500000000018, "e_density": 2.7060265882991192},
    {"thickness": 222.9849999999995, "e_density": 4.616852676057599},
    {"thickness": 159.27500000000015, "e_density": 4.7314662806016},
    {"thickness": 159.27500000000015, "e_density": 4.8389409101856},
    {"thickness": 159.27500000000015, "e_density": 4.9395178620096},
    {"thickness": 191.12999999999982, "e_density": 5.051445016012798},
    {"thickness": 191.13000000000017, "e_density": 5.1542039435712},
    {"thickness": 222.98499999999984, "e_density": 5.26305801},
    {"thickness": 254.83999999999986, "e_density": 5.3737265820191995},
    {"thickness": 286.6950000000003, "e_density": 5.481896620588801},
    {"thickness": 350.405, "e_density": 5.592602227363198},
    {"thickness": 286.6949999999999, "e_density": 5.945453401104},
    {"thickness": 700.8100000000001, "e_density": 6.067669476095999},
]

# Usual 4-layer simple Earth profile based on the PREM model
simple_earth_profile = [
    {"thickness": 35, "density": 2.7},
    {"thickness": 2900, "density": 3.3},
    {"thickness": 2200, "density": 10.0},
    {"thickness": 1270, "density": 12.8},
]

# Custom profile based on the PREM model.
def custom_profile(step=1e2, change_rate=None):
    import pandas as pd
    from scipy.interpolate import PchipInterpolator
    fi = pd.read_csv("data/EARTH_MODEL_PREM.dat", sep=" ")
    r = fi["radius"]
    rho = fi["density"]
    ye = fi["Ye"]
    r0 = 0.0
    f = PchipInterpolator(r * R_EARTH, ye * rho)
    del fi
    earth_profile = []
    if change_rate is None:
        sys.exit("Please specify at least step of change rate for computing the density profile.")
    elif change_rate:
        rho0 = f(r0)
        while r0 < R_EARTH:
            r = r0 + step
            if np.abs(f(r) - rho0) / rho0 > change_rate:
                earth_profile.append({"thickness": R_EARTH*(r0-r), "e_density": f(r)})
                r0 = r
                rho0 = f(r)
    return earth_profile



class Earth:
    def __init__(self, label):
        self.profile(label=label)

    def profile(self, label="7p5percent", change_rate=None):
        profile = []
        current_radius = R_EARTH
        if label == "2percent":
            print(r"Using Earth profile with steps at 2% variation in density.")
            for layer in twopct_earth_profile:
                inner_radius = current_radius - layer["thickness"]
                profile.append((inner_radius, current_radius, layer["e_density"]))
                current_radius = inner_radius
        elif label == "7p5percent":
            print(r"Using Earth profile with steps at 7.5% variation in density.")
            for layer in sevenpct_earth_profile:
                inner_radius = current_radius - layer["thickness"]
                profile.append((inner_radius, current_radius, layer["e_density"]))
                current_radius = inner_radius
        elif label == "custom":
            print(f"Using custom Earth profile with at most {change_rate*100}% difference in each step.")
            prof = custom_profile(change_rate=change_rate)
            for layer in prof:
                inner_radius = current_radius - layer["thickness"]
                profile.append((inner_radius, current_radius, layer["e_density"]))
                current_radius = inner_radius
        else:
            print("Simple 4-layer Earth profile.")
            Y_e = 0.5
            for layer in simple_earth_profile:
                inner_radius = current_radius - layer["thickness"]
                profile.append((inner_radius, current_radius, layer["density"] * Y_e))
                current_radius = inner_radius
        self.profile = profile

    def path(self, cos_zentih):
        # compute neutrino path and return ordered segment lengths and densities
        sin_zentih = np.sqrt(1 - cos_zentih**2)
        segments = []
        densities = []
        path_pos = 0

        for inner_r, outer_r, density in self.profile:
            # If the ray does not penetrate to this layer, skip
            if sin_zentih * R_EARTH > outer_r:
                continue
            if sin_zentih * R_EARTH > inner_r:
                continue
            # Calculate chord length inside the shell
            try:
                x1 = 2 * np.sqrt(outer_r**2 - (R_EARTH * sin_zentih) ** 2)
                x2 = 2 * np.sqrt(inner_r**2 - (R_EARTH * sin_zentih) ** 2)
                segments.append(x1 - x2)  # km
                densities.append(density)  # density, g/cm³
            except ValueError:
                # Ray doesn't reach this depth
                continue

        return segments, densities
