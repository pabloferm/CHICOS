import numpy as np

##### SUN #####
# Sun radius
R_SUN = 6.9634e5  # km

def solar_density(r, gamma=10, delta=1.28):
    # solar model following https://arxiv.org/pdf/math/9505213
    rho0 = 158 # gcm^−3
    rr = r / R_SUN
    return rho0 * (1 - rr**delta)**gamma

# Custom profile based on the PREM model: TO BE DONE!
def custom_profile(step=1e2, change_rate=None):
    sun_profile = []
    if change_rate is None:
        sys.exit("Please specify at least step of change rate for computing the density profile.")
    elif change_rate:
        r0 = 0.0
        rho0 = solar_density(r0)
        while r0 < R_EARTH:
            r = r0 + step
            if np.abs(solar_density(r) - rho0) / rho0 > change_rate:
                sun_profile.append({"thickness": R_SUN*(r0-r), "e_density": solar_density(r)})
                r0 = r
                rho0 = f(r)
    return sun_profile


class Sun:
    def __init__(self, label):
        self.profile(label=label)

    def profile(self, change_rate=1e2):
        profile = []
        current_radius = 0.0 # center of the Sun
        print(f"Using custom Sun profile with at most {change_rate*100}% difference in each step.")
            prof = custom_profile(change_rate=change_rate)
            for layer in prof:
                inner_radius = current_radius - layer["thickness"]
                profile.append((inner_radius, current_radius, layer["e_density"]))
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
            if sin_zentih * R_SUN > outer_r:
                continue
            if sin_zentih * R_SUN > inner_r:
                continue
            # Calculate chord length inside the shell
            try:
                x1 = 2 * np.sqrt(outer_r**2 - (R_SUN * sin_zentih) ** 2)
                x2 = 2 * np.sqrt(inner_r**2 - (R_SUN * sin_zentih) ** 2)
                segments.append(x1 - x2)  # km
                densities.append(density)  # density, g/cm³
            except ValueError:
                # Ray doesn't reach this depth
                continue

        return segments, densities
