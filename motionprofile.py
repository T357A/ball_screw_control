import numpy as np


def make_profile(
    n, jmax=50.0, amax=2.0, vmax=0.25, dmax=0.5, dt=1e-4, end_time=3.0,
):
    """ This method assumes the 1/3, 1/3, 1/3 trapezoidal velocity profile is used. """
    t_t, dt = end_time, dt  # total time and time step
    t = t_t / 3
    if n % t_t == 0:
        n = int(n)
        index = int(n / 3)
        d = np.zeros(n)
        v = np.zeros(n)
        a_a = np.zeros(index) + vmax / t
        a_c = np.zeros(index)  # this is always zero (constant speed)
        a_d = np.zeros(index) - vmax / t
        for k in range(n):
            if k < int(n / 3):
                d[k] = vmax * (k * dt) ** 2 * 0.5 / t
                v[k] = vmax * k * dt / t
            elif (k >= int(n / 3)) and (k < 2 * int(n / 3)):
                d[k] = vmax * k * dt + t * vmax * (0.5 - t)
                v[k] = vmax
            else:
                d[k] = (
                    -vmax * ((k * dt) ** 2) / (2 * t)
                    + 3 * vmax * k * dt
                    - 5 * t * vmax / 2
                )
                v[k] = -vmax * k * dt / t + 3 * vmax
        a_profile = np.concatenate((a_a, a_c, a_d))
        profile = np.stack((d, v, a_profile))
        return profile
    else:
        raise Exception("Error in profile generator!")

