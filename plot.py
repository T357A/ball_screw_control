import numpy as np
import matplotlib.pyplot as plt


time_vec = np.load("time_vec.npy")

e_smc = np.load("smc_pos_error.npy")
edot_smc = np.load("smc_vel_error.npy")
eddot_smc = np.load("smc_accel_error.npy")

e_smc_inertia = np.load("smc_pos_error_inertia.npy")
edot_smc_inertia = np.load("smc_vel_error_inertia.npy")
eddot_smc_inertia = np.load("smc_accel_error_inertia.npy")

e_smc_fric = np.load("smc_pos_error_fric.npy")
edot_smc_fric = np.load("smc_vel_error_fric.npy")
eddot_smc_fric = np.load("smc_accel_error_fric.npy")

e_smc_inertia_fric = np.load("smc_pos_error_inertia_fric.npy")
edot_smc_inertia_fric = np.load("smc_vel_error_inertia_fric.npy")
eddot_smc_inertia_fric = np.load("smc_accel_error_inertia_fric.npy")

fig_e, ax_e = plt.subplots()
ax_e.plot(time_vec[:-1], e_smc[:-1] * 1e3, label="Baseline")
ax_e.plot(time_vec[:-1], e_smc_inertia[:-1] * 1e3, label=r"+15% Inertia")
ax_e.plot(time_vec[:-1], e_smc_fric[:-1] * 1e3, label=r"+15% Friction")
ax_e.plot(
    time_vec[:-1], e_smc_inertia_fric[:-1] * 1e3, label=r"+15% Inertia & +15% Friction"
)
ax_e.legend()
ax_e.set(xlabel="Time (s)", ylabel="Error (mm)", title="Position Error")
ax_e.ticklabel_format(style="sci", axis="y", scilimits=(-3, 3))
ax_e.yaxis.major.formatter._useMathText = True

fig_edot, ax_edot = plt.subplots()
ax_edot.plot(time_vec[:-1], edot_smc[:-1] * 1e3, label="Baseline")
ax_edot.plot(time_vec[:-1], edot_smc_inertia[:-1] * 1e3, label=r"+15% Inertia")
ax_edot.plot(time_vec[:-1], edot_smc_fric[:-1] * 1e3, label=r"+15% Friction")
ax_edot.plot(
    time_vec[:-1],
    edot_smc_inertia_fric[:-1] * 1e3,
    label=r"+15% Inertia & +15% Friction",
)
ax_edot.legend()
ax_edot.set(xlabel="Time (s)", ylabel="Error (mm/s)", title="Velocity Error")

fig_eddot, ax_eddot = plt.subplots()
ax_eddot.plot(time_vec[:-1], eddot_smc[:-1] * 1e3, label="Baseline")
ax_eddot.plot(time_vec[:-1], eddot_smc_inertia[:-1] * 1e3, label=r"+15% Inertia")
ax_eddot.plot(time_vec[:-1], eddot_smc_fric[:-1] * 1e3, label=r"+15% Friction")
ax_eddot.plot(
    time_vec[:-1],
    eddot_smc_inertia_fric[:-1] * 1e3,
    label=r"+15% Inertia & +15% Friction",
)
ax_eddot.legend()
ax_eddot.set(xlabel="Time (s)", ylabel=r"Error (mm/s$^2$)", title="Acceleration Error")

plt.show()
