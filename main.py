import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from signalgen import SignalGeneratorRT, SignalGeneratorFunction
import ballscrew
import controller
import motionprofile

# Time parameter initialization
start_time = 0.0
end_time = 3.0
dt = 1e-3
time_vec = np.arange(start_time, end_time, dt)
n = time_vec.size

# Model class instantiation
screw_lead = 10e-3  # [m]
rg = screw_lead / (2 * np.pi)
current_gain = 6.4898  # [A/V]
torque_coeff = 0.4769  # [N.m/A]
inertia = 7.8715e-3  # [kg.m^2]
fric_coeff = 19.8e-3  # [(kg.m^2)/s]
inertia_alt = inertia * 1.15  # Used for modelling uncertainties
fric_coeff_alt = fric_coeff * 1.15  # Used for modelling uncertainties
bs_model = ballscrew.BallScrewModel(
    dt, current_gain, inertia, fric_coeff, torque_coeff, screw_lead
)
# bs_model = ballscrew.BallScrewModel(
#     dt, current_gain, inertia_alt, fric_coeff_alt, torque_coeff, screw_lead
# )  # Used for modelling uncertainties

## Model limitations
jmax = 50.0  # max jerk [m/s^3]
amax = 2.0  # max acceleration [m/s^2]
vmax = 0.25  # max velocity [m/s]
dmax = 0.5  # max displacement [m]

# P-PI control class instantiation
p_control = controller.PControl(dt)
pi_control = controller.PIControl(dt)

# SMC control initialization
gamma = 200.0  # [rad/s] desired/achievable bandwidth of the drive
ks = 0.8 / 1e-3  # feedback gain
rho = 25.0 / 1e-3  # parameter adaptation gain
kappa = 0.0  # limits of disturbance control action
je = inertia / (current_gain * torque_coeff * rg)
be = fric_coeff / (current_gain * torque_coeff * rg)
a1 = 0.8  # Smoothing coefficient for reference vel
a2 = 0.8  # Smoothing coefficient for reference accel

# Trajectory generation
profile = motionprofile.make_profile(n, jmax, amax, vmax, dmax, dt, end_time)
reference_p = profile[0]  # position reference
reference_v = profile[1]  # velocity reference
reference_a = profile[2]  # acceleration reference

# State, control, and error vectors
## Feed drive vectors
y_ppi = np.zeros((n, 2))
y_ddot = np.zeros(n)
y_smc = np.zeros((n, 2))
yddot_smc = np.zeros(n)

## P-PI control input vecotrs
u_p = np.zeros(n)
u_pi = np.zeros(n)

## SMC control input vecotrs
u_smc = np.zeros(n)

## Disturbance vector
d = SignalGeneratorFunction(0.25, 0.0).step(time_vec) / (current_gain * torque_coeff)
noise = SignalGeneratorFunction(0.0, 0.0, 0.01).random(time_vec)
d_hat = np.zeros(n)  # estimated disturbance for SMC control
d_neg = 0.05  # lower bound on the external disturbance
d_pos = 0.10  # upper bound on the external disturbance

## Tracking error vectors
e_ppi = np.zeros(n)
edot_ppi = np.zeros(n)
eddot_ppi = np.zeros(n)
e_smc = np.zeros(n)
edot_smc = np.zeros(n)
eddot_smc = np.zeros(n)

# Reference signals for filtering
rdot = reference_v.copy()
rddot = reference_a.copy()

# P-PI Control Scheme
for k in range(0, n - 1):
    r = reference_p[k]
    u_p[k] = p_control.update(r, y_ppi[k, 0]) + profile[1, k]  # + vel FFW
    u_pi[k] = pi_control.update(u_p[k], y_ppi[k, 1]) + profile[2, k]  # + accel FFW
    y_ppi[k + 1] = bs_model.update(
        u_pi[k] - d[k] + noise[k]
    )  # disturbance and noise added here
    y_ddot[k] = (-be * y_ppi[k, 1] + u_pi[k]) / je
    e_ppi[k] = r - y_ppi[k, 0]
    edot_ppi[k] = reference_v[k] - y_ppi[k, 1]
    eddot_ppi[k] = reference_a[k] - y_ddot[k]


## NEED TO COMMENT OUT THE ABOVE LOOP BEFORE RUNNING LOOP BELOW ##


# SMC Control Scheme
# for k in range(1, n - 1):
#     r = reference_p[k]
#     rdot[k] = a1 * rdot[k] + (1 - a1) * rdot[k - 1]
#     rddot[k] = a2 * rddot[k] + (1 - a2) * rddot[k - 1]
#     # sliding surface calculation
#     s = gamma * (r - y_smc[k, 0]) + (rdot[k] - y_smc[k, 1])
#     # disturbance estimation
#     d_hat[k] = d_hat[k - 1] + rho * kappa * s * dt
#     if (d_hat[k] <= d_neg) and (s <= 0):
#         kappa = 0.0
#     elif (d_hat[k] >= d_pos) and (s >= 0):
#         kappa = 0.0
#     else:
#         kappa = 1.0
#     # Control input calculation
#     u_smc[k] = (
#         je * (gamma * (rdot[k] - y_smc[k, 1]) + rddot[k])
#         + be * y_smc[k, 1]
#         + d_hat[k]
#         + ks * s
#     )
#     # Dynamics
#     y_smc[k + 1] = bs_model.update(u_smc[k] - d[k] + noise[k])
#     yddot_smc[k] = (-be * y_smc[k, 1] + u_smc[k]) / je
#     # Error calculations
#     e_smc[k] = r - y_smc[k, 0]
#     edot_smc[k] = reference_v[k] - y_smc[k, 1]
#     eddot_smc[k] = reference_a[k] - yddot_smc[k]


# Plots

## P-PI Control Plots
fig0, ax0 = plt.subplots()
ax0.plot(time_vec, y_ppi[:, 0] * 1e3, label="Actual")
ax0.plot(time_vec, reference_p * 1e3, label="Reference")
ax0.legend()
ax0.set(xlabel="Time (s)", ylabel="Position (mm)", title="Position")

fig1, ax1 = plt.subplots()
ax1.plot(time_vec, y_ppi[:, 1] * 1e3, label="Actual")
ax1.plot(time_vec, reference_v * 1e3, label="Reference")
ax1.legend()
ax1.set(xlabel="Time (s)", ylabel="Velocity (mm/s)", title="Velocity")

fig2, ax2 = plt.subplots()
ax2.plot(time_vec, y_ddot * 1e3, label="Actual")
ax2.plot(time_vec, reference_a * 1e3, label="Reference")
ax2.legend()
ax2.set(xlabel="Time (s)", ylabel=r"Acceleration (mm/s$^2$)", title="Acceleration")

fig_e, ax_e = plt.subplots()
ax_e.plot(time_vec[:-1], e_ppi[:-1] * 1e3, label="Error")
# ax_e.legend()
ax_e.set(xlabel="Time (s)", ylabel="Error (mm)", title="Position Error")
ax_e.ticklabel_format(style="sci", axis="y", scilimits=(-3, 3))
ax_e.yaxis.major.formatter._useMathText = True

fig_edot, ax_edot = plt.subplots()
ax_edot.plot(time_vec[:-1], edot_ppi[:-1] * 1e3, label="Error")
# ax_edot.legend()
ax_edot.set(xlabel="Time (s)", ylabel="Error (mm/s)", title="Velocity Error")

fig_eddot, ax_eddot = plt.subplots()
ax_eddot.plot(time_vec[:-1], eddot_ppi[:-1] * 1e3, label="Error")
# ax_eddot.legend()
ax_eddot.set(xlabel="Time (s)", ylabel=r"Error (mm/s$^2$)", title="Acceleration Error")

fig_smc_d, ax_smc_d = plt.subplots()
ax_smc_d.plot(time_vec, d + noise, label="Actual")
# ax_smc_d.legend()
ax_smc_d.set(xlabel="Time (s)", ylabel="Disturbance (V)", title="Disturbance")


## SMC Control Plots
# fig_smc0, ax_smc0 = plt.subplots()
# ax_smc0.plot(time_vec, y_smc[:, 0] * 1e3, label="Actual")
# ax_smc0.plot(time_vec, reference_p * 1e3, label="Reference")
# ax_smc0.legend()
# ax_smc0.set(xlabel="Time (s)", ylabel="Position (mm)", title="Position")

# fig_smc_1, ax_smc_1 = plt.subplots()
# ax_smc_1.plot(time_vec, y_smc[:, 1] * 1e3, label="Actual")
# ax_smc_1.plot(time_vec, reference_v * 1e3, label="Reference")
# ax_smc_1.legend()
# ax_smc_1.set(xlabel="Time (s)", ylabel="Velocity (mm/s)", title="Velocity")

# fig_smc_2, ax_smc_2 = plt.subplots()
# ax_smc_2.plot(time_vec, yddot_smc * 1e3, label="Actual")
# ax_smc_2.plot(time_vec, reference_a * 1e3, label="Reference")
# ax_smc_2.legend()
# ax_smc_2.set(xlabel="Time (s)", ylabel=r"Acceleration (mm/s$^2$)", title="Acceleration")

# fig_smc_e, ax_smc_e = plt.subplots()
# ax_smc_e.plot(time_vec, e_smc * 1e3, label="Error")
# # ax_smc_e.legend()
# ax_smc_e.set(xlabel="Time (s)", ylabel="Error (mm)", title="Position Error")
# ax_smc_e.ticklabel_format(style="sci", axis="y", scilimits=(-3, 3))
# ax_smc_e.yaxis.major.formatter._useMathText = True

# fig_smc_edot, ax_smc_edot = plt.subplots()
# ax_smc_edot.plot(time_vec, edot_smc * 1e3, label="Error")
# # ax_smc_edot.legend()
# ax_smc_edot.set(xlabel="Time (s)", ylabel="Error (mm/s)", title="Velocity Error")

# fig_smc_eddot, ax_smc_eddot = plt.subplots()
# ax_smc_eddot.plot(time_vec, eddot_smc * 1e3, label="Error")
# # ax_smc_eddot.legend()
# ax_smc_eddot.set(
#     xlabel="Time (s)", ylabel=r"Error (mm/s$^2$)", title="Acceleration Error"
# )

# fig_smc_d, ax_smc_d = plt.subplots()
# ax_smc_d.plot(time_vec, d + noise, label="Actual")
# ax_smc_d.plot(time_vec, d_hat, label="Estimated")
# ax_smc_d.legend()
# ax_smc_d.set(xlabel="Time (s)", ylabel="Disturbance (V)", title="Disturbance")


## Reference trajectories
# fig_pos, ax_pos = plt.subplots()
# ax_pos.plot(time_vec, profile[0] * 1e3, label="Reference Position")
# # ax_pos.legend()
# ax_pos.set(xlabel="Time (s)", ylabel="Position (mm)", title="Position Profile")

# fig_vel, ax_vel = plt.subplots()
# ax_vel.plot(time_vec, profile[1] * 1e3, label="Reference Velocity")
# # ax_vel.legend()
# ax_vel.set(xlabel="Time (s)", ylabel="Velcotiy (mm/s)", title="Velocity Profile")

# fig_acc, ax_acc = plt.subplots()
# ax_acc.plot(time_vec, profile[2] * 1e3, label="Reference Acceleration")
# # ax_acc.legend()
# ax_acc.set(
#     xlabel="Time (s)", ylabel=r"Acceleration (mm/s$^2$)", title="Acceleration Profile"
# )

plt.show()
