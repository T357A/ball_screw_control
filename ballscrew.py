import numpy as np


class BallScrewModel:
    """Creates ball screw model object. Includes state space calculation.
    """

    def __init__(
        self,
        time_step,
        current_gain,
        inertia,
        fric_coeff,
        torque_coeff=0.2,
        screw_lead=10e-3,
    ):
        """Initializes model physical parameters and system matrices."""
        self.dt = time_step
        # Initial conditions for each state
        x0 = 0.0
        xdot0 = 0.0
        # Transmission Ratio
        self.r = screw_lead / (2 * np.pi)
        # EoM coefficients
        self.je = inertia / (current_gain * torque_coeff * self.r)
        self.be = fric_coeff / (current_gain * torque_coeff * self.r)
        self.state = np.array([[x0, xdot0]])  # initial condition for x and xdot
        self.limit = 20.0  # 10 Amp limit, could be as high as 20 if needed.

    def f(self, state, u):
        ydot = state.item(1)
        # Equation of motion
        yddot = (-self.be * ydot + u) / self.je
        # Building xdot vector
        xdot = np.array([[ydot, yddot]])
        return xdot

    def h(self):
        # Returns the output y=h(x)
        y1 = self.state.item(0)
        y2 = self.state.item(1)
        y = np.array([[y1, y2]])
        return y

    def update(self, u):
        # u = self.saturate(u)
        self.rk4(u)
        y = self.h()
        return y

    def rk4(self, u):
        K1 = self.f(self.state, u)
        K2 = self.f(self.state + (self.dt / 2 * K1), u)
        K3 = self.f(self.state + (self.dt / 2 * K2), u)
        K4 = self.f(self.state + self.dt * K3, u)
        self.state += (self.dt / 6.0) * (K1 + 2 * K2 + 2 * K3 + K4)

    def saturate(self, u):
        if abs(u) > self.limit:
            u = self.limit * np.sign(u)
        return u

