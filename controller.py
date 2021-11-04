import numpy as np


class PControl:
    def __init__(self, time_step):

        self.kp = 200.0

        self.dt = time_step
        self.limit = 10.0

    def update(self, r, y):
        e = r - y
        u = self.kp * e
        # u = self.saturate(u)
        return u

    def saturate(self, u):
        if abs(u) > self.limit:
            u = self.limit * np.sign(u)
        return u


class PIControl:
    def __init__(self, time_step):

        self.kp = 600.0
        self.ki = 800.0

        self.dt = time_step
        self.limit = 30.0
        self.integrator = 0.0
        self.error_d1 = 0.0

    def update(self, r, y):
        e = r - y
        I = self.integrate_e(e)
        u_unsat = self.kp * e + self.ki * I
        u_sat = self.saturate(u_unsat)
        if self.ki != 0:
            self.anti_windup(u_unsat, u_sat)
        return u_sat

    def integrate_e(self, e):
        self.integrator = self.integrator + (self.dt / 2) * (e + self.error_d1)
        self.error_d1 = e
        return self.integrator

    def anti_windup(self, u_unsat, u_sat):
        # Integrator anti-windup
        self.integrator = self.integrator + (self.dt / self.ki) * (u_sat - u_unsat)

    def saturate(self, u):
        if abs(u) > self.limit:
            u = self.limit * np.sign(u)
        return u
