import numpy as np


class DOB:
    def __init__(
        self,
        time_step,
        current_gain,
        inertia,
        fric_coeff,
        torque_coeff=0.2,
        screw_lead=10e-3,
    ):
        self.dt = time_step
        r = screw_lead / (2 * np.pi)
        k = r * current_gain * torque_coeff
        self.je = inertia / k
        self.be = fric_coeff / k
        u0 = 0.0
        udot0 = 0.0
        self.state = np.array([[u0, udot0]])

        alpha0 = 1.0
        alpha1 = 1.0
        tau = 0.1

        n = 2
        a0 = 1.0
        a1 = 2.0
        a2 = 1.0
        b0 = self.je + self.be
        b1 = -2 * self.je
        b2 = self.je - self.be
        # a0 = 4 / self.dt ** 2 + alpha0 / tau + alpha1 / tau ** 2
        # a1 = 2 * alpha0 / tau ** 2 - 8 / self.dt ** 2
        # a2 = 4 / self.dt ** 2 - alpha1 / tau + alpha0 / tau ** 2
        # b0 = alpha0 * (4 * self.je + 2 * self.be * self.dt) / (tau * self.dt) ** 2
        # b1 = -8 * alpha0 * self.je / (tau * self.dt) ** 2
        # b2 = alpha0 * (4 * self.je - 2 * self.be * self.dt) / (tau * self.dt) ** 2
        self.a = [a0, a1, a2]
        self.b = [b0, b1, b2]
        self.w = np.zeros(n + 1)

        self.v = np.zeros(n + 1)
        self.g = np.zeros(2)
        self.h = 0
        self.alpha1 = 0.2
        self.alpha2 = 0.2

    def update(self, x0, x1):
        dt = self.dt
        a1, a2 = self.alpha1, self.alpha2
        g = self.g

        g[0] = x0

        # first derivative
        xdot = a1 * g[0] + (1 - a1) * (x0 - x1) / dt

        # second derivative
        xddot = a2 * g[1] + (1 - a2) * (xdot - g[0]) / dt

        g[0] = xdot
        g[1] = xddot

        # xdot = (x0 - x1) / self.dt
        # xddot = (x0 - 2 * x1 - x2) / (self.dt ** 2)
        u = self.je * xddot + self.be * xdot

        return u

    def updateIIR(self, x):
        # Direct Form 2
        u = self.df2filt(x)

        # Direct Form I
        # a, b, v, w = self.a, self.b, self.v, self.w
        # v[0] = x
        # w[0] = -a[1] * w[1] - a[2] * w[2] + b[0] * v[0] + b[1] * v[1] + b[2] * v[2]
        # u = w[0]
        # v[2] = v[1]
        # w[2] = w[1]
        # v[1] = v[0]
        # w[1] = w[0]
        return u

    def df2filt(self, input):
        """Filters input using Direct Form II algorithm.

        Transfer functions must be of the form H(z) = N(z)/D(z) where:

            N(z) = b0 + b1 * z^-1 + ... + bL * z^-L
            D(z) = 1 + a1 * z^-1 + ... + aM * z^-M

        Args:
            input (float): Filter input signal of type float.

        Returns:
            float: Filtered signal point of type float.
        """
        a, b, w = self.a, self.b, self.w
        n = len(a)
        m = len(b)
        x = input
        w[0] = x

        k = np.amax((n, m))

        for i in range(1, n):
            w[0] = w[0] - a[i] * w[i]

        for i in range(m):
            if i == 0:
                y = b[i] * w[i]
            else:
                y = y + b[i] * w[i]

        for i in range(k - 1, 0, -1):
            w[i] = w[i - 1]

        return y
