"""
Ordinary differential equation (ODE) solver takes in scalar and vector ODEs
(systems of ODEs).

Works on ODEs of the form

    dx/dt = f(x, u)

which are not explicit functions of time, 't'. The input 'u' is a vector
input function that is time invariant.

Includes the function:  
    - Solver

with methods:            
    - set_initial_condition (takes in either a scalar IC or an array of ICs)
        set_initial_condition(initial_condition)

    - solve (takes in a time step and a final simulation time)
        solve(time_step, end_time)

Different numerical schemes are included as subclasses and functions:
    - ForwardEuler
    - HeunsMethod
    - RungeKutta4
"""

import numpy as np


class Solver(object):
    """
    Class for solving a scalar or vector ODE,

      du/dt = f(u, v)

    by multiple numerical schemes.

    Time vector, t, is implicitly defined.

    Class attributes:
    t: array of time values
    u: array of solution values (at time points t)
    v: array of input values (at time points t)
    k: step number of the most recently computed solution
    f: callable object implementing f(u, v)
    """

    def __init__(self, f):
        """Generates callable function or list of callable functions."""
        self.functions = []
        self.solutions = []
        if isinstance(
            f, list
        ):  # Need to generate a list of solutions for each function
            for function in f:
                if not callable(function):
                    raise TypeError(
                        f"Input {f.index(function)} is {type(function)},\
                             not a function."
                    )
                else:
                    self.functions = self.functions + [function]
        else:
            if not callable(f):
                raise TypeError(f"Input is {type(f)}, not a function")
            # Investiage if I need this lamda function.
            self.f = lambda u, k: np.squeeze(np.asarray(f(u, k)))

    def set_initial_condition(self, U0):
        """Sets initial conditions"""
        if U0 is None:
            raise TypeError(f"Initial condition is {type(U0)}, needs to be a list.")
        if isinstance(U0, (float, int)):  # scalar ODE
            self.neq = 1
            self.U0 = float(U0)
        else:  # system of ODEs
            if isinstance(U0, np.ndarray):
                self.U0 = float(U0)
            else:
                U0 = np.asarray(U0)
                self.neq = U0.size
                self.U0 = U0

    def solve(self, time_step, end_time):
        """Compute u for t values in time vector."""
        self.dt = time_step
        self.t = np.arange(0, end_time, self.dt)
        n = self.t.size
        if self.neq == 1:  # scalar ODE
            self.u = np.zeros(n)
        else:  # systems of ODEs
            self.u = np.zeros((n, self.neq))

        # Assume self.t[0] corresponds to self.U0
        self.u[0] = self.U0

        # Time loop
        if len(self.functions) != 0:
            i = list(range(len(self.functions)))
            for k in range(n - 1):
                self.k = k
                self.solutions[i][k + 1] = self.advance(
                    self.functions[i], self.solutions, i
                )
                # self.u[k + 1] = self.advance(self.functions[i], self.u)
            return self.u, self.t
        else:
            for k in range(n - 1):
                self.k = k
                self.u[k + 1] = self.advance(self.f, self.u, input_fn=None, index=None)
            return self.u, self.t

    def advance(self, f, u, input_fn=None, index=None):
        """Advance solution one time step."""
        raise NotImplementedError


class ForwardEuler(Solver):
    def advance(self, f, u, input_fn=None, index=None):
        u, f, t, k, dt = self.u, self.f, self.t, self.k, self.dt
        if isinstance(u, list):
            u_new = u[index][k] + dt * f[index](u[index - 1 : index], t, k)
        else:
            k = index
            u_new = u[k] + dt * f(u[k], input_fn[k], k)
        return u_new


class HeunsMethod(Solver):
    def advance(self, f, u, input_fn=None, index=None):
        u, f, t, k, dt = self.u, self.f, self.t, self.k, self.dt
        f_k = f(u[k], k)
        u_star = u[k] + dt * f_k
        u_new = u[k] + 0.5 * dt * f_k + 0.5 * dt * f(u_star, k + 1)
        return u_new


class RungeKutta4(Solver):
    def advance(self, f, u, input=None, index=None):
        u, f, t, k, dt = self.u, self.f, self.t, self.k, self.dt
        # dt = t[k + 1] - t[k]
        # dt2 = dt / 2.0
        K1 = dt * f(u[k], k)
        K2 = dt * f(u[k] + 0.5 * K1, k)
        K3 = dt * f(u[k] + 0.5 * K2, k)
        K4 = dt * f(u[k] + K3, k)
        u_new = u[k] + (1 / 6.0) * (K1 + 2 * K2 + 2 * K3 + K4)
        return u_new


# Function implementations of numerical methods
def fwdeuler(f, u, dt, input_fn=None, index=None):
    k = index
    u_new = u[k] + dt * f(u[k], input_fn[k], k)
    return u_new


def heuns(f, u, dt, input_fn=None, index=None):
    k = index
    f_k = f(u[k], input_fn[k], k)
    u_star = u[k] + dt * f_k
    u_new = u[k] + 0.5 * dt * f_k + 0.5 * dt * f(u_star, input_fn[k], k + 1)
    return u_new


def rk4(f, u, dt, input_fn=None, index=None):
    k = index
    # dt = t[k + 1] - t[k]
    # dt2 = dt / 2.0
    K1 = dt * f(u[k], input_fn[k], k)
    K2 = dt * f(u[k] + 0.5 * K1, input_fn[k], k)
    K3 = dt * f(u[k] + 0.5 * K2, input_fn[k], k)
    K4 = dt * f(u[k] + K3, input_fn[k], k)
    u_new = u[k] + (1 / 6.0) * (K1 + 2 * K2 + 2 * K3 + K4)
    return u_new


if __name__ == "__main__":

    def f(x, t, k):
        return x

    # test = ForwardEuler(f)
    # test = HeunsMethod(f)
    test = RungeKutta4(f)
    test.set_initial_condition(1)
    u, t = test.solve(0.2, 4)
    u_exact = np.exp(t)
    error = u_exact - u

    import matplotlib.pyplot as plt

    plt.plot(t, u, label="Numerical")
    plt.plot(t, u_exact, label="Exact")
    plt.plot(t, error, label="Error", color="r")
    plt.legend()
    plt.show()

