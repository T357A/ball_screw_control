import numpy as np
from numpy import tan, cos, sin, sqrt


class FilterMethods:
    """Methods are used in filter classes. Methods include Direct Form II
    filtering and Second-Order Section filtering.

    Currently, these methods only apply to any order fitler for df2filt()
    and any number of cascaded 2nd order sections for sosfilt().

    Coefficients are required to be normalized by the leading denominator
    a0 such that a0 becomes 1 (before using this class).
    """

    def __init__(self):
        super().__init__()

    def _listdepth(g, count=0):
        return count if not isinstance(g, list) else max([f(x, count + 1) for x in g])

    def _complexdf2(self, input):
        """Use for future multi-layered direct form II computation.
        """
        depth = _listdepth(self.a)
        if depth > 1:
            pass

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

    def sosfilt(self, input):
        y = input
        for i in range(self.k + 1):
            y = self._sos(y, i)
        return y

    def _sos(self, y, i):
        a, b, w = self.a, self.b, self.w
        w[i][0] = y - a[i][1] * w[i][1] - a[i][2] * w[i][2]
        y = b[i][0] * w[i][0] + b[i][1] * w[i][1] + b[i][2] * w[i][2]
        w[i][2] = w[i][1]
        w[i][1] = w[i][0]
        return y

    def fosfilt(self, input):
        y = input
        for k in range(self.i):
            y = self._fos(y, k)
        return y

    def _fos(self, y, j):
        a, b, w = self.a, self.b, self.w
        # n = len(a[j])
        # m = len(b[j])
        # x = input
        # w[j][0] = x

        # k = np.amax((n, m))

        # for i in range(1, n):
        #     w[j][0] = w[j][0] - a[j][i] * w[j][i]

        # for i in range(m):
        #     if i == 0:
        #         y = b[j][i] * w[j][i]
        #     else:
        #         y = y + b[j][i] * w[j][i]

        # for i in range(k - 1, 0, -1):
        #     w[j][i] = w[j][i - 1]

        w[j][0] = (
            y
            - a[j][1] * w[j][1]
            - a[j][2] * w[j][2]
            - a[j][3] * w[j][3]
            - a[j][4] * w[j][4]
        )
        y = (
            b[j][0] * w[j][0]
            + b[j][1] * w[j][1]
            + b[j][2] * w[j][2]
            + b[j][3] * w[j][3]
            + b[j][4] * w[j][4]
        )
        w[j][4] = w[j][3]
        w[j][3] = w[j][2]
        w[j][2] = w[j][1]
        w[j][1] = w[j][0]
        return y


class LowpassFilter(FilterMethods):
    """First-order lowpass filter.

    Args:
        FilterMethods (class): Methods are used in filter classes. 
        Methods include Direct Form II filtering and Second-Order 
        Section filtering.
    """

    def __init__(self, time_step, cutoff_freq):
        super().__init__()
        self.fs = 1 / time_step
        self.fc = float(cutoff_freq)
        sampling_thrm("LPF", self.fs, self.fc)
        self.m = int(2)  # Order of numerator
        self.n = int(2)  # Order of denominator
        # Coefficients
        self.attenuation = 3.0
        g = 10 ** (-self.attenuation / 20)
        alpha = (g / np.sqrt(1 - g ** 2)) * tan(np.pi * self.fc / self.fs)
        a0 = 1.0
        a1 = -(1 - alpha) / (1 + alpha)
        b0 = (1 - a1) / 2
        b1 = b0
        self.a = [a0, a1]
        self.b = [b0, b1]
        self.w = np.zeros(self.n + 1)


class SecondOrderLPF(FilterMethods):
    def __init__(self, time_step, cutoff_freq):
        pass


class Butterworth(FilterMethods):
    """[summary]

    Args:
        FilterMethods ([type]): [description]
    """

    def __init__(self, time_step, cutoff_freq=400.0):
        super().__init__()
        self.fs = 1 / time_step
        self.fc = float(cutoff_freq)
        sampling_thrm("Butterworth", self.fs, self.fc)  # Checks Nyquist criterion
        self.n = int(8)

        if self.n % 2 != 0:
            raise TypeError(f"Order must be even. Current order, {self.n}, is odd.")

        self.k = int(self.n / 2 - 1)
        self.gamma = 1 / tan(np.pi * self.fc / self.fs)

        self.alpha_k = []
        for i in range(self.k + 1):
            alpha_i = 2 * np.cos(2 * np.pi * (2 * i + self.n + 1) / (4 * self.n))
            self.alpha_k.append(alpha_i)

        self.a = []
        self.b = []
        for i in range(self.k + 1):
            a0 = self.gamma ** 2 - self.alpha_k[i] * self.gamma + 1
            a1 = -(2 * self.gamma ** 2 - 2) / a0
            a2 = (self.gamma ** 2 + self.alpha_k[i] * self.gamma + 1) / a0
            b0 = 1 / a0
            b1 = 2 / a0
            b2 = 1 / a0
            a0 = a0 / a0
            self.a.append(np.array([a0, a1, a2]))
            self.b.append(np.array([b0, b1, b2]))

        self.w = [
            np.zeros(self.k),
            np.zeros(self.k),
            np.zeros(self.k),
            np.zeros(self.k),
        ]

    def cot(self, x):
        return 1 / tan(x)


class Notch(FilterMethods):
    """Filters input data with a 2nd-order narrow-band notch filter.
    Assumes 3.0 dB attenuation.


    Args:
        FilterMethods (class): Methods are used in filter classes. Methods include 
        Direct Form II filtering and Second-Order Section filtering.
    """

    def __init__(
        self, time_step, freq_low=None, freq_hi=None, bandwidth=None, central_freq=None
    ):
        super().__init__()
        self.fs = 1 / float(time_step)
        self.m = int(2)  # Order of numerator
        self.n = int(2)  # Order of denominator
        # Calculate digital central reject freq
        if central_freq == None:
            wo = 2 * np.pi * sqrt(freq_low * freq_hi) / self.fs
        else:
            wo = 2 * np.pi * central_freq / self.fs
        # Calculate digital bandwidth freq
        if bandwidth == None:
            dw = 2 * np.pi * (freq_hi - freq_low) / self.fs
        else:
            dw = 2 * np.pi * bandwidth / self.fs

        self.attenuation = 3.0
        g = 10 ** (-self.attenuation / 20)
        beta = 1 / (1 + (sqrt(1 - g ** 2) / g) * tan(dw / 2))
        # Coefficients
        b0 = beta
        b1 = -2.0 * b0 * cos(wo)
        b2 = b0
        a0 = 1.0
        a1 = -2 * b0 * cos(wo)
        a2 = 2 * b0 - 1
        self.a = [a0, a1, a2]
        self.b = [b0, b1, b2]
        self.w = np.zeros(self.n + 1)


class BandstopFilter(FilterMethods):
    def __init__(self, time_step, pass_a, stop_a, pass_b, stop_b):
        super().__init__()
        if stop_b > pass_b:
            raise TypeError("Passband 'b' must be greater than stopband 'b'.")
        fs = 1 / time_step
        wpa = 2 * np.pi * pass_a / fs
        wsa = 2 * np.pi * stop_a / fs
        wpb = 2 * np.pi * pass_b / fs
        wsb = 2 * np.pi * stop_b / fs
        A_p = 3.0
        A_s = 10.0
        c = sin(wpa + wpb) / (sin(wpa) + sin(wpb))
        omega_p = np.abs(sin(wpb) / (cos(wpb) - c))
        omega_sa = sin(wsa) / (cos(wsa) - c)
        omega_sb = sin(wsb) / (cos(wsb) - c)
        omega_s = np.min((np.abs(omega_sa), np.abs(omega_sb)))
        e = sqrt((10 ** (A_s / 10) - 1) / (10 ** (A_p / 10) - 1))
        w = omega_s / omega_p
        self.n = int(np.ceil(np.log(e) / np.log(w)))
        omega_norm = omega_p / (10 ** (A_p / 10) - 1) ** (1 / (2 * self.n))
        self.i = int(np.floor(self.n / 2))
        theta_i = []
        G_i = []
        self.a = []
        self.b = []
        for k in range(self.i):
            theta_i.append((self.n - 1 + 2 * (k + 1)) * np.pi / (2 * self.n))
            G_i.append(
                omega_norm ** 2
                / (1 - 2 * omega_norm * cos(theta_i[k]) + omega_norm ** 2)
            )
            a_den = 1 - 2 * omega_norm * cos(theta_i[k]) + omega_norm ** 2
            a0 = 1.0
            a1 = 4 * c * omega_norm * (cos(theta_i[k]) - omega_norm) / a_den
            a2 = 2 * (2 * c ** 2 * omega_norm ** 2 + omega_norm ** 2 - 1) / a_den
            a3 = 4 * c * omega_norm * (cos(theta_i[k]) + omega_norm) / a_den
            a4 = (1 + 2 * omega_norm * cos(theta_i[k]) + omega_norm ** 2) / a_den
            b0 = G_i[k]
            b1 = -b0 * 4 * c
            b2 = b0 * (2 + 4 * c ** 2)
            b3 = b1
            b4 = b0
            self.a.append([a0, a1, a2, a3, a4])  # Need to refactor for multi-indexing
            self.b.append([b0, b1, b2, b3, b4])
            # self.a = [a0, a1, a2, a3, a4]
            # self.b = [b0, b1, b2, b3, b4]
        self.w = np.zeros((self.i, len(self.a[0])))


def sampling_thrm(filter_type, fs, fc):
    if fc >= (fs / 2):
        raise TypeError(
            f"{filter_type} sampling frequency of {fs} is too large for a cutoff frequency of {fc}."
        )


if __name__ == "__main__":
    from numpy import random
    import matplotlib.pyplot as plt

    time_step = 1e-4
    time_vec = np.arange(0, 1, time_step)

    fc_1 = 4000.0 / (2 * np.pi)
    fc_2 = 6000.0 / (2 * np.pi)

    x = sin(275 * 2 * np.pi * time_vec) + sin(600 * 2 * np.pi * time_vec)
    y = np.zeros(time_vec.shape)

    butter = Butterworth(time_step)
    notch = Notch(time_step, 300, 500)  # Unstable at 500 Hz for fs = 1000 Hz
    lpf_1 = LowpassFilter(time_step, fc_1)
    lpf_2 = LowpassFilter(time_step, fc_2)
    bsf = BandstopFilter(time_step, 300, 350, 500, 450)  # Unstable, too many coeffs?

    def update(x):
        filt1 = lpf_1.df2filt(x)
        out = lpf_2.df2filt(filt1)
        return out

    for k in range(len(time_vec)):
        # y[k] = update(x[k])  # Test simulates ccloop.py
        y[k] = lpf_1.df2filt(x[k])  # Tests LowpassFilter
        # y[k] = notch.df2filt(x[k])  # Tests Notch
        # y[k] = butter.sosfilt(x[k])  # Tests Butterworth
        # y[k] = bsf.fosfilt(x[k])  # Tests BandstopFilter

    plt.plot(time_vec, x, label="Unfiltered")
    plt.plot(time_vec, y, label="Filtered")
    plt.legend()
    plt.show()
